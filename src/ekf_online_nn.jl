"""
    ekf_online_nn(lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb, meas,
                  dt, itp_mapS, x_nn, m, y_norms, P0, Qd, R;
                  baro_tau   = 3600.0,
                  acc_tau    = 3600.0,
                  gyro_tau   = 3600.0,
                  fogm_tau   = 600.0,
                  date       = get_years(2020,185),
                  core::Bool = false,
                  terms      = [:permanent,:induced,:eddy])

Extended Kalman filter (EKF) with online learning of neural network weights.

**Arguments:**
- `lat`:      latitude  [rad]
- `lon`:      longitude [rad]
- `alt`:      altitude  [m]
- `vn`:       north velocity [m/s]
- `ve`:       east  velocity [m/s]
- `vd`:       down  velocity [m/s]
- `fn`:       north specific force [m/s^2]
- `fe`:       east  specific force [m/s^2]
- `fd`:       down  specific force [m/s^2]
- `Cnb`:      direction cosine matrix (body to navigation) [-]
- `meas`:     scalar magnetometer measurement [nT]
- `dt`:       measurement time step [s]
- `itp_mapS`: scalar map interpolation function (`f(lat,lon)` or `f(lat,lon,alt)`)
- `x_nn`:     `N` x `Nf` data matrix for neural network (`Nf` is number of features)
- `m`:        neural network model, does not work with skip connections
- `y_norms`:  tuple of `y` normalizations, i.e., `(y_bias,y_scale)`
- `P0`:       initial covariance matrix
- `Qd`:       discrete time process/system noise matrix
- `R`:        measurement (white) noise variance
- `baro_tau`: (optional) barometer time constant [s]
- `acc_tau`:  (optional) accelerometer time constant [s]
- `gyro_tau`: (optional) gyroscope time constant [s]
- `fogm_tau`: (optional) FOGM catch-all time constant [s]
- `date`:     (optional) measurement date for IGRF [yr]
- `core`:     (optional) if true, include core magnetic field in measurement
- `terms`:    (optional) Tolles-Lawson terms to use {`:permanent`,`:induced`,`:eddy`,`:bias`}

**Returns:**
- `filt_res`: `FILTres` filter results struct
"""
function ekf_online_nn(lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb, meas,
                       dt, itp_mapS, x_nn, m, y_norms, P0, Qd, R;
                       baro_tau   = 3600.0,
                       acc_tau    = 3600.0,
                       gyro_tau   = 3600.0,
                       fogm_tau   = 600.0,
                       date       = get_years(2020,185),
                       core::Bool = false)

    (y_bias,y_scale) = y_norms

    N     = length(lat)
    nx    = size(P0,1)
    nx_nn = sum(length,Flux.params(m))
    ny    = size(meas,2)
    x_out = zeros(eltype(P0),nx,N)
    P_out = zeros(eltype(P0),nx,nx,N)
    r_out = zeros(eltype(P0),ny,N)
    x     = zeros(eltype(P0),nx) # state estimate
    P     = P0 # covariance matrix

    (w0_nn,re) = destructure(m)
    x[end-nx_nn:end-1] .= w0_nn

    map_cache = itp_mapS isa Map_Cache ? itp_mapS : nothing

    for t = 1:N
        # custom itp_mapS from map cache, if available
        if map_cache isa Map_Cache
            itp_mapS = get_cached_map(map_cache,lat[t],lon[t],alt[t];silent=true)
        end

        # Pinson matrix exponential
        Phi = get_Phi(nx,lat[t],vn[t],ve[t],vd[t],fn[t],fe[t],fd[t],Cnb[:,:,t],
                      baro_tau,acc_tau,gyro_tau,fogm_tau,dt)

        # measurement residual [ny]
        m     = re(x[end-nx_nn:end-1])
        resid = meas[t,:] .- (m(x_nn[t,:]).*y_scale.+y_bias) .-
                get_h(itp_mapS,x,lat[t],lon[t],alt[t];date=date,core=core)

        # measurement Jacobian (repeated gradient here) [ny x nx]
        Hll = get_H(itp_mapS,x,lat[t],lon[t],alt[t];date=date,core=core)'
        Hnn = get_Hnn(nn_grad(m,x_nn[t,:]))
        H1  = [Hll[1:2]; zeros(eltype(Hll),nx-3-nx_nn); Hnn.*y_scale; 1]
        H   = repeat(H1',ny,1)

        # measurement residual covariance
        S = H*P*H' .+ R         # S_t [ny x ny]

        # Kalman gain
        K = (P*H') / S          # K_t [nx x ny]

        # state and covariance update
        x = x + K*resid         # x_t [nx]
        P = (I - K*H) * P   # P_t [nx x nx]

        # state, covariance, and residual store
        x_out[:,t]   = x
        P_out[:,:,t] = P
        r_out[:,t]   = resid

        # state and covariance propagate (predict)
        x = Phi*x               # x_t|t-1 [nx]
        P = Phi*P*Phi' + Qd     # P_t|t-1 [nx x nx]
    end

    return FILTres(x_out, P_out, r_out, true)
end # function ekf_online_nn

"""
    nn_grad(m::Chain,x)

Internal helper function to get neural network gradients.

**Arguments:**
- `m`: neural network model
- `x`: `N` x `Nf` data matrix for neural network (`Nf` is number of features)

**Returns:**
- `g`: tuple of gradients of `m` using `x`
"""
nn_grad(m::Chain,x) = gradient(m -> sum(m(x)), m)[1].layers # function nn_grad

"""
    get_Hnn(g::Tuple)

Internal helper function to reshape neural network gradients.

**Arguments:**
- `g`: tuple of gradients

**Returns:**
- `Hnn`: vector of gradients
"""
function get_Hnn(g::Tuple)
    Hnn = Float32[]
    for i in eachindex(g)
        Hnn = [Hnn;vec(g[i].weight);]
        !(g[i].bias isa Nothing) && (Hnn = [Hnn;vec(g[i].bias);])
    end
    return (Hnn)
end # function get_Hnn

"""
    ekf_online_nn(ins::INS, meas, itp_mapS, x_nn, m, y_norms, P0, Qd, R;
                  baro_tau   = 3600.0,
                  acc_tau    = 3600.0,
                  gyro_tau   = 3600.0,
                  fogm_tau   = 600.0,
                  date       = get_years(2020,185),
                  core::Bool = false)

Extended Kalman filter (EKF) with online learning of neural network weights.

**Arguments:**
- `ins`:      `INS` inertial navigation system struct
- `meas`:     scalar magnetometer measurement [nT]
- `itp_mapS`: scalar map interpolation function (`f(lat,lon)` or `f(lat,lon,alt)`)
- `x_nn`:     `N` x `Nf` data matrix for neural network (`Nf` is number of features)
- `m`:        neural network model, does not work with skip connections
- `y_norms`:  tuple of `y` normalizations, i.e., `(y_bias,y_scale)`
- `P0`:       initial covariance matrix
- `Qd`:       discrete time process/system noise matrix
- `R`:        measurement (white) noise variance
- `baro_tau`: (optional) barometer time constant [s]
- `acc_tau`:  (optional) accelerometer time constant [s]
- `gyro_tau`: (optional) gyroscope time constant [s]
- `fogm_tau`: (optional) FOGM catch-all time constant [s]
- `date`:     (optional) measurement date for IGRF [yr]
- `core`:     (optional) if true, include core magnetic field in measurement

**Returns:**
- `filt_res`: `FILTres` filter results struct
"""
function ekf_online_nn(ins::INS, meas, itp_mapS, x_nn, m, y_norms, P0, Qd, R;
                       baro_tau   = 3600.0,
                       acc_tau    = 3600.0,
                       gyro_tau   = 3600.0,
                       fogm_tau   = 600.0,
                       date       = get_years(2020,185),
                       core::Bool = false)
    ekf_online_nn(ins.lat,ins.lon,ins.alt,ins.vn,ins.ve,ins.vd,
                  ins.fn,ins.fe,ins.fd,ins.Cnb,meas,
                  ins.dt,itp_mapS,x_nn,m,y_norms,P0,Qd,R;
                  baro_tau = baro_tau,
                  acc_tau  = acc_tau,
                  gyro_tau = gyro_tau,
                  fogm_tau = fogm_tau,
                  date     = date,
                  core     = core)
end # function ekf_online_nn

# # old attempt
# function ekf_online_nn_setup(x, y, m, y_norms; N_sigma::Int=1000)

#     (y_bias,y_scale) = y_norms # unpack normalizations

#     m    = deepcopy(m) # don't modify original NN model
#     w_nn = destructure(m)[1] # weights
#     opt  = Adam() # optimizer
#     loss(x,y) = mse(m(x),y) # mean squared error

#     N        = length(y)
#     N_sigma  = min(N,N_sigma)
#     coef_set = zeros(eltype(w_nn),length(w_nn),N_sigma)
#     for i = 1:N_sigma
#         ind = ((i-1)*floor(Int,(N-1)/N_sigma)).+(1:1)
#         # ind = (i-1)*bS+1:i*bS
#         d = DataLoader((x[ind,:]',y[ind]'),shuffle=true,batchsize=1) # bs
#         Flux.train!(loss,Flux.params(m),d,opt)
#         coef_set[:,i] = deepcopy(destructure(m)[1])
#     end

#     # nn_sigma = vec(std(coef_set,dims=2))
#     nn_sigma = vec(abs.(median(coef_set[:,2:end]-coef_set[:,1:end-1],dims=2)))

#     P0_nn = Diagonal(nn_sigma.^2)

#     return (P0_nn, nn_sigma) # , w_nn_store)
# end # function ekf_online_nn_setup

"""
    ekf_online_nn_setup(x, y, m, y_norms; N_sigma::Int=1000)

Setup for extended Kalman filter (EKF) with online learning of neural network weights.

**Arguments:**
- `x`:       `N` x `Nf` data matrix (`Nf` is number of features)
- `y`:       length `N` target vector
- `m`:       neural network model, does not work with skip connections
- `y_norms`: tuple of `y` normalizations, i.e., `(y_bias,y_scale)`
- `N_sigma`: (optional) number of neural network weights sets to use to create `nn_sigma`

**Returns:**
- `P0_nn`:    initial neural network weights covariance matrix
- `nn_sigma`: initial neural network weights estimate std dev
"""
function ekf_online_nn_setup(x, y, m, y_norms; N_sigma::Int=1000)
    N_sigma_min = 10
    @assert N_sigma >= N_sigma_min "increase N_sigma to $N_sigma_min"
    (y_bias,y_scale) = y_norms # unpack normalizations
    m = deepcopy(m) # don't modify original NN model
    (w_nn,re)  = destructure(m) # weights, restructure
    w_nn_store = zeros(eltype(w_nn),length(w_nn),N_sigma) # initialize weights matrix
    P = I(length(w_nn)) # initialize covariance matrix
    for i = 1:N_sigma   # run recursive least squares
        m = re(w_nn)
        K = P*w_nn/(1+w_nn'*P*w_nn)
        P     -= P*w_nn*w_nn'*P/(1+w_nn'*P*w_nn)
        w_nn  += K.*(y[i].-(m(x[i,:]).*y_scale.+y_bias)) ./ y_scale
        w_nn_store[:,i] = w_nn
        # println(w_nn[1])
    end

    P0_nn = abs.(P) # make sure P is positive
    # nn_sigma = sqrt.(diag(P0_TL))
    # nn_sigma = vec(abs.(median(w_nn_store[:,2:end]-w_nn_store[:,1:end-1],dims=2)))
    # nn_sigma = vec(std(abs.(w_nn_store[:,2:5]-w_nn_store[:,1:4]),dims=2))
    nn_sigma = vec(minimum(abs.(w_nn_store[:,2:end]-w_nn_store[:,1:end-1]),dims=2))
    # nn_sigma = vec(abs.(w_nn_store[:,2]-w_nn_store[:,1]))

    return (P0_nn, nn_sigma) # , w_nn_store)
end # function ekf_online_nn_setup
