"""
    nekf(lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb, meas, dt, itp_mapS,
         x_nn::Matrix = meas[:,:],
         m            = Dense(1 => 1);
         P0           = create_P0(),
         Qd           = create_Qd(),
         R            = 1.0,
         baro_tau     = 3600.0,
         acc_tau      = 3600.0,
         gyro_tau     = 3600.0,
         fogm_tau     = 600.0,
         date         = get_years(2020,185),
         core::Bool   = false)

Measurement noise covariance-adaptive neural extended Kalman filter (nEKF)
for airborne magnetic anomaly navigation.

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
- `m`:        neural network model
- `P0`:       (optional) initial covariance matrix
- `Qd`:       (optional) discrete time process/system noise matrix
- `R`:        (optional) measurement (white) noise variance
- `baro_tau`: (optional) barometer time constant [s]
- `acc_tau`:  (optional) accelerometer time constant [s]
- `gyro_tau`: (optional) gyroscope time constant [s]
- `fogm_tau`: (optional) FOGM catch-all time constant [s]
- `date`:     (optional) measurement date (decimal year) for IGRF [yr]
- `core`:     (optional) if true, include core magnetic field in measurement

**Returns:**
- `filt_res`: `FILTres` filter results struct
"""
function nekf(lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb, meas, dt, itp_mapS,
              x_nn::Matrix = meas[:,:],
              m            = Dense(1 => 1);
              P0           = create_P0(),
              Qd           = create_Qd(),
              R            = 1.0,
              baro_tau     = 3600.0,
              acc_tau      = 3600.0,
              gyro_tau     = 3600.0,
              fogm_tau     = 600.0,
              date         = get_years(2020,185),
              core::Bool   = false)

    N      = length(lat)
    nx     = size(P0,1)
    ny     = size(meas,2)
    x_out  = zeros(eltype(P0),nx,N)
    P_out  = zeros(eltype(P0),nx,nx,N)
    r_out  = zeros(eltype(P0),ny,N)
    R_nn   = m(x_nn') # pre-compute R correction

    x = zeros(eltype(P0),nx) # state estimate
    P = P0 # covariance matrix
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
        resid = meas[t,:] .- get_h(itp_mapS,x,lat[t],lon[t],alt[t];
                                   date=date,core=core)

        # measurement Jacobian (repeated gradient here) [ny x nx]
        H = repeat(get_H(itp_mapS,x,lat[t],lon[t],alt[t];
                         date=date,core=core)',ny,1)

        # measurement residual covariance
        S = H*P*H' .+ R .* (1 + R_nn[t]) # S_t [ny x ny]

        # Kalman gain
        K = (P*H') / S          # K_t [nx x ny]

        # state & covariance update
        x = x + K*resid         # x_t [nx]
        P = (I - K*H) * P       # P_t [nx x nx]

        # state, covariance, & residual store
        x_out[:,t]   = x
        P_out[:,:,t] = P
        r_out[:,t]   = resid

        # state & covariance propagate (predict)
        x = Phi*x               # x_t|t-1 [nx]
        P = Phi*P*Phi' + Qd     # P_t|t-1 [nx x nx]
    end

    println("R_nn at [1, N/2, N]: ",round.(R_nn[[1,round(Int,N/2),N]],digits=5))

    return FILTres(x_out, P_out, r_out, true)
end # function nekf

"""
    nekf(ins::INS, meas, itp_mapS,
         x_nn::Matrix = meas[:,:],
         m            = Dense(1 => 1);
         P0           = create_P0(),
         Qd           = create_Qd(),
         R            = 1.0,
         baro_tau     = 3600.0,
         acc_tau      = 3600.0,
         gyro_tau     = 3600.0,
         fogm_tau     = 600.0,
         date         = get_years(2020,185),
         core::Bool   = false)

Measurement noise covariance-adaptive neural extended Kalman filter (nEKF)
for airborne magnetic anomaly navigation.

**Arguments:**
- `ins`:      `INS` inertial navigation system struct
- `meas`:     scalar magnetometer measurement [nT]
- `itp_mapS`: scalar map interpolation function (`f(lat,lon)` or `f(lat,lon,alt)`)
- `x_nn`:     `N` x `Nf` data matrix for neural network (`Nf` is number of features)
- `m`:        neural network model
- `P0`:       (optional) initial covariance matrix
- `Qd`:       (optional) discrete time process/system noise matrix
- `R`:        (optional) measurement (white) noise variance
- `baro_tau`: (optional) barometer time constant [s]
- `acc_tau`:  (optional) accelerometer time constant [s]
- `gyro_tau`: (optional) gyroscope time constant [s]
- `fogm_tau`: (optional) FOGM catch-all time constant [s]
- `date`:     (optional) measurement date (decimal year) for IGRF [yr]
- `core`:     (optional) if true, include core magnetic field in measurement

**Returns:**
- `filt_res`: `FILTres` filter results struct
"""
function nekf(ins::INS, meas, itp_mapS,
              x_nn::Matrix = meas[:,:],
              m            = Dense(1 => 1);
              P0           = create_P0(),
              Qd           = create_Qd(),
              R            = 1.0,
              baro_tau     = 3600.0,
              acc_tau      = 3600.0,
              gyro_tau     = 3600.0,
              fogm_tau     = 600.0,
              date         = get_years(2020,185),
              core::Bool   = false)
    nekf(ins.lat,ins.lon,ins.alt,ins.vn,ins.ve,ins.vd,
         ins.fn,ins.fe,ins.fd,ins.Cnb,meas,ins.dt,itp_mapS,x_nn,m;
         P0       = P0,
         Qd       = Qd,
         R        = R,
         baro_tau = baro_tau,
         acc_tau  = acc_tau,
         gyro_tau = gyro_tau,
         fogm_tau = fogm_tau,
         date     = date,
         core     = core)
end # function nekf

"""
    ekf_single(lat, lon, alt, Phi, meas, itp_mapS,
               P            = create_P0(),
               Qd           = create_Qd(),
               R            = 1.0,
               R_nn         = 0.0,
               x            = zeros(eltype(P),18);
               date         = get_years(2020,185),
               core::Bool   = false)

Internal helper function to run an extended Kalman filter (EKF) for a single
time step with a pre-computed `Phi` dynamics matrix.

**Arguments:**
- `lat`:      latitude  [rad]
- `lon`:      longitude [rad]
- `alt`:      altitude  [m]
- `meas`:     scalar magnetometer measurement [nT]
- `itp_mapS`: scalar map interpolation function (`f(lat,lon)` or `f(lat,lon,alt)`)
- `P`:        non-linear covariance matrix
- `Qd`:       discrete time process/system noise matrix
- `R`:        measurement (white) noise variance
- `R_nn`:     measurement (white) noise variance scaling factor from neural network
- `x`:        filtered states, i.e., E(x_t | y_1,..,y_t)
- `date`:     (optional) measurement date (decimal year) for IGRF [yr]
- `core`:     (optional) if true, include core magnetic field in measurement

**Returns:**
- `P`: non-linear covariance matrix
- `x`: filtered states, i.e., E(x_t | y_1,..,y_t)
"""
function ekf_single(lat, lon, alt, Phi, meas, itp_mapS,
                    P            = create_P0(),
                    Qd           = create_Qd(),
                    R            = 1.0,
                    R_nn         = 0.0,
                    x            = zeros(eltype(P),18);
                    date         = get_years(2020,185),
                    core::Bool   = false)

    @assert !(itp_mapS isa Map_Cache) "Map_Cache not supported for nEKF training"

    ny = length(meas)

    # measurement residual [ny]
    resid = meas .- get_h(itp_mapS,x,lat,lon,alt;date=date,core=core)

    # measurement Jacobian (repeated gradient here) [ny x nx]
    H = repeat(get_H(itp_mapS,x,lat,lon,alt;date=date,core=core)',ny,1)

    # measurement residual covariance
    S = H*P*H' .+ R .* (1 + R_nn) # S_t [ny x ny]

    # Kalman gain
    K = (P*H') / S          # K_t [nx x ny]

    # state & covariance update
    x = x + K*resid         # x_t [nx]
    P = (I - K*H) * P       # P_t [nx x nx]

    # state & covariance propagate (predict)
    x = Phi*x               # x_t|t-1 [nx]
    P = Phi*P*Phi' + Qd     # P_t|t-1 [nx x nx]

    return (P, x)
end # function ekf_single

"""
    nekf_train(lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb, meas, dt,
               itp_mapS, x_nn::Matrix, y_nn::Matrix;
               P0                   = create_P0(),
               Qd                   = create_Qd(),
               R                    = 1.0,
               baro_tau             = 3600.0,
               acc_tau              = 3600.0,
               gyro_tau             = 3600.0,
               fogm_tau             = 600.0,
               η_adam               = 0.1,
               epoch_adam::Int      = 10,
               hidden::Int          = 1,
               activation::Function = swish,
               l_window::Int        = 50,
               date                 = get_years(2020,185),
               core::Bool           = false)

Train a measurement noise covariance-adaptive neural extended Kalman filter
(nEKF) model for airborne magnetic anomaly navigation.

**Arguments:**
- `lat`:        latitude  [rad]
- `lon`:        longitude [rad]
- `alt`:        altitude  [m]
- `vn`:         north velocity [m/s]
- `ve`:         east  velocity [m/s]
- `vd`:         down  velocity [m/s]
- `fn`:         north specific force [m/s^2]
- `fe`:         east  specific force [m/s^2]
- `fd`:         down  specific force [m/s^2]
- `Cnb`:        direction cosine matrix (body to navigation) [-]
- `meas`:       scalar magnetometer measurement [nT]
- `dt`:         measurement time step [s]
- `itp_mapS`:   scalar map interpolation function (`f(lat,lon)` or `f(lat,lon,alt)`)
- `x_nn`:       `N` x `Nf` data matrix for neural network (`Nf` is number of features)
- `y_nn`:       `y` target matrix for neural network (`[latitude longitude]`)
- `P0`:         (optional) initial covariance matrix
- `Qd`:         (optional) discrete time process/system noise matrix
- `R`:          (optional) measurement (white) noise variance
- `baro_tau`:   (optional) barometer time constant [s]
- `acc_tau`:    (optional) accelerometer time constant [s]
- `gyro_tau`:   (optional) gyroscope time constant [s]
- `fogm_tau`:   (optional) FOGM catch-all time constant [s]
- `η_adam`:     (optional) learning rate for Adam optimizer
- `epoch_adam`: (optional) number of epochs for Adam optimizer
- `hidden`:     (optional) hidden layers & nodes (e.g., `[8,8]` for 2 hidden layers, 8 nodes each)
- `activation`: (optional) activation function
    - `relu`  = rectified linear unit
    - `σ`     = sigmoid (logistic function)
    - `swish` = self-gated
    - `tanh`  = hyperbolic tan
    - run `plot_activation()` for a visual
- `l_window`:   (optional) temporal window length
- `date`:       (optional) measurement date (decimal year) for IGRF [yr]
- `core`:       (optional) if true, include core magnetic field in measurement

**Returns:**
- `m`: neural network model
"""
function nekf_train(lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb, meas, dt,
                    itp_mapS, x_nn::Matrix, y_nn::Matrix;
                    P0                   = create_P0(),
                    Qd                   = create_Qd(),
                    R                    = 1.0,
                    baro_tau             = 3600.0,
                    acc_tau              = 3600.0,
                    gyro_tau             = 3600.0,
                    fogm_tau             = 600.0,
                    η_adam               = 0.1,
                    epoch_adam::Int      = 10,
                    hidden::Int          = 1,
                    activation::Function = swish,
                    l_window::Int        = 50,
                    date                 = get_years(2020,185),
                    core::Bool           = false)

    (N,Nf) = size(x_nn) # number of samples (instances) & features
    Ny = 1 # length of output
    m  = Chain(LSTM(Nf => hidden), Dense(hidden => Ny, activation))

    x_seqs = chunk_data(Float32.(x_nn),zero(lat),l_window)[1]
    y_seqs = chunk_data(Float32.(y_nn),zero(lat),l_window)[1]
    N_seqs = [[N_nn;;] for N_nn in l_window:l_window:N]

    # pre-compute Phi
    N   = length(lat)
    Phi = zeros(Float64,18,18,N)
    for t = 1:N
        Phi[:,:,t] = get_Phi(size(Phi)[1],lat[t],vn[t],ve[t],vd[t],
                             fn[t],fe[t],fd[t],Cnb[:,:,t],
                             baro_tau,acc_tau,gyro_tau,fogm_tau,dt)
    end

    P = P0

    # setup loss function
    function loss(m, x_nn, y_nn, N_nn)
        N = size(x_nn,2)
        x = zero(P[:,1])

        R_nn = m(x_nn) # pre-compute R correction

        l = 0
        for i = 1:N
            t = N_nn[1] - N + i
            (P,x) = ekf_single(lat[t],lon[t],alt[t],Phi[:,:,t],meas[t],
                               itp_mapS,P,Qd,R,R_nn[i],x;date=date,core=core)

            l += dlat2dn(y_nn[1,i] - (lat[t]+x[1]), lat[t]+x[1])^2 +
                 dlon2de(y_nn[2,i] - (lon[t]+x[2]), lat[t]+x[1])^2
        end

        return sqrt(l/N) # DRMS
    end # function loss

    # setup Adam optimizer
    opt = Flux.setup(Adam(η_adam),m)

    # train RNN with Adam optimizer
    for _ = 1:epoch_adam
        Flux.train!(loss,m,zip(x_seqs,y_seqs,N_seqs),opt)
        println("loss: ",sum(loss.((m,),x_seqs,y_seqs,N_seqs)))
    end

    return (m)
end # function nekf_train

"""
    nekf_train(ins::INS, meas, itp_mapS, x_nn::Matrix, y_nn::Matrix;
               P0                   = create_P0(),
               Qd                   = create_Qd(),
               R                    = 1.0,
               baro_tau             = 3600.0,
               acc_tau              = 3600.0,
               gyro_tau             = 3600.0,
               fogm_tau             = 600.0,
               η_adam               = 0.1,
               epoch_adam::Int      = 10,
               hidden::Int          = 1,
               activation::Function = swish,
               l_window::Int        = 50,
               date                 = get_years(2020,185),
               core::Bool           = false)

Train a measurement noise covariance-adaptive neural extended Kalman filter
(nEKF) model for airborne magnetic anomaly navigation.

**Arguments:**
- `ins`:        `INS` inertial navigation system struct
- `meas`:       scalar magnetometer measurement [nT]
- `itp_mapS`:   scalar map interpolation function (`f(lat,lon)` or `f(lat,lon,alt)`)
- `x_nn`:       `N` x `Nf` data matrix for neural network (`Nf` is number of features)
- `y_nn`:       `y` target matrix for neural network (`[latitude longitude]`)
- `P0`:         (optional) initial covariance matrix
- `Qd`:         (optional) discrete time process/system noise matrix
- `R`:          (optional) measurement (white) noise variance
- `baro_tau`:   (optional) barometer time constant [s]
- `acc_tau`:    (optional) accelerometer time constant [s]
- `gyro_tau`:   (optional) gyroscope time constant [s]
- `fogm_tau`:   (optional) FOGM catch-all time constant [s]
- `η_adam`:     (optional) learning rate for Adam optimizer
- `epoch_adam`: (optional) number of epochs for Adam optimizer
- `hidden`:     (optional) hidden layers & nodes (e.g., `[8,8]` for 2 hidden layers, 8 nodes each)
- `activation`: (optional) activation function
    - `relu`  = rectified linear unit
    - `σ`     = sigmoid (logistic function)
    - `swish` = self-gated
    - `tanh`  = hyperbolic tan
    - run `plot_activation()` for a visual
- `l_window`:   (optional) temporal window length
- `date`:       (optional) measurement date (decimal year) for IGRF [yr]
- `core`:       (optional) if true, include core magnetic field in measurement

**Returns:**
- `m`: neural network model
"""
function nekf_train(ins::INS, meas, itp_mapS, x_nn::Matrix, y_nn::Matrix;
                    P0                   = create_P0(),
                    Qd                   = create_Qd(),
                    R                    = 1.0,
                    baro_tau             = 3600.0,
                    acc_tau              = 3600.0,
                    gyro_tau             = 3600.0,
                    fogm_tau             = 600.0,
                    η_adam               = 0.1,
                    epoch_adam::Int      = 10,
                    hidden::Int          = 1,
                    activation::Function = swish,
                    l_window::Int        = 50,
                    date                 = get_years(2020,185),
                    core::Bool           = false)
    nekf_train(ins.lat,ins.lon,ins.alt,ins.vn,ins.ve,ins.vd,
               ins.fn,ins.fe,ins.fd,ins.Cnb,meas,ins.dt,itp_mapS,x_nn,y_nn;
               P0=P0,Qd=Qd,R=R,
               baro_tau   = baro_tau,
               acc_tau    = acc_tau,
               gyro_tau   = gyro_tau,
               fogm_tau   = fogm_tau,
               η_adam     = η_adam,
               epoch_adam = epoch_adam,
               hidden     = hidden,
               activation = activation,
               l_window   = l_window,
               date       = date,
               core       = core);
end # function nekf_train

"""
    nekf_train(xyz::XYZ, ind, meas, itp_mapS, x::Matrix;
               P0                   = create_P0(),
               Qd                   = create_Qd(),
               R                    = 1.0,
               baro_tau             = 3600.0,
               acc_tau              = 3600.0,
               gyro_tau             = 3600.0,
               fogm_tau             = 600.0,
               η_adam               = 0.1,
               epoch_adam::Int      = 10,
               hidden::Int          = 1,
               activation::Function = swish,
               l_window::Int        = 50,
               date                 = get_years(2020,185),
               core::Bool           = false)

Train a measurement noise covariance-adaptive neural extended Kalman filter
(nEKF) model for airborne magnetic anomaly navigation.

**Arguments:**
- `xyz`:        `XYZ` flight data struct
- `ind`:        selected data indices
- `meas`:       scalar magnetometer measurement [nT]
- `itp_mapS`:   scalar map interpolation function (`f(lat,lon)` or `f(lat,lon,alt)`)
- `x`:          `N` x `Nf` data matrix (`Nf` is number of features)
- `P0`:         (optional) initial covariance matrix
- `Qd`:         (optional) discrete time process/system noise matrix
- `R`:          (optional) measurement (white) noise variance
- `baro_tau`:   (optional) barometer time constant [s]
- `acc_tau`:    (optional) accelerometer time constant [s]
- `gyro_tau`:   (optional) gyroscope time constant [s]
- `fogm_tau`:   (optional) FOGM catch-all time constant [s]
- `η_adam`:     (optional) learning rate for Adam optimizer
- `epoch_adam`: (optional) number of epochs for Adam optimizer
- `hidden`:     (optional) hidden layers & nodes (e.g., `[8,8]` for 2 hidden layers, 8 nodes each)
- `activation`: (optional) activation function
    - `relu`  = rectified linear unit
    - `σ`     = sigmoid (logistic function)
    - `swish` = self-gated
    - `tanh`  = hyperbolic tan
    - run `plot_activation()` for a visual
- `l_window`:   (optional) temporal window length
- `date`:       (optional) measurement date (decimal year) for IGRF [yr]
- `core`:       (optional) if true, include core magnetic field in measurement

**Returns:**
- `m`:          neural network model
- `data_norms`: length-`3` tuple of data normalizations, `(v_scale,x_bias,x_scale)`
"""
function nekf_train(xyz::XYZ, ind, meas, itp_mapS, x::Matrix;
                    P0                   = create_P0(),
                    Qd                   = create_Qd(),
                    R                    = 1.0,
                    baro_tau             = 3600.0,
                    acc_tau              = 3600.0,
                    gyro_tau             = 3600.0,
                    fogm_tau             = 600.0,
                    η_adam               = 0.1,
                    epoch_adam::Int      = 10,
                    hidden::Int          = 1,
                    activation::Function = swish,
                    l_window::Int        = 50,
                    date                 = get_years(2020,185),
                    core::Bool           = false)

    # get traj, ins, & y_nn (position)
    traj = get_traj(xyz,ind)
    ins  = get_ins(xyz,ind;N_zero_ll=1)
    y_nn = [traj.lat traj.lon]

    # normalize x
    (x_bias,x_scale,x_norm) = norm_sets(x;norm_type=:standardize)
    (_,S,V) = svd(cov(x_norm))
    v_scale = V[:,1:1]*inv(Diagonal(sqrt.(S[1:1])))
    x_nn = x_norm * v_scale

    m = nekf_train(ins,meas,itp_mapS,x_nn,y_nn;
                   P0=P0,Qd=Qd,R=R,
                   baro_tau   = baro_tau,
                   acc_tau    = acc_tau,
                   gyro_tau   = gyro_tau,
                   fogm_tau   = fogm_tau,
                   η_adam     = η_adam,
                   epoch_adam = epoch_adam,
                   hidden     = hidden,
                   activation = activation,
                   l_window   = l_window,
                   date       = date,
                   core       = core)

    # pack normalizations
    data_norms = (v_scale,x_bias,x_scale)

    return (m, data_norms)
end # function nekf_train
