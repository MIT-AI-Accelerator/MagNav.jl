"""
    ekf_online(lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb, meas,
               Bx, By, Bz, dt, itp_mapS, x0_TL, P0, Qd, R;
               baro_tau   = 3600.0,
               acc_tau    = 3600.0,
               gyro_tau   = 3600.0,
               fogm_tau   = 600.0,
               date       = get_years(2020,185),
               core::Bool = false,
               terms      = [:permanent,:induced,:eddy,:bias],
               Bt_scale   = 50000)

Extended Kalman filter (EKF) with online learning of Tolles-Lawson coefficients.

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
- `Bx,By,Bz`: vector magnetometer measurements [nT]
- `dt`:       measurement time step [s]
- `itp_mapS`: scalar map grid interpolation
- `x0_TL`:    initial Tolles-Lawson coefficient states
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
- `Bt_scale`: (optional) scaling factor for induced and eddy current terms [nT]

**Returns:**
- `filt_res`: `FILTres` filter results struct
"""
function ekf_online(lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb, meas,
                    Bx, By, Bz, dt, itp_mapS, x0_TL, P0, Qd, R;
                    baro_tau   = 3600.0,
                    acc_tau    = 3600.0,
                    gyro_tau   = 3600.0,
                    fogm_tau   = 600.0,
                    date       = get_years(2020,185),
                    core::Bool = false,
                    terms      = [:permanent,:induced,:eddy,:bias],
                    Bt_scale   = 50000)

    N      = length(lat)
    ny     = size(meas,2)
    nx     = size(P0,1)
    nx_TL  = length(x0_TL)
    nx_vec = nx - 18 - nx_TL
    x_out  = zeros(nx,N)
    P_out  = zeros(nx,nx,N)
    r_out  = zeros(ny,N)

    x = zeros(nx) # state estimate
    P = P0        # covariance matrix
    A = create_TL_A(Bx,By,Bz;
                    # Bt       = meas[:,1],
                    terms    = terms,
                    Bt_scale = Bt_scale)

    # function f(Bx,By,Bz,meas,terms,Bt_scale,x_TL)
    #     create_TL_A(Bx,By,Bz;
    #                 Bt       = meas,
    #                 terms    = terms,
    #                 Bt_scale = Bt_scale)[2,:]'*x_TL
    # end # function f

    x[end-nx_vec-nx_TL:end-nx_vec-1] = x0_TL

    vec_states = nx_vec > 0 ? true : false

    for t = 1:N

        # overwrite vector magnetometer measurements
        nx_vec == 3 && (x[end-3:end-1] = [Bx[t],By[t],Bz[t]])

        # Pinson matrix exponential
        Phi = get_Phi(nx,lat[t],vn[t],ve[t],vd[t],fn[t],fe[t],fd[t],Cnb[:,:,t],
                      baro_tau,acc_tau,gyro_tau,fogm_tau,dt;vec_states=vec_states)

        # measurement residual [ny]
        x_TL = x[end-nx_vec-nx_TL:end-nx_vec-1]
        resid = meas[t,:] .- A[t,:]'*x_TL .-
                get_h(itp_mapS,x,lat[t],lon[t],alt[t];date=date,core=core)

        # measurement Jacobian (repeated gradient here) [ny x nx]
        Hll = get_H(itp_mapS,x,lat[t],lon[t],alt[t];date=date,core=core)'
        if nx_vec == 3
            # 1st 2 terms ~1000x greater than last (eddy current) term
            # ind = max(1,t-1):min(t+1,N)
            # HBx = ForwardDiff.gradient(Bx -> f(Bx,By[ind],Bz[ind],meas[ind],
            #                                    terms,Bt_scale,x_TL),Bx[ind])[2]
            # HBy = ForwardDiff.gradient(By -> f(Bx[ind],By,Bz[ind],meas[ind],
            #                                    terms,Bt_scale,x_TL),By[ind])[2]
            # HBz = ForwardDiff.gradient(Bz -> f(Bx[ind],By[ind],Bz,meas[ind],
            #                                    terms,Bt_scale,x_TL),Bz[ind])[2]
            HBx = x_TL[1] / meas[t,1] + 
                  (2*A[t,1]*x_TL[4] + A[t,2]*x_TL[5] + A[t,3]*x_TL[6]) / Bt_scale + 
                  (A[t,10:12]'*x_TL[10:12]) / A[t,4] * A[t,1] / Bt_scale
            HBy = x_TL[2] / meas[t,1] + 
                  (A[t,1]*x_TL[5] + 2*A[t,2]*x_TL[7] + A[t,3]*x_TL[8]) / Bt_scale + 
                  (A[t,10:12]'*x_TL[13:15]) / A[t,4] * A[t,1] / Bt_scale
            HBz = x_TL[3] / meas[t,1] + 
                  (A[t,1]*x_TL[6] + A[t,2]*x_TL[8] + 2*A[t,3]*x_TL[9]) / Bt_scale + 
                  (A[t,10:12]'*x_TL[16:18]) / A[t,4] * A[t,1] / Bt_scale
            H1  = [Hll[1:2]; zeros(nx-3-nx_vec-nx_TL); A[t,:]; HBx; HBy; HBz; 1]
        else
            H1  = [Hll[1:2]; zeros(nx-3-nx_TL); A[t,:]; 1]
        end

        H = repeat(H1',ny,1)

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

    return FILTres(x_out, P_out, r_out, false)
end # function ekf_online

"""
    ekf_online(ins::INS, meas, flux::MagV, itp_mapS, x0_TL, P0, Qd, R;
               baro_tau   = 3600.0,
               acc_tau    = 3600.0,
               gyro_tau   = 3600.0,
               fogm_tau   = 600.0,
               date       = get_years(2020,185),
               core::Bool = false,
               terms      = [:permanent,:induced,:eddy,:bias],
               Bt_scale   = 50000)

Extended Kalman filter (EKF) with online learning of Tolles-Lawson coefficients.

**Arguments:**
- `ins`:      `INS` inertial navigation system struct
- `meas`:     scalar magnetometer measurement [nT]
- `flux`:     `MagV` vector magnetometer measurement struct
- `itp_mapS`: scalar map grid interpolation
- `x0_TL`:    initial Tolles-Lawson coefficient states
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
- `Bt_scale`: (optional) scaling factor for induced and eddy current terms [nT]

**Returns:**
- `filt_res`: `FILTres` filter results struct
"""
function ekf_online(ins::INS, meas, flux::MagV, itp_mapS, x0_TL, P0, Qd, R;
                    baro_tau   = 3600.0,
                    acc_tau    = 3600.0,
                    gyro_tau   = 3600.0,
                    fogm_tau   = 600.0,
                    date       = get_years(2020,185),
                    core::Bool = false,
                    terms      = [:permanent,:induced,:eddy,:bias],
                    Bt_scale   = 50000)
    ekf_online(ins.lat,ins.lon,ins.alt,ins.vn,ins.ve,ins.vd,ins.fn,ins.fe,ins.fd,
               ins.Cnb,meas,flux.x,flux.y,flux.z,ins.dt,itp_mapS,x0_TL,P0,Qd,R;
               baro_tau = baro_tau,
               acc_tau  = acc_tau,
               gyro_tau = gyro_tau,
               fogm_tau = fogm_tau,
               date     = date,
               core     = core,
               terms    = terms,
               Bt_scale = Bt_scale)
end # function ekf_online

"""
    ekf_online_setup(flux::MagV, meas, ind=trues(length(meas));
                     Bt           = sqrt.(flux.x.^2+flux.y.^2+flux.z.^2)[ind],
                     λ            = 0.025,
                     terms        = [:permanent,:induced,:eddy,:bias],
                     pass1        = 0.1,
                     pass2        = 0.9,
                     fs           = 10.0,
                     pole::Int    = 4,
                     trim::Int    = 20,
                     N_sigma::Int = 100,
                     Bt_scale     = 50000)

Setup for extended Kalman filter (EKF) with online learning of Tolles-Lawson coefficients.

**Arguments:**
- `flux`:     `MagV` vector magnetometer measurement struct
- `meas`:     scalar magnetometer measurement [nT]
- `ind`:      selected data indices
- `Bt`:       (optional) magnitude of vector magnetometer measurements or scalar magnetometer measurements for modified Tolles-Lawson [nT]
- `λ`:        (optional) ridge parameter
- `terms`:    (optional) Tolles-Lawson terms to use {`:permanent`,`:induced`,`:eddy`,`:bias`}
- `pass1`:    (optional) first passband frequency [Hz]
- `pass2`:    (optional) second passband frequency [Hz]
- `fs`:       (optional) sampling frequency [Hz]
- `pole`:     (optional) number of poles for Butterworth filter
- `trim`:     (optional) number of elements to trim after filtering
- `N_sigma`:  (optional) number of Tolles-Lawson coefficient sets to use to create `TL_sigma`
- `Bt_scale`: (optional) scaling factor for induced and eddy current terms [nT]

**Returns:**
- `x0_TL`:    initial Tolles-Lawson coefficient states
- `P0_TL`:    initial Tolles-Lawson covariance matrix
- `TL_sigma`: Tolles-Lawson coefficients estimate std dev
"""
function ekf_online_setup(flux::MagV, meas, ind=trues(length(meas));
                          Bt           = sqrt.(flux.x.^2+flux.y.^2+flux.z.^2),
                          λ            = 0.025,
                          terms        = [:permanent,:induced,:eddy,:bias],
                          pass1        = 0.1,
                          pass2        = 0.9,
                          fs           = 10.0,
                          pole::Int    = 4,
                          trim::Int    = 20,
                          N_sigma::Int = 100,
                          Bt_scale     = 50000)

    (x0_TL,y_var) = create_TL_coef(flux,meas,ind;Bt=Bt,λ=λ,terms=terms,
                                   pass1=pass1,pass2=pass2,fs=fs,pole=pole,
                                   trim=trim,Bt_scale=Bt_scale,return_var=true)

    A     = create_TL_A(flux,ind;Bt=Bt,terms=terms,Bt_scale=Bt_scale)
    P0_TL = inv(A'*A)*y_var
    N_ind = length(meas[ind])
    N     = min(N_ind - max(2*trim,50), N_sigma) # avoid bpf issues
    inds  = sortperm(ind.==1,rev=true)[1:N_ind]

    N_min = 10
    @assert N >= N_min "increase N_sigma to $N_min or use more data"

    coef_set = zeros(size(A,2),N)
    for i = 1:N
        coef_set[:,i] = create_TL_coef(flux,meas,inds[i:end+i-N];
                                       Bt=Bt,λ=λ,terms=terms,pass1=pass1,
                                       pass2=pass2,fs=fs,pole=pole,trim=trim,
                                       Bt_scale=Bt_scale,return_var=false)
    end

    TL_sigma = vec(std(coef_set,dims=2))
    # TL_sigma = vec(minimum(abs.(coef_set[:,2:end]-coef_set[:,1:end-1]),dims=2))
    # TL_sigma = vec(abs.(median(coef_set[:,2:end]-coef_set[:,1:end-1],dims=2)))

    # P0_TL = rls(bpf_data(A)[1+2000:end-2000+1,:],bpf_data(meas)[1+2000:end-2000+1],x0_TL)

    return (x0_TL, P0_TL, TL_sigma)
end # function ekf_online_setup

# function rls(x, y, TL_coef::Vector, n::Int=100)
#     TL_coef = deepcopy(TL_coef) # don't modify original TL_coef
#     P = I(length(TL_coef)) # initialize covariance matrix
#     for i = 1:n
#         K = P*TL_coef/(1+TL_coef'*P*TL_coef)
#         P       -= P*TL_coef*TL_coef'*P/(1+TL_coef'*P*TL_coef)
#         TL_coef += K.*(y[i].-x[i,:].*TL_coef)
#     end
#     return (P)
# end # function rls
