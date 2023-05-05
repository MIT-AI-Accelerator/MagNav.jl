"""
    ekf(lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb, meas, dt, itp_mapS;
        P0         = create_P0(),
        Qd         = create_Qd(),
        R          = 1.0,
        baro_tau   = 3600.0,
        acc_tau    = 3600.0,
        gyro_tau   = 3600.0,
        fogm_tau   = 600.0,
        date       = get_years(2020,185),
        core::Bool = false,
        der_mapS   = nothing,
        map_alt    = 0)

Extended Kalman filter (EKF) for airborne magnetic anomaly navigation.

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
- `itp_mapS`: scalar map grid interpolation
- `P0`:       (optional) initial covariance matrix
- `Qd`:       (optional) discrete time process/system noise matrix
- `R`:        (optional) measurement (white) noise variance
- `baro_tau`: (optional) barometer time constant [s]
- `acc_tau`:  (optional) accelerometer time constant [s]
- `gyro_tau`: (optional) gyroscope time constant [s]
- `fogm_tau`: (optional) FOGM catch-all time constant [s]
- `date`:     (optional) measurement date for IGRF [yr]
- `core`:     (optional) if true, include core magnetic field in measurement
- `der_mapS`: (optional) scalar map vertical derivative grid interpolation
- `map_alt`:  (optional) map altitude [m]

**Returns:**
- `filt_res`: `FILTres` filter results struct
"""
function ekf(lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb, meas, dt, itp_mapS;
             P0         = create_P0(),
             Qd         = create_Qd(),
             R          = 1.0,
             baro_tau   = 3600.0,
             acc_tau    = 3600.0,
             gyro_tau   = 3600.0,
             fogm_tau   = 600.0,
             date       = get_years(2020,185),
             core::Bool = false,
             der_mapS   = nothing,
             map_alt    = 0)

    N     = length(lat)
    nx    = size(P0,1)
    ny    = size(meas,2)
    x_out = zeros(nx,N)
    P_out = zeros(nx,nx,N)
    r_out = zeros(ny,N)

    if length(R) == 2
        adapt = true
        (R_min,R_max) = R
        R = mean(R)
    else
        adapt = false
    end

    x = zeros(nx) # state estimate
    P = P0        # covariance matrix

    for t = 1:N

        # Pinson matrix exponential
        Phi = get_Phi(nx,lat[t],vn[t],ve[t],vd[t],fn[t],fe[t],fd[t],Cnb[:,:,t],
                      baro_tau,acc_tau,gyro_tau,fogm_tau,dt)

        # measurement residual [ny]
        if (map_alt > 0) & (der_mapS !== nothing)
            resid = meas[t,:] .- get_h(itp_mapS,der_mapS,x,lat[t],lon[t],
                                       alt[t],map_alt;date=date,core=core)
        else
            resid = meas[t,:] .- get_h(itp_mapS,x,lat[t],lon[t],alt[t];
                                       date=date,core=core)
        end

        # residual store
        r_out[:,t] = resid

        # measurement Jacobian (repeated gradient here) [ny x nx]
        H = repeat(get_H(itp_mapS,x,lat[t],lon[t],alt[t];
                         date=date,core=core)',ny,1)

        if adapt
            n = 10
            if t >= n+1
                r = r_out[:,(t-n):(t-1)] # ny x n
                # println(((r*r')/n - H*P*H')[1][1])
                R = clamp(((r*r')/n - H*P*H')[1][1],R_min,R_max) # ny x ny
                mod(t,n) == 0 && println(t," ",sqrt(R))
            end
        end

        # measurement residual covariance
        S = H*P*H' .+ R         # S_t [ny x ny]

        # Kalman gain
        K = (P*H') / S          # K_t [nx x ny]

        # state and covariance update
        x = x + K*resid         # x_t [nx]
        P = (I - K*H) * P       # P_t [nx x nx]

        # state and covariance store
        x_out[:,t]   = x
        P_out[:,:,t] = P

        # state and covariance propagate (predict)
        x = Phi*x               # x_t|t-1 [nx]
        P = Phi*P*Phi' + Qd     # P_t|t-1 [nx x nx]
    end

    return FILTres(x_out, P_out, r_out, false)
end # function ekf

"""
    ekf(ins::INS, meas, itp_mapS;
        P0         = create_P0(),
        Qd         = create_Qd(),
        R          = 1.0,
        baro_tau   = 3600.0,
        acc_tau    = 3600.0,
        gyro_tau   = 3600.0,
        fogm_tau   = 600.0,
        date       = get_years(2020,185),
        core::Bool = false,
        der_mapS   = map_itp(zeros(2,2),[-pi,pi],[-pi/2,pi/2]),
        map_alt    = 0)

Extended Kalman filter (EKF) for airborne magnetic anomaly navigation.

**Arguments:**
- `ins`:      `INS` inertial navigation system struct
- `meas`:     scalar magnetometer measurement [nT]
- `itp_mapS`: scalar map grid interpolation
- `P0`:       (optional) initial covariance matrix
- `Qd`:       (optional) discrete time process/system noise matrix
- `R`:        (optional) measurement (white) noise variance
- `baro_tau`: (optional) barometer time constant [s]
- `acc_tau`:  (optional) accelerometer time constant [s]
- `gyro_tau`: (optional) gyroscope time constant [s]
- `fogm_tau`: (optional) FOGM catch-all time constant [s]
- `date`:     (optional) measurement date for IGRF [yr]
- `core`:     (optional) if true, include core magnetic field in measurement
- `der_mapS`: (optional) scalar map vertical derivative grid interpolation
- `map_alt`:  (optional) map altitude [m]

**Returns:**
- `filt_res`: `FILTres` filter results struct
"""
function ekf(ins::INS, meas, itp_mapS;
             P0         = create_P0(),
             Qd         = create_Qd(),
             R          = 1.0,
             baro_tau   = 3600.0,
             acc_tau    = 3600.0,
             gyro_tau   = 3600.0,
             fogm_tau   = 600.0,
             date       = get_years(2020,185),
             core::Bool = false,
             der_mapS   = map_itp(zeros(2,2),[-pi,pi],[-pi/2,pi/2]),
             map_alt    = 0)

    ekf(ins.lat,ins.lon,ins.alt,ins.vn,ins.ve,ins.vd,ins.fn,ins.fe,ins.fd,
        ins.Cnb,meas,ins.dt,itp_mapS;
        P0       = P0,
        Qd       = Qd,
        R        = R,
        baro_tau = baro_tau,
        acc_tau  = acc_tau,
        gyro_tau = gyro_tau,
        fogm_tau = fogm_tau,
        date     = date,
        core     = core,
        der_mapS = der_mapS,
        map_alt  = map_alt)

end # function ekf

"""
    crlb(lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb, dt, itp_mapS;
         P0         = create_P0(),
         Qd         = create_Qd(),
         R          = 1.0,
         baro_tau   = 3600.0,
         acc_tau    = 3600.0,
         gyro_tau   = 3600.0,
         fogm_tau   = 600.0,
         date       = get_years(2020,185),
         core::Bool = false)

Cramér–Rao lower bound (CRLB) computed with classic Kalman Filter. 
Equations evaluated about true trajectory.

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
- `dt`:       measurement time step [s]
- `itp_mapS`: scalar map grid interpolation
- `P0`:       (optional) initial covariance matrix
- `Qd`:       (optional) discrete time process/system noise matrix
- `R`:        (optional) measurement (white) noise variance
- `baro_tau`: (optional) barometer time constant [s]
- `acc_tau`:  (optional) accelerometer time constant [s]
- `gyro_tau`: (optional) gyroscope time constant [s]
- `fogm_tau`: (optional) FOGM catch-all time constant [s]
- `date`:     (optional) measurement date for IGRF [yr]
- `core`:     (optional) if true, include core magnetic field in measurement

**Returns:**
- `P_out`: non-linear covariance matrix
"""
function crlb(lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb, dt, itp_mapS;
              P0         = create_P0(),
              Qd         = create_Qd(),
              R          = 1.0,
              baro_tau   = 3600.0,
              acc_tau    = 3600.0,
              gyro_tau   = 3600.0,
              fogm_tau   = 600.0,
              date       = get_years(2020,185),
              core::Bool = false)

    N     = length(lat)
    nx    = size(P0,1)
    P_out = zeros(nx,nx,N)
    P     = P0 # covariance matrix

    length(R) == 2 && (R = mean(R))

    for t = 1:N
        # Pinson matrix exponential
        Phi = get_Phi(nx,lat[t],vn[t],ve[t],vd[t],fn[t],fe[t],fd[t],Cnb[:,:,t],
                      baro_tau,acc_tau,gyro_tau,fogm_tau,dt)
        H = get_H(itp_mapS,zeros(nx),lat[t],lon[t],alt[t];date=date,core=core)'
        S = H*P*H' .+ R
        K = (P*H') / S
        P = (I - K*H) * P
        P_out[:,:,t] = P

        # propagate
        P = Phi*P*Phi' + Qd
    end

    return (P_out)
end # function crlb

"""
    crlb(traj::Traj, itp_mapS;
         P0         = create_P0(),
         Qd         = create_Qd(),
         R          = 1.0,
         baro_tau   = 3600.0,
         acc_tau    = 3600.0,
         gyro_tau   = 3600.0,
         fogm_tau   = 600.0,
         date       = get_years(2020,185),
         core::Bool = false)

Cramér–Rao lower bound (CRLB) computed with classic Kalman Filter. 
Equations evaluated about true trajectory.

**Arguments:**
- `traj`:     `Traj` trajectory struct
- `itp_mapS`: scalar map grid interpolation
- `P0`:       (optional) initial covariance matrix
- `Qd`:       (optional) discrete time process/system noise matrix
- `R`:        (optional) measurement (white) noise variance
- `baro_tau`: (optional) barometer time constant [s]
- `acc_tau`:  (optional) accelerometer time constant [s]
- `gyro_tau`: (optional) gyroscope time constant [s]
- `fogm_tau`: (optional) FOGM catch-all time constant [s]
- `date`:     (optional) measurement date for IGRF [yr]
- `core`:     (optional) if true, include core magnetic field in measurement

**Returns:**
- `P_out`: non-linear covariance matrix
"""
function crlb(traj::Traj, itp_mapS;
              P0         = create_P0(),
              Qd         = create_Qd(),
              R          = 1.0,
              baro_tau   = 3600.0,
              acc_tau    = 3600.0,
              gyro_tau   = 3600.0,
              fogm_tau   = 600.0,
              date       = get_years(2020,185),
              core::Bool = false)

    crlb(traj.lat,traj.lon,traj.alt,traj.vn,traj.ve,traj.vd,
         traj.fn,traj.fe,traj.fd,traj.Cnb,traj.dt,itp_mapS;
         P0       = P0,
         Qd       = Qd,
         R        = R,
         baro_tau = baro_tau,
         acc_tau  = acc_tau,
         gyro_tau = gyro_tau,
         fogm_tau = fogm_tau,
         date     = date,
         core     = core)

end # function crlb
