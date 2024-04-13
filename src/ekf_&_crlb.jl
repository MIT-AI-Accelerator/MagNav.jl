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
- `itp_mapS`: scalar map interpolation function (`f(lat,lon)` or `f(lat,lon,alt)`)
- `P0`:       (optional) initial covariance matrix
- `Qd`:       (optional) discrete time process/system noise matrix
- `R`:        (optional) measurement (white) noise variance
- `baro_tau`: (optional) barometer time constant [s]
- `acc_tau`:  (optional) accelerometer time constant [s]
- `gyro_tau`: (optional) gyroscope time constant [s]
- `fogm_tau`: (optional) FOGM catch-all time constant [s]
- `date`:     (optional) measurement date for IGRF [yr]
- `core`:     (optional) if true, include core magnetic field in measurement
- `der_mapS`: (optional) scalar map vertical derivative map interpolation function (`f(lat,lon)` or (`f(lat,lon,alt)`)
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
    x_out = zeros(eltype(P0),nx,N)
    P_out = zeros(eltype(P0),nx,nx,N)
    r_out = zeros(eltype(P0),ny,N)
    x     = zeros(eltype(P0),nx) # state estimate
    P     = P0 # covariance matrix

    if length(R) == 2
        adapt = true
        (R_min,R_max) = R
        R = mean(R)
    else
        adapt = false
    end

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
        if (map_alt > 0) & !(der_mapS isa Nothing)
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
                t % n == 0 && println(t," ",sqrt(R))
            end
        end

        # measurement residual covariance
        S = H*P*H' .+ R         # S_t [ny x ny]

        # Kalman gain
        K = (P*H') / S          # K_t [nx x ny]

        # state & covariance update
        x = x + K*resid         # x_t [nx]
        P = (I - K*H) * P       # P_t [nx x nx]

        # state & covariance store
        x_out[:,t]   = x
        P_out[:,:,t] = P

        # state & covariance propagate (predict)
        x = Phi*x               # x_t|t-1 [nx]
        P = Phi*P*Phi' + Qd     # P_t|t-1 [nx x nx]
    end

    return FILTres(x_out, P_out, r_out, true)
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
- `itp_mapS`: scalar map interpolation function (`f(lat,lon)` or `f(lat,lon,alt)`)
- `P0`:       (optional) initial covariance matrix
- `Qd`:       (optional) discrete time process/system noise matrix
- `R`:        (optional) measurement (white) noise variance
- `baro_tau`: (optional) barometer time constant [s]
- `acc_tau`:  (optional) accelerometer time constant [s]
- `gyro_tau`: (optional) gyroscope time constant [s]
- `fogm_tau`: (optional) FOGM catch-all time constant [s]
- `date`:     (optional) measurement date for IGRF [yr]
- `core`:     (optional) if true, include core magnetic field in measurement
- `der_mapS`: (optional) scalar map vertical derivative map interpolation function (`f(lat,lon)` or (`f(lat,lon,alt)`)
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
    (ekf_rt::EKF_RT)(lat, lon, alt, vn, ve, vd, fn, fe, fd,
                     Cnb, meas, t, itp_mapS;
                     der_mapS = nothing,
                     map_alt  = -1,
                     dt       = 0.1)

Real-time (RT) extended Kalman filter (EKF) for airborne magnetic anomaly
navigation. Receives individual samples and updates time, covariance matrix,
state vector, and measurement residual within `ekf_rt` struct.

**Arguments:**
- `ekf_rt`:   `EKF_RT` real-time extended Kalman filter struct
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
- `t`:        time [s]
- `itp_mapS`: scalar map interpolation function (`f(lat,lon)` or `f(lat,lon,alt)`)
- `der_mapS`: (optional) scalar map vertical derivative map interpolation function (`f(lat,lon)` or (`f(lat,lon,alt)`)
- `map_alt`:  (optional) map altitude [m]
- `dt`:       (optional) measurement time step [s], only used if `ekf_rt.t < 0`

**Returns:**
- `filt_res`: `FILTres` filter results struct,
- `ekf_rt`:   `t`, `P`, `x`, & `r` fields are mutated
"""
function (ekf_rt::EKF_RT)(lat, lon, alt, vn, ve, vd, fn, fe, fd,
                          Cnb, meas, t, itp_mapS;
                          der_mapS = nothing,
                          map_alt  = -1,
                          dt       = 0.1)

    o = ekf_rt

    @assert o.ny == size(meas,2) "measurement dimension is inconsistent"

    dt = o.t < 0 ? dt : (t - o.t)

    Phi = get_Phi(o.nx,lat,vn,ve,vd,fn,fe,fd,Cnb,
                  o.baro_tau,o.acc_tau,o.gyro_tau,o.fogm_tau,dt)

    # get map interpolation function from map cache (based on location)
    if itp_mapS isa Map_Cache
        itp_mapS = get_cached_map(itp_mapS,lat,lon,alt)
        der_mapS = nothing
    end

    if (map_alt > 0) & !(der_mapS isa Nothing)
        resid = meas .- get_h(itp_mapS,der_mapS,o.x,lat,lon,alt,map_alt;
                              date=o.date,core=o.core)
    else
        resid = meas .- get_h(itp_mapS,o.x,lat,lon,alt;
                              date=o.date,core=o.core)
    end

    o.t = t
    o.r = resid

    H = repeat(get_H(itp_mapS,o.x,lat,lon,alt,date=o.date,core=o.core)',o.ny,1)

    S = H*o.P*H' .+ o.R

    K = (o.P*H') / S

    o.x = o.x + K*resid
    o.P = (I - K*H) * o.P

    # state, covariance, & residual store, matching ekf()
    x_out = reshape(o.x,(size(o.x)...,1))
    P_out = reshape(o.P,(size(o.P)...,1))
    r_out = reshape(o.r,(size(o.r)...,1))

    o.x = Phi*o.x
    o.P = Phi*o.P*Phi' + o.Qd

    return FILTres(x_out, P_out, r_out, true)
end # function EKF_RT

"""
    (ekf_rt::EKF_RT)(ins::INS, meas, itp_mapS;
                     der_mapS = nothing,
                     map_alt  = -1)

Real-time (RT) extended Kalman filter (EKF) for airborne magnetic anomaly
navigation. Receives individual samples and updates time, covariance matrix,
state vector, and measurement residual within `ekf_rt` struct.

**Arguments:**
- `ekf_rt`:   `EKF_RT` real-time extended Kalman filter struct
- `ins`:      `INS` inertial navigation system struct with individual sample (`ins.N=1`)
- `meas`:     scalar magnetometer measurement [nT]
- `itp_mapS`: scalar map interpolation function (`f(lat,lon)` or `f(lat,lon,alt)`)
- `der_mapS`: (optional) scalar map vertical derivative map interpolation function (`f(lat,lon)` or (`f(lat,lon,alt)`)
- `map_alt`:  (optional) map altitude [m]

**Returns:**
- `filt_res`: `FILTres` filter results struct,
- `ekf_rt`:   `t`, `P`, `x`, & `r` fields are mutated
"""
function (ekf_rt::EKF_RT)(ins::INS, meas, itp_mapS;
                          der_mapS = nothing,
                          map_alt  = -1)
    @assert ins.N == 1 "only individual samples are supported for EKF_RT"
    ekf_rt(ins.lat[1],ins.lon[1],ins.alt[1],ins.vn[1],ins.ve[1],ins.vd[1],
           ins.fn[1],ins.fe[1],ins.fd[1],ins.Cnb[:,:,1],meas,ins.tt[1],itp_mapS;
           der_mapS = der_mapS,
           map_alt  = map_alt,
           dt       = ins.dt)
end # function EKF_RT

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
- `itp_mapS`: scalar map interpolation function (`f(lat,lon)` or `f(lat,lon,alt)`)
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
- `P`: non-linear covariance matrix
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
    P_out = zeros(eltype(P0),nx,nx,N)
    x     = zeros(eltype(P0),nx) # state estimate
    P     = P0 # covariance matrix

    length(R) == 2 && (R = mean(R))
    map_cache = itp_mapS isa Map_Cache ? itp_mapS : nothing

    for t = 1:N
        # custom itp_mapS from map cache, if available
        if map_cache isa Map_Cache
            itp_mapS = get_cached_map(map_cache,lat[t],lon[t],alt[t];silent=true)
        end
        # Pinson matrix exponential
        Phi = get_Phi(nx,lat[t],vn[t],ve[t],vd[t],fn[t],fe[t],fd[t],Cnb[:,:,t],
                      baro_tau,acc_tau,gyro_tau,fogm_tau,dt)
        H = get_H(itp_mapS,x,lat[t],lon[t],alt[t];date=date,core=core)'
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
- `itp_mapS`: scalar map interpolation function (`f(lat,lon)` or `f(lat,lon,alt)`)
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
- `P`: non-linear covariance matrix
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
