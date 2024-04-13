"""
    mpf(lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb, meas, dt, itp_mapS;
        P0         = create_P0(),
        Qd         = create_Qd(),
        R          = 1.0,
        num_part   = 1000,
        thresh     = 0.8,
        baro_tau   = 3600.0,
        acc_tau    = 3600.0,
        gyro_tau   = 3600.0,
        fogm_tau   = 600.0,
        date       = get_years(2020,185),
        core::Bool = false)

Rao-Blackwellized (marginalized) particle filter (MPF) for airborne magnetic
anomaly navigation. This simplified MPF works only with LINEAR dynamics.
This allows the same Kalman filter covariance matrices to be used with
each particle, simplifying the filter and reducing the computational load.
It is especially suited for map-matching navigation in which there is a
highly non-linear, non-Gaussian MEASUREMENT, but NOT non-linear dynamics.
The filter also assumes NON-correlated measurements to speed up computation.

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
- `num_part`: (optional) number of particles
- `thresh`:   (optional) resampling threshold fraction {0:1}
- `baro_tau`: (optional) barometer time constant [s]
- `acc_tau`:  (optional) accelerometer time constant [s]
- `gyro_tau`: (optional) gyroscope time constant [s]
- `fogm_tau`: (optional) FOGM catch-all time constant [s]
- `date`:     (optional) measurement date for IGRF [yr]
- `core`:     (optional) if true, include core magnetic field in measurement

**Returns:**
- `filt_res`: `FILTres` filter results struct
"""
function mpf(lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb, meas, dt, itp_mapS;
             P0         = create_P0(),
             Qd         = create_Qd(),
             R          = 1.0,
             num_part   = 1000,
             thresh     = 0.8,
             baro_tau   = 3600.0,
             acc_tau    = 3600.0,
             gyro_tau   = 3600.0,
             fogm_tau   = 600.0,
             date       = get_years(2020,185),
             core::Bool = false)

    N      = length(lat)  # number of samples (instances)
    np     = num_part     # number of particles
    nx     = size(P0,1)   # total state dimension
    nxn    = 2            # non-linear state dimension (inherent to this model)
    nxl    = nx - nxn     # linear state dimension
    ny     = size(meas,2) # measurement dimension
    T2     = eltype(P0)   # floating precision type

    H      = [zeros(T2,1,nxl-1) 1] # linear measurement Jacobian
    x_out  = zeros(T2,nx,N)        # filtered states, i.e., E[x(t) | y_1,..,y_t]
    Pn_out = zeros(T2,nxn,nxn,N)   # non-linear portion of covariance matrix
    Pl_out = zeros(T2,nxl,nxl,N)   # linear portion of covariance matrix
    resid  = zeros(T2,ny,N)        # measurement residuals

    # initialize non-linear states
    xn = chol(P0[1:nxn,1:nxn])'*randn(T2,nxn,np) # 2 x np

    # initialize conditionally linear Gaussian states
    xl = zeros(T2,nxl,np) # 16 x np

    # initialize linear portion of covariance matrix
    Pl = P0[nxn+1:end,nxn+1:end] # 16 x 16

    # initialize particle weights
    q = ones(T2,np)/np # np

    map_cache = itp_mapS isa Map_Cache ? itp_mapS : nothing

    for t = 1:N
        # custom itp_mapS from map cache, if available
        if map_cache isa Map_Cache
            itp_mapS = get_cached_map(map_cache,lat[t],lon[t],alt[t];silent=true)
        end

        # Pinson matrix exponential (dynamics matrix)
        Phi  = get_Phi(nx,lat[t],vn[t],ve[t],vd[t],fn[t],fe[t],fd[t],Cnb[:,:,t],
                       baro_tau,acc_tau,gyro_tau,fogm_tau,dt) # 18 x 18

        # dynamics matrix sub-matrices
        An_l = Phi[1:nxn,nxn+1:end]     # 2  x 16
        An_n = Phi[1:nxn,1:nxn]         # 2  x 2
        Al_l = Phi[nxn+1:end,nxn+1:end] # 16 x 16
        Al_n = Phi[nxn+1:end,1:nxn]     # 16 x 2

        # expected measurement and residual
        y_hat = get_h(itp_mapS,[xn;xl],lat[t],lon[t],alt[t];date=date,core=core)
        e = repeat(meas[t,:],1,np) - repeat(y_hat',ny,1)
        resid[:,t] = mean(e,dims=2)

        # particle update                                           # eq 25a
        V = H*Pl*H' .+ R
        for i = 1:ny
            q = q.*exp.(-0.5*(e[i,:].*(1/V[i,i]).*e[i,:])) # weight particles
            # weighting only works for NON-correlated measurements
        end
        # check divergence, resample, store non-linear portion
        if sum(q) > eps(T2) # filter has not diverged
            q = q/sum(q) # normalize particle weights

            # store non-linear particle states
            for i = 1:nxn
                x_out[i,t] = sum(q.*xn[i,:])
            end

            # store non-linear portion of covariance matrix
            Pn_out[:,:,t] = part_cov(q,xn,x_out[1:nxn,t])

            N_eff = 1/sum(q.^2) # effective particles
            if N_eff < np*thresh
                ind = sys_resample(q) # resample
                xn  = xn[:,ind] # resampled non-linear particles
                xl  = xl[:,ind] # resampled linear particles
                q   = ones(T2,np)/np
            end
        else # filter has diverged, so exit
            converge = false
            P_out = filter_exit(Pl_out,Pn_out,t,converge)
            return FILTres(x_out, P_out, resid, converge)
        end

        # Kalman filter covariance update
        K  = Pl * H' / V # 16 x ny                                  # eq 22d
        Pl = Pl - K*V*K' # 16 x 16                                  # eq 22b

        # Kalman filter mean update
        y_hat = get_h(itp_mapS,[xn;xl],lat[t],lon[t],alt[t];date=date,core=core)
        e  = repeat(meas[t,:],1,np) - repeat(y_hat',ny,1)
        xl_temp = xl
        xl = xl + K*e                                               # eq 22a

        # store linear particle states
        for i = 1:nxl
            x_out[nxn+i,t] = sum(q.*xl[i,:])
        end

        # store linear portion of covariance matrix
        Pl_out[:,:,t] = part_cov(q,xl,x_out[1+nxn:end,t],Pl)

        # Kalman filter covariance propagation
        M  = An_l*Pl*An_l' + Qd[1:nxn,1:nxn]                        # eq 23c
        L  = Al_l*Pl*An_l' / M                                      # eq 23d
        Pl = Al_l*Pl*Al_l' + Qd[nxn+1:end,nxn+1:end] - L*M*L'       # eq 23b
        Pl = (Pl+Pl')/2 # ensure symmetry

        # particle filter propagation                               # eq 25b
        xn_temp = xn
        xn = An_n*xn_temp + An_l*xl_temp + chol(M)'*randn(T2,nxn,np)

        # Kalman filter mean propagation
        z  = xn - An_n*xn_temp                                      # eq 24a
        xl = Al_n*xn_temp + Al_l*xl + L*(z-An_l*xl)                 # eq 23a

    end

    converge = true
    P_out = filter_exit(Pl_out,Pn_out,N,converge)
    return FILTres(x_out, P_out, resid, converge)
end # function mpf

"""
    mpf(ins::INS, meas, itp_mapS;
        P0         = create_P0(),
        Qd         = create_Qd(),
        R          = 1.0,
        num_part   = 1000,
        thresh     = 0.8,
        baro_tau   = 3600.0,
        acc_tau    = 3600.0,
        gyro_tau   = 3600.0,
        fogm_tau   = 600.0,
        date       = get_years(2020,185),
        core::Bool = false)

Rao-Blackwellized (marginalized) particle filter (MPF) for airborne magnetic
anomaly navigation. This simplified MPF works only with LINEAR dynamics.
This allows the same Kalman filter covariance matrices to be used with
each particle, simplifying the filter and reducing the computational load.
It is especially suited for map-matching navigation in which there is a
highly non-linear, non-Gaussian MEASUREMENT, but NOT non-linear dynamics.
The filter also assumes NON-correlated measurements to speed up computation.

**Arguments:**
- `ins`:      `INS` inertial navigation system struct
- `meas`:     scalar magnetometer measurement [nT]
- `itp_mapS`: scalar map interpolation function (`f(lat,lon)` or `f(lat,lon,alt)`)
- `P0`:       (optional) initial covariance matrix
- `Qd`:       (optional) discrete time process/system noise matrix
- `R`:        (optional) measurement (white) noise variance
- `num_part`: (optional) number of particles
- `thresh`:   (optional) resampling threshold fraction {0:1}
- `baro_tau`: (optional) barometer time constant [s]
- `acc_tau`:  (optional) accelerometer time constant [s]
- `gyro_tau`: (optional) gyroscope time constant [s]
- `fogm_tau`: (optional) FOGM catch-all time constant [s]
- `date`:     (optional) measurement date for IGRF [yr]
- `core`:     (optional) if true, include core magnetic field in measurement

**Returns:**
- `filt_res`: `FILTres` filter results struct
"""
function mpf(ins::INS, meas, itp_mapS;
             P0         = create_P0(),
             Qd         = create_Qd(),
             R          = 1.0,
             num_part   = 1000,
             thresh     = 0.8,
             baro_tau   = 3600.0,
             acc_tau    = 3600.0,
             gyro_tau   = 3600.0,
             fogm_tau   = 600.0,
             date       = get_years(2020,185),
             core::Bool = false)
    mpf(ins.lat,ins.lon,ins.alt,ins.vn,ins.ve,ins.vd,ins.fn,ins.fe,ins.fd,
        ins.Cnb,meas,ins.dt,itp_mapS;
        P0       = P0,
        Qd       = Qd,
        R        = R,
        num_part = num_part,
        thresh   = thresh,
        baro_tau = baro_tau,
        acc_tau  = acc_tau,
        gyro_tau = gyro_tau,
        fogm_tau = fogm_tau,
        date     = date,
        core     = core)
end # function mpf

function sys_resample(q)
    qc = cumsum(q)
    np = length(q)
    u  = ((1:np) .- rand(eltype(q),1)) ./ np
    i  = zeros(Int,np)
    k  = 1

    for j = 1:np
        while (qc[k] < u[j]) & (k < np)
            k += 1
        end
        i[j] = k
    end

    return (i)
end # function sys_resample

function part_cov(q, x, x_mean, P = 0)
    (nx,np) = size(x)
    P_temp  = x - repeat(x_mean,1,np)
    P_out   = repeat(q',nx,1).*P_temp*P_temp' .+ P
    return (P_out)
end # function part_cov

function filter_exit(Pl_out, Pn_out, t::Int, converge::Bool = true)
    nxl   = size(Pl_out,1)
    nxn   = size(Pn_out,1)
    nx    = nxl+nxn
    N     = size(Pl_out,3)
    P_out = zeros(eltype(Pl_out),nx,nx,N) # covariance matrix
    P_out[1:nxn,1:nxn,:]         = Pn_out # non-linear portion
    P_out[nxn+1:end,nxn+1:end,:] = Pl_out # linear portion
    converge || @info("filter diverged, particle weights ~0 at time step $t")
    return (P_out)
end # function filter_exit
