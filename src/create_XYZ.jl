"""
    create_XYZ0(mapS::Union{MapS,MapSd,MapS3D} = get_map(namad);
                alt            = 1000,
                dt             = 0.1,
                t              = 300,
                v              = 68,
                ll1            = (),
                ll2            = (),
                N_waves        = 1,
                flight         = 1,
                line           = 1,
                attempts::Int  = 10,
                mapV::MapV     = get_map(emm720),
                cor_sigma      = 1.0,
                cor_tau        = 600.0,
                cor_var        = 1.0^2,
                cor_drift      = 0.001,
                cor_perm_mag   = 5.0,
                cor_ind_mag    = 5.0,
                cor_eddy_mag   = 0.5,
                init_pos_sigma = 3.0,
                init_alt_sigma = 0.001,
                init_vel_sigma = 0.01,
                init_att_sigma = deg2rad(0.01),
                VRW_sigma      = 0.000238,
                ARW_sigma      = 0.000000581,
                baro_sigma     = 1.0,
                ha_sigma       = 0.001,
                a_hat_sigma    = 0.01,
                acc_sigma      = 0.000245,
                gyro_sigma     = 0.00000000727,
                fogm_sigma     = 1.0,
                baro_tau       = 3600.0,
                acc_tau        = 3600.0,
                gyro_tau       = 3600.0,
                fogm_tau       = 600.0,
                save_h5::Bool  = false,
                xyz_h5::String = "xyz_data.h5",
                silent::Bool   = false)

Create basic flight data. Assumes constant altitude (2D flight).
May create a trajectory that passes over map areas that are missing map data.
No required arguments, though many are available to create custom data.

**Trajectory Arguments:**
- `mapS`:     (optional) `MapS`, `MapSd`, or `MapS3D` scalar magnetic anomaly map struct
- `alt`:      (optional) altitude  [m]
- `dt`:       (optional) measurement time step [s]
- `t`:        (optional) total flight time, ignored if `ll2` is set [s]
- `v`:        (optional) approximate aircraft velocity [m/s]
- `ll1`:      (optional) inital (lat,lon) point [deg]
- `ll2`:      (optional) final  (lat,lon) point [deg]
- `N_waves`:  (optional) number of sine waves along path
- `mapV`:     (optional) `MapV` vector magnetic anomaly map struct
- `flight`:   (optional) flight number
- `line`:     (optional) line number, i.e., segment within `flight`
- `attempts`: (optional) maximum attempts at creating flight path on `mapS`
- `save_h5`:  (optional) if true, save `xyz` to `xyz_h5`
- `xyz_h5`:   (optional) path/name of flight data HDF5 file to save (`.h5` extension optional)

**Compensated Measurement Corruption Arguments:**
- `cor_var`:        (optional) corruption measurement (white) noise variance [nT^2]
- `fogm_sigma`:     (optional) FOGM catch-all bias [nT]
- `fogm_tau`:       (optional) FOGM catch-all time constant [s]

**Uncompensated Measurement Corruption Arguments:**
- `cor_sigma`:      (optional) corruption FOGM catch-all bias [nT]
- `cor_tau`:        (optional) corruption FOGM catch-all time constant [s]
- `cor_var`:        (optional) corruption measurement (white) noise variance [nT^2]
- `cor_drift`:      (optional) corruption measurement linear drift [nT/s]
- `cor_perm_mag`:   (optional) corruption permanent field TL coef std dev
- `cor_ind_mag`:    (optional) corruption induced field TL coef std dev
- `cor_eddy_mag`:   (optional) corruption eddy current TL coef std dev

**INS Arguments:**
- `init_pos_sigma`: (optional) initial position uncertainty [m]
- `init_alt_sigma`: (optional) initial altitude uncertainty [m]
- `init_vel_sigma`: (optional) initial velocity uncertainty [m/s]
- `init_att_sigma`: (optional) initial attitude uncertainty [rad]
- `VRW_sigma`:      (optional) velocity random walk [m/s^2 /sqrt(Hz)]
- `ARW_sigma`:      (optional) angular random walk [rad/s /sqrt(Hz)]
- `baro_sigma`:     (optional) barometer bias [m]
- `ha_sigma`:       (optional) barometer aiding altitude bias [m]
- `a_hat_sigma`:    (optional) barometer aiding vertical accel bias [m/s^2]
- `acc_sigma`:      (optional) accelerometer bias [m/s^2]
- `gyro_sigma`:     (optional) gyroscope bias [rad/s]
- `baro_tau`:       (optional) barometer time constant [s]
- `acc_tau`:        (optional) accelerometer time constant [s]
- `gyro_tau`:       (optional) gyroscope time constant [s]

**General Arguments:**
- `silent`: (optional) if true, no print outs

**Returns:**
- `xyz`: `XYZ0` flight data struct
"""
function create_XYZ0(mapS::Union{MapS,MapSd,MapS3D} = get_map(namad);
                     alt            = 1000,
                     dt             = 0.1,
                     t              = 300,
                     v              = 68,
                     ll1            = (), # (0.54, -1.44)
                     ll2            = (), # (0.55, -1.45)
                     N_waves        = 1,
                     flight         = 1,
                     line           = 1,
                     attempts::Int  = 10,
                     mapV::MapV     = get_map(emm720),
                     cor_sigma      = 1.0,
                     cor_tau        = 600.0,
                     cor_var        = 1.0^2,
                     cor_drift      = 0.001,
                     cor_perm_mag   = 5.0,
                     cor_ind_mag    = 5.0,
                     cor_eddy_mag   = 0.5,
                     init_pos_sigma = 3.0,
                     init_alt_sigma = 0.001,
                     init_vel_sigma = 0.01,
                     init_att_sigma = deg2rad(0.01),
                     VRW_sigma      = 0.000238,
                     ARW_sigma      = 0.000000581,
                     baro_sigma     = 1.0,
                     ha_sigma       = 0.001,
                     a_hat_sigma    = 0.01,
                     acc_sigma      = 0.000245,
                     gyro_sigma     = 0.00000000727,
                     fogm_sigma     = 1.0,
                     baro_tau       = 3600.0,
                     acc_tau        = 3600.0,
                     gyro_tau       = 3600.0,
                     fogm_tau       = 600.0,
                     save_h5::Bool  = false,
                     xyz_h5::String = "xyz_data.h5",
                     silent::Bool   = false)

    xyz_h5 = add_extension(xyz_h5,".h5")

    # create trajectory
    traj = create_traj(mapS;
                       alt      = alt,
                       dt       = dt,
                       t        = t,
                       v        = v,
                       ll1      = ll1,
                       ll2      = ll2,
                       N_waves  = N_waves,
                       attempts = attempts,
                       save_h5  = save_h5,
                       traj_h5  = xyz_h5)

    # create INS
    ins = create_ins(traj;
                     init_pos_sigma = init_pos_sigma,
                     init_alt_sigma = init_alt_sigma,
                     init_vel_sigma = init_vel_sigma,
                     init_att_sigma = init_att_sigma,
                     VRW_sigma      = VRW_sigma,
                     ARW_sigma      = ARW_sigma,
                     baro_sigma     = baro_sigma,
                     ha_sigma       = ha_sigma,
                     a_hat_sigma    = a_hat_sigma,
                     acc_sigma      = acc_sigma,
                     gyro_sigma     = gyro_sigma,
                     baro_tau       = baro_tau,
                     acc_tau        = acc_tau,
                     gyro_tau       = gyro_tau,
                     save_h5        = save_h5,
                     ins_h5         = xyz_h5)

    # create compensated (clean) scalar magnetometer measurements
    mag_1_c = create_mag_c(traj,mapS;
                           meas_var   = cor_var,
                           fogm_sigma = fogm_sigma,
                           fogm_tau   = fogm_tau,
                           silent     = silent)

    # create compensated (clean) vector magnetometer measurements
    flux_a = create_flux(traj,mapV;
                         meas_var   = cor_var,
                         fogm_sigma = fogm_sigma,
                         fogm_tau   = fogm_tau,
                         silent     = silent)

    # create uncompensated (corrupted) scalar magnetometer measurements
    (mag_1_uc,_) = corrupt_mag(mag_1_c,flux_a;
                               dt           = dt,
                               cor_sigma    = cor_sigma,
                               cor_tau      = cor_tau,
                               cor_var      = cor_var,
                               cor_drift    = cor_drift,
                               cor_perm_mag = cor_perm_mag,
                               cor_ind_mag  = cor_ind_mag,
                               cor_eddy_mag = cor_eddy_mag)

    flights = flight*one.(traj.lat)
    lines   = line  *one.(traj.lat)

    if save_h5 # save `xyz_h5`
        h5open(xyz_h5,"cw") do file # read-write, create file if not existing, preserve existing contents
            write(file,"flux_a_x",flux_a.x)
            write(file,"flux_a_y",flux_a.y)
            write(file,"flux_a_z",flux_a.z)
            write(file,"flux_a_t",flux_a.t)
            write(file,"mag_1_uc",mag_1_uc)
            write(file,"mag_1_c" ,mag_1_c)
            write(file,"flight"  ,flights)
            write(file,"line"    ,lines)
        end
    end

    return XYZ0(traj, ins, flux_a, flights, lines, mag_1_c, mag_1_uc)
end # function create_XYZ0

"""
    create_traj(mapS::Union{MapS,MapSd,MapS3D} = get_map(namad);
                alt             = 1000,
                dt              = 0.1,
                t               = 300,
                v               = 68,
                ll1             = (),
                ll2             = (),
                N_waves         = 1,
                attempts::Int   = 10,
                save_h5::Bool   = false,
                traj_h5::String = "traj_data.h5")

Create `Traj` trajectory struct with a straight or sinusoidal flight path at
constant altitude (2D flight). May create a trajectory that passes over
map areas that are missing map data if an artificially filled-in map is used.

**Arguments:**
- `mapS`:     (optional) `MapS`, `MapSd`, or `MapS3D` scalar magnetic anomaly map struct
- `alt`:      (optional) altitude  [m]
- `dt`:       (optional) measurement time step [s]
- `t`:        (optional) total flight time, ignored if `ll2` is set [s]
- `v`:        (optional) approximate aircraft velocity [m/s]
- `ll1`:      (optional) inital (lat,lon) point [deg]
- `ll2`:      (optional) final  (lat,lon) point [deg]
- `N_waves`:  (optional) number of sine waves along path
- `attempts`: (optional) maximum attempts at creating flight path on `mapS`
- `save_h5`:  (optional) if true, save `traj` to `traj_h5`
- `traj_h5`:  (optional) path/name of trajectory data HDF5 file to save (`.h5` extension optional)

**Returns:**
- `traj`: `Traj` trajectory struct
"""
function create_traj(mapS::Union{MapS,MapSd,MapS3D} = get_map(namad);
                     alt             = 1000,
                     dt              = 0.1,
                     t               = 300,
                     v               = 68,
                     ll1             = (),
                     ll2             = (),
                     N_waves         = 1,
                     attempts::Int   = 10,
                     save_h5::Bool   = false,
                     traj_h5::String = "traj_data.h5")

    traj_h5 = add_extension(traj_h5,".h5")

    # check flight altitude
    ind  = map_params(mapS)[2]
    alt_ = typeof(mapS) <: Union{MapSd} ? median(mapS.alt[ind]) : mapS.alt[1]
    alt  < alt_ && error("flight altitude $alt < map altitude $alt_")

    i   = 0
    N   = 2
    lat = zeros(eltype(mapS.xx),N)
    lon = zeros(eltype(mapS.xx),N)
    while (!map_check(mapS,lat,lon) & (i <= attempts)) | (i == 0)
        i += 1

        if isempty(ll1) # put initial point in middle 50% of map
            (lat_min,lat_max) = extrema(mapS.yy)
            (lon_min,lon_max) = extrema(mapS.xx)
            lat1 = lat_min + (lat_max - lat_min) * (0.25 + 0.50*rand())
            lon1 = lon_min + (lon_max - lon_min) * (0.25 + 0.50*rand())
        else # use given inital point
            (lat1,lon1) = deg2rad.(ll1)
        end

        if isempty(ll2) # use given velocity & time to set distance
            N     = round(Int,t/dt+1)
            d     = v*t # distance
            θ_utm = 2*pi*rand() # random heading in utm coordinates
            lat2  = lat1 + dn2dlat(d*sin(θ_utm),lat1)
            lon2  = lon1 + de2dlon(d*cos(θ_utm),lat1)
        else # use given final point directly
            N = 1000*(N_waves+1) # estimated N, to be corrected
            (lat2,lon2) = deg2rad.(ll2)
        end

        dlat = lat2 - lat1 # straight line Δ latitude,  to be corrected
        dlon = lon2 - lon1 # straight line Δ longitude, to be corrected

        θ_ll = atan(dlat,dlon) # heading in lat & lon coordinates

        lat  = [LinRange(lat1,lat2,N);] # initial latitudes
        lon  = [LinRange(lon1,lon2,N);] # initial longitudes

        # p1 = plot(lon,lat)
        # plot!(p1,[lon[1],lon[1]],[lat[1],lat[1]],markershape=:circle)

        if N_waves > 0
            ϕ   = LinRange(0,N_waves*2*pi,N) # waves steps
            wav = [ϕ sin.(ϕ)] # waves
            rot = [cos(θ_ll) -sin(θ_ll); sin(θ_ll) cos(θ_ll)] # rotation matrix
            cor = wav*rot' # waves correction after rotation
            lat = (cor[:,2] .- cor[1,2] .+ lat1) # latitude  with waves
            lon = (cor[:,1] .- cor[1,1] .+ lon1) # longitude with waves
            # plot(wav[:,1],wav[:,2])
            # plot(cor[:,1],cor[:,2])
        end

        frac1 = 0
        frac2 = 0
        while !(frac1 ≈ 1) | !(frac2 ≈ 1) # typically 4-5 iterations
            dx = dlon2de.(fdm(lon),lat) # easting  distance per time step
            dy = dlat2dn.(fdm(lat),lat) # northing distance per time step
            d_now = sum(sqrt.(dx[2:N].^2+dy[2:N].^2)) # current distance

            if isempty(ll2) # scale to target distance
                frac1 = d / d_now # scaling factor
                frac2 = frac1
            else  # scale to target end point
                frac1 = (lat2 - lat[1])/(lat[end] - lat[1]) # scaling factor
                frac2 = (lon2 - lon[1])/(lon[end] - lon[1]) # scaling factor
            end

            lat = (lat .- lat[1])*frac1 .+ lat[1] # scale to target
            lon = (lon .- lon[1])*frac2 .+ lon[1] # scale to target
        end

        dx = dlon2de.(fdm(lon),lat) # easting  distance per time step
        dy = dlat2dn.(fdm(lat),lat) # northing distance per time step
        d_now = sum(sqrt.(dx[2:N].^2+dy[2:N].^2)) # current distance

        if ll2 != () # correct time & N for true distance & given velocity
            range_old = LinRange(0,1,N) # starting range {0:1} with N_old
            t         = d_now/v # true time from true distance & given velocity
            N         = round(Int,t/dt+1) # time steps needed for t with dt
            range_new = LinRange(0,1,N) # ending range {0:1} with N_new
            itp_lat   = linear_interpolation(range_old,lat) # lat itp = f(0:1)
            itp_lon   = linear_interpolation(range_old,lon) # lon itp = f(0:1)
            lat       = itp_lat.(range_new) # get interpolated lat
            lon       = itp_lon.(range_new) # get interpolated lon
        end

        # plot!(p1,lon,lat)

    end

    # println(mapS.xx[1]," ",mapS.xx[end]," ",mapS.yy[1]," ",mapS.yy[end])
    # println(lon[1]," ",lon[end]," ",lat[1]," ",lat[end])
    # println(extrema(lon)," ",extrema(lat))
    # println(map_check(mapS,lat,lon))
    # println(i)
    # println(attempts)

    @assert i <= attempts "maximum attempts reached, decrease t or increase v"

    (zone_utm,is_north) = utm_zone(mean(rad2deg.(lat)),mean(rad2deg.(lon)))
    lla2utm = UTMfromLLA(zone_utm,is_north,WGS84)
    utms    = lla2utm.(LLA.(rad2deg.(lat),rad2deg.(lon)))

    # velocities and specific forces from position
    vn = fdm([utm.y for utm in utms]) / dt
    ve = fdm([utm.x for utm in utms]) / dt
    vd = zero(lat)
    fn = fdm(vn) / dt
    fe = fdm(ve) / dt
    fd = fdm(vd) / dt .- g_earth
    tt = [LinRange(0,t,N);]

    # direction cosine matrix (body to navigation) estimate from heading
    Cnb = create_dcm(vn,ve,dt,:body2nav)
    (roll,pitch,yaw) = dcm2euler(Cnb,:body2nav)

    if save_h5 # save `traj_h5`
        h5open(traj_h5,"cw") do file # read-write, create file if not existing, preserve existing contents
            write(file,"tt"   ,tt)
            write(file,"lat"  ,lat)
            write(file,"lon"  ,lon)
            write(file,"alt"  ,alt*one.(lat))
            write(file,"vn"   ,vn)
            write(file,"ve"   ,ve)
            write(file,"vd"   ,vd)
            write(file,"fn"   ,fn)
            write(file,"fe"   ,fe)
            write(file,"fd"   ,fd)
            write(file,"roll" ,roll)
            write(file,"pitch",pitch)
            write(file,"yaw"  ,yaw)
        end
    end

    return Traj(N, dt, tt, lat, lon, alt*one.(lat), vn, ve, vd, fn, fe, fd, Cnb)
end # function create_traj

"""
    create_ins(traj::Traj;
               init_pos_sigma = 3.0,
               init_alt_sigma = 0.001,
               init_vel_sigma = 0.01,
               init_att_sigma = deg2rad(0.01),
               VRW_sigma      = 0.000238,
               ARW_sigma      = 0.000000581,
               baro_sigma     = 1.0,
               ha_sigma       = 0.001,
               a_hat_sigma    = 0.01,
               acc_sigma      = 0.000245,
               gyro_sigma     = 0.00000000727,
               baro_tau       = 3600.0,
               acc_tau        = 3600.0,
               gyro_tau       = 3600.0,
               save_h5::Bool  = false,
               ins_h5::String = "ins_data.h5")

Creates an INS trajectory about a true trajectory. Propagates a 17-state
Pinson error model to create INS errors.

**Arguments:**
- `traj`:           `Traj` trajectory struct
- `init_pos_sigma`: (optional) initial position uncertainty [m]
- `init_alt_sigma`: (optional) initial altitude uncertainty [m]
- `init_vel_sigma`: (optional) initial velocity uncertainty [m/s]
- `init_att_sigma`: (optional) initial attitude uncertainty [rad]
- `VRW_sigma`:      (optional) velocity random walk [m/s^2 /sqrt(Hz)]
- `ARW_sigma`:      (optional) angular random walk [rad/s /sqrt(Hz)]
- `baro_sigma`:     (optional) barometer bias [m]
- `ha_sigma`:       (optional) barometer aiding altitude bias [m]
- `a_hat_sigma`:    (optional) barometer aiding vertical accel bias [m/s^2]
- `acc_sigma`:      (optional) accelerometer bias [m/s^2]
- `gyro_sigma`:     (optional) gyroscope bias [rad/s]
- `baro_tau`:       (optional) barometer time constant [s]
- `acc_tau`:        (optional) accelerometer time constant [s]
- `gyro_tau`:       (optional) gyroscope time constant [s]
- `save_h5`:        (optional) if true, save `ins` to `ins_h5`
- `ins_h5`:         (optional) path/name of INS data HDF5 file to save (`.h5` extension optional)

**Returns:**
- `ins`: `INS` inertial navigation system struct
"""
function create_ins(traj::Traj;
                    init_pos_sigma = 3.0,
                    init_alt_sigma = 0.001,
                    init_vel_sigma = 0.01,
                    init_att_sigma = deg2rad(0.00001),
                    VRW_sigma      = 0.000238,
                    ARW_sigma      = 0.000000581,
                    baro_sigma     = 1.0,
                    ha_sigma       = 0.001,
                    a_hat_sigma    = 0.01,
                    acc_sigma      = 0.000245,
                    gyro_sigma     = 0.00000000727,
                    baro_tau       = 3600.0,
                    acc_tau        = 3600.0,
                    gyro_tau       = 3600.0,
                    save_h5::Bool  = false,
                    ins_h5::String = "ins_data.h5")

    ins_h5 = add_extension(ins_h5,".h5")

    N  = traj.N
    dt = traj.dt
    nx = 17 # total state dimension (inherent to this model)

    (P0,Qd,_) = create_model(dt,traj.lat[1];
                             init_pos_sigma = init_pos_sigma,
                             init_alt_sigma = init_alt_sigma,
                             init_vel_sigma = init_vel_sigma,
                             init_att_sigma = init_att_sigma,
                             VRW_sigma      = VRW_sigma,
                             ARW_sigma      = ARW_sigma,
                             baro_sigma     = baro_sigma,
                             ha_sigma       = ha_sigma,
                             a_hat_sigma    = a_hat_sigma,
                             acc_sigma      = acc_sigma,
                             gyro_sigma     = gyro_sigma,
                             baro_tau       = baro_tau,
                             acc_tau        = acc_tau,
                             gyro_tau       = gyro_tau,
                             fogm_state     = false)
    P   = zeros(nx,nx,N)
    err = zeros(nx,N)

    P[:,:,1] = P0
    err[:,1] = rand(MvNormal(P0),1) # mean = 0, covariance = P0
    Q_chol   = chol(Qd)

    for k = 1:N-1
        Phi = get_Phi(nx,traj.lat[k],traj.vn[k],traj.ve[k],traj.vd[k],
                      traj.fn[k],traj.fe[k],traj.fd[k],traj.Cnb[:,:,k],
                      baro_tau,acc_tau,gyro_tau,0,dt;fogm_state=false)
        err[:,k+1] = Phi*err[:,k] + Q_chol*randn(nx)
        P[:,:,k+1] = Phi*P[:,:,k]*Phi' + Qd
    end

    # for debugging INS Cnb
    # println("INS_err_n ",round.(rad2deg.(extrema(err[7,:])),digits=5)," deg")
    # println("INS_err_e ",round.(rad2deg.(extrema(err[8,:])),digits=5)," deg")
    # println("INS_err_d ",round.(rad2deg.(extrema(err[9,:])),digits=5)," deg")

    tt  = traj.tt
    lat = traj.lat - err[1,:]
    lon = traj.lon - err[2,:]
    alt = traj.alt - err[3,:]
    vn  = traj.vn  - err[4,:]
    ve  = traj.ve  - err[5,:]
    vd  = traj.vd  - err[6,:]
    fn  = fdm(vn) / dt
    fe  = fdm(ve) / dt
    fd  = fdm(vd) / dt .- g_earth
    Cnb = correct_Cnb(traj.Cnb, -err[7:9,:])
    if any(Cnb .> 1) | any(Cnb .< -1)
        error("create_ins() failed, likely due to bad trajectory data, re-run")
    else
        (roll,pitch,yaw) = dcm2euler(Cnb,:body2nav)
    end

    if save_h5 # save `ins_h5`
        h5open(ins_h5,"cw") do file # read-write, create file if not existing, preserve existing contents
            write(file,"ins_tt"   ,tt)
            write(file,"ins_lat"  ,lat)
            write(file,"ins_lon"  ,lon)
            write(file,"ins_alt"  ,alt)
            write(file,"ins_vn"   ,vn)
            write(file,"ins_ve"   ,ve)
            write(file,"ins_vd"   ,vd)
            write(file,"ins_fn"   ,fn)
            write(file,"ins_fe"   ,fe)
            write(file,"ins_fd"   ,fd)
            write(file,"ins_roll" ,roll)
            write(file,"ins_pitch",pitch)
            write(file,"ins_yaw"  ,yaw)
            write(file,"ins_P"    ,P)
        end
    end

    return INS(N, dt, tt, lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb, P)
end # function create_ins

"""
    create_mag_c(lat, lon, mapS::Union{MapS,MapSd,MapS3D} = get_map(namad);
                 alt          = 1000,
                 dt           = 0.1,
                 meas_var     = 1.0^2,
                 fogm_sigma   = 1.0,
                 fogm_tau     = 600.0,
                 silent::Bool = false)

Create compensated (clean) scalar magnetometer measurements from a scalar
magnetic anomaly map.

**Arguments:**
- `lat`:        latitude  [rad]
- `lon`:        longitude [rad]
- `mapS`:       (optional) `MapS`, `MapSd`, or `MapS3D` scalar magnetic anomaly map struct
- `alt`:        (optional) altitude  [m]
- `dt`:         (optional) measurement time step [s]
- `meas_var`:   (optional) measurement (white) noise variance [nT^2]
- `fogm_sigma`: (optional) FOGM catch-all bias [nT]
- `fogm_tau`:   (optional) FOGM catch-all time constant [s]
- `silent`:     (optional) if true, no print outs

**Returns:**
- `mag_c`: compensated (clean) scalar magnetometer measurements [nT]
"""
function create_mag_c(lat, lon, mapS::Union{MapS,MapSd,MapS3D} = get_map(namad);
                      alt          = 1000,
                      dt           = 0.1,
                      meas_var     = 1.0^2,
                      fogm_sigma   = 1.0,
                      fogm_tau     = 600.0,
                      silent::Bool = false)

    # convert MapS3D to MapS at alt
    typeof(mapS) <: MapS3D && (mapS = upward_fft(mapS,alt))

    N = length(lat)
    (ind0,ind1,_,_) = map_params(mapS)

    # fill map if >1% unfilled
    if sum(ind0)/sum(ind0+ind1) > 0.01
        silent || @info("filling in scalar map")
        mapS = map_fill(map_trim(mapS))
    end

    # map values along trajectory
    silent || @info("getting scalar map values, possibly with upward/downward continuation")
    map_val = get_map_val(mapS,lat,lon,alt;α=200)

    # FOGM & white noise
    silent || @info("adding FOGM & white noise to scalar map values")
    mag_c = map_val + fogm(fogm_sigma,fogm_tau,dt,N) + sqrt(meas_var)*randn(N)

    return (mag_c)
end # function create_mag_c

"""
    create_mag_c(path::Path, mapS::Union{MapS,MapSd,MapS3D} = get_map(namad);
                 meas_var     = 1.0^2,
                 fogm_sigma   = 1.0,
                 fogm_tau     = 600.0,
                 silent::Bool = false)

Create compensated (clean) scalar magnetometer measurements from a scalar
magnetic anomaly map.

**Arguments:**
- `path`:       `Path` struct, i.e., `Traj` trajectory struct, `INS` inertial navigation system struct, or `FILTout` filter extracted output struct
- `mapS`:       (optional) `MapS`, `MapSd`, or `MapS3D` scalar magnetic anomaly map struct
- `meas_var`:   (optional) measurement (white) noise variance [nT^2]
- `fogm_sigma`: (optional) FOGM catch-all bias [nT]
- `fogm_tau`:   (optional) FOGM catch-all time constant [s]
- `silent`:     (optional) if true, no print outs

**Returns:**
- `mag_c`: compensated (clean) scalar magnetometer measurements [nT]
"""
function create_mag_c(path::Path, mapS::Union{MapS,MapSd,MapS3D} = get_map(namad);
                      meas_var     = 1.0^2,
                      fogm_sigma   = 1.0,
                      fogm_tau     = 600.0,
                      silent::Bool = false)
    create_mag_c(path.lat,path.lon,mapS;
                 alt        = median(path.alt),
                 dt         = path.dt,
                 meas_var   = meas_var,
                 fogm_sigma = fogm_sigma,
                 fogm_tau   = fogm_tau,
                 silent     = silent)
end # function create_mag_c

"""
    corrupt_mag(mag_c, Bx, By, Bz;
                dt           = 0.1,
                cor_sigma    = 1.0,
                cor_tau      = 600.0,
                cor_var      = 1.0^2,
                cor_drift    = 0.001,
                cor_perm_mag = 5.0,
                cor_ind_mag  = 5.0,
                cor_eddy_mag = 0.5)

Corrupt compensated (clean) magnetometer measurements with random, FOGM, drift,
and Tolles-Lawson noise to create uncompensated (corrupted) scalar magnetometer
measurements. FOGM is a First-order Gauss-Markov stochastic process.

**Arguments:**
- `mag_c`:        compensated (clean) scalar magnetometer measurements [nT]
- `Bx`,`By`,`Bz`: vector magnetometer measurements [nT]
- `dt`:           (optional) measurement time step [s]
- `cor_sigma`:    (optional) corruption FOGM catch-all bias [nT]
- `cor_tau`:      (optional) corruption FOGM catch-all time constant [s]
- `cor_var`:      (optional) corruption measurement (white) noise variance [nT^2]
- `cor_drift`:    (optional) corruption measurement linear drift [nT/s]
- `cor_perm_mag`: (optional) corruption permanent field TL coef std dev
- `cor_ind_mag`:  (optional) corruption induced field TL coef std dev
- `cor_eddy_mag`: (optional) corruption eddy current TL coef std dev

**Returns:**
- `mag_uc`:  uncompensated (corrupted) scalar magnetometer measurements [nT]
- `TL_coef`: Tolles-Lawson coefficients (partially) used to create `mag_uc`
"""
function corrupt_mag(mag_c, Bx, By, Bz;
                     dt           = 0.1,
                     cor_sigma    = 1.0,
                     cor_tau      = 600.0,
                     cor_var      = 1.0^2,
                     cor_drift    = 0.001,
                     cor_perm_mag = 5.0,
                     cor_ind_mag  = 5.0,
                     cor_eddy_mag = 0.5)

    P = Diagonal(vcat(repeat([cor_perm_mag],3),
                      repeat([cor_ind_mag ],6),
                      repeat([cor_eddy_mag],9))).^2

    # sample from MvNormal distribution with mean = 0, covariance = P
    TL_coef = vec(rand(MvNormal(P),1))

    N = length(mag_c)

    mag_uc = mag_c + sqrt(cor_var)*randn(N) +
                     fogm(cor_sigma,cor_tau,dt,N) +
                     cor_drift*rand()*(0:dt:dt*(N-1))

    # corrupt with TL if vector magnetometer measurements are non-zero
    ~iszero(Bx) && (mag_uc += create_TL_A(Bx,By,Bz)*TL_coef)

    return (mag_uc, TL_coef)
end # function corrupt_mag

"""
    corrupt_mag(mag_c, flux;
                dt           = 0.1,
                cor_sigma    = 1.0,
                cor_tau      = 600.0,
                cor_var      = 1.0^2,
                cor_drift    = 0.001,
                cor_perm_mag = 5.0,
                cor_ind_mag  = 5.0,
                cor_eddy_mag = 0.5)

Corrupt compensated (clean) magnetometer measurements with random, FOGM, drift,
and Tolles-Lawson noise to create uncompensated (corrupted) scalar magnetometer
measurements. FOGM is a First-order Gauss-Markov stochastic process.

**Arguments:**
- `mag_c`:        compensated (clean) scalar magnetometer measurements [nT]
- `flux`:         `MagV` vector magnetometer measurement struct
- `dt`:           (optional) measurement time step [s]
- `cor_sigma`:    (optional) corruption FOGM catch-all bias [nT]
- `cor_tau`:      (optional) corruption FOGM catch-all time constant [s]
- `cor_var`:      (optional) corruption measurement (white) noise variance [nT^2]
- `cor_drift`:    (optional) corruption measurement linear drift [nT/s]
- `cor_perm_mag`: (optional) corruption permanent field TL coef std dev
- `cor_ind_mag`:  (optional) corruption induced field TL coef std dev
- `cor_eddy_mag`: (optional) corruption eddy current TL coef std dev

**Returns:**
- `mag_uc`:  uncompensated (corrupted) scalar magnetometer measurements [nT]
- `TL_coef`: Tolles-Lawson coefficients (partially) used to create `mag_uc`
"""
function corrupt_mag(mag_c, flux;
                     dt           = 0.1,
                     cor_sigma    = 1.0,
                     cor_tau      = 600.0,
                     cor_var      = 1.0^2,
                     cor_drift    = 0.001,
                     cor_perm_mag = 5.0,
                     cor_ind_mag  = 5.0,
                     cor_eddy_mag = 0.5)
    corrupt_mag(mag_c,flux.x,flux.y,flux.z;
                dt           = dt,
                cor_sigma    = cor_sigma,
                cor_tau      = cor_tau,
                cor_var      = cor_var,
                cor_drift    = cor_drift,
                cor_perm_mag = cor_perm_mag,
                cor_ind_mag  = cor_ind_mag,
                cor_eddy_mag = cor_eddy_mag)
end # function corrupt_mag

"""
    create_flux(lat, lon, mapV::MapV = get_map(emm720);
                Cnb          = repeat(I(3),1,1,length(lat)),
                alt          = 1000,
                dt           = 0.1,
                meas_var     = 1.0^2,
                fogm_sigma   = 1.0,
                fogm_tau     = 600.0,
                silent::Bool = false)

Create compensated (clean) vector magnetometer measurements from a vector
magnetic anomaly map.

**Arguments:**
- `lat`:        latitude  [rad]
- `lon`:        longitude [rad]
- `mapV`:       (optional) `MapV` vector magnetic anomaly map struct
- `Cnb`:        (optional) direction cosine matrix (body to navigation) [-]
- `alt`:        (optional) altitude  [m]
- `dt`:         (optional) measurement time step [s]
- `meas_var`:   (optional) measurement (white) noise variance [nT^2]
- `fogm_sigma`: (optional) FOGM catch-all bias [nT]
- `fogm_tau`:   (optional) FOGM catch-all time constant [s]
- `silent`:     (optional) if true, no print outs

**Returns:**
- `flux`: `MagV` vector magnetometer measurement struct
"""
function create_flux(lat, lon, mapV::MapV = get_map(emm720);
                     Cnb          = repeat(I(3),1,1,length(lat)),
                     alt          = 1000,
                     dt           = 0.1,
                     meas_var     = 1.0^2,
                     fogm_sigma   = 1.0,
                     fogm_tau     = 600.0,
                     silent::Bool = false)

    N = length(lat)

    # map values along trajectory
    silent || @info("getting vector map values, possibly with upward/downward continuation")
    (Bx,By,Bz) = get_map_val(mapV,lat,lon,alt;α=200)

    # FOGM & white noise
    silent || @info("adding FOGM & white noise to vector map values")
    Bx += fogm(fogm_sigma,fogm_tau,dt,N) + sqrt(meas_var)*randn(N)
    By += fogm(fogm_sigma,fogm_tau,dt,N) + sqrt(meas_var)*randn(N)
    Bz += fogm(fogm_sigma,fogm_tau,dt,N) + sqrt(meas_var)*randn(N)
    Bt  = sqrt.(Bx.^2+By.^2+Bz.^2)

    # put measurements into body frame
    for i = 1:N
        (Bx[i],By[i],Bz[i]) = Cnb[:,:,i]' * [Bx[i],By[i],Bz[i]]
    end

    return MagV(Bx, By, Bz, Bt)
end # function create_flux

"""
    create_flux(path::Path, mapV::MapV = get_map(emm720);
                meas_var     = 1.0^2,
                fogm_sigma   = 1.0,
                fogm_tau     = 600.0,
                silent::Bool = false)

Create compensated (clean) vector magnetometer measurements from a vector
magnetic anomaly map.

**Arguments:**
- `path`:       `Path` struct, i.e., `Traj` trajectory struct, `INS` inertial navigation system struct, or `FILTout` filter extracted output struct
- `mapV`:       (optional) `MapV` vector magnetic anomaly map struct
- `meas_var`:   (optional) measurement (white) noise variance [nT^2]
- `fogm_sigma`: (optional) FOGM catch-all bias [nT]
- `fogm_tau`:   (optional) FOGM catch-all time constant [s]
- `silent`:     (optional) if true, no print outs

**Returns:**
- `flux`: `MagV` vector magnetometer measurement struct
"""
function create_flux(path::Path, mapV::MapV = get_map(emm720);
                     meas_var     = 1.0^2,
                     fogm_sigma   = 1.0,
                     fogm_tau     = 600.0,
                     silent::Bool = false)
    create_flux(path.lat,path.lon,mapV;
                Cnb        = path.Cnb,
                alt        = median(path.alt),
                dt         = path.dt,
                meas_var   = meas_var,
                fogm_sigma = fogm_sigma,
                fogm_tau   = fogm_tau,
                silent     = silent)
end # function create_flux

"""
    create_dcm(vn, ve, dt=0.1, order::Symbol=:body2na)

Internal helper function to estimate a direction cosine matrix using known
heading with FOGM noise.

**Arguments:**
- `vn`:    length `N` north velocity [m/s]
- `ve`:    length `N` east  velocity [m/s]
- `dt`:    (optional) measurement time step [s]
- `order`: (optional) rotation order {`:body2nav`,`:nav2body`}

**Returns:**
- `dcm`: `3` x `3` x `N` direction cosine matrix [-]
"""
function create_dcm(vn, ve, dt=0.1, order::Symbol=:body2nav)

    N     = length(vn)
    roll  = fogm(deg2rad(2  ),2,dt,N)
    pitch = fogm(deg2rad(0.5),2,dt,N)
    yaw   = fogm(deg2rad(1  ),2,dt,N)

    bpf   = get_bpf(;pass1=1e-6,pass2=1)
    roll  = bpf_data(roll ;bpf=bpf)
    pitch = bpf_data(pitch;bpf=bpf) .+ deg2rad(2)   # pitch typically ~2 deg
    yaw   = bpf_data(yaw  ;bpf=bpf) .+ atan.(ve,vn) # yaw definition
    dcm   = euler2dcm(roll,pitch,yaw,order)

    return (dcm)
end # function create_dcm

"""
    calculate_imputed_TL_earth(xyz::Union{XYZ1,XYZ20,XYZ21}, ind,
                               map_val, set_igrf::Bool, TL_coef;
                               terms    = [:permanent,:induced,:eddy],
                               Bt_scale = 50000)

Internal helper function to get the imputed Earth vector between two locations.

**Arguments:**
- `xyz`:      `XYZ` flight data struct
- `ind`:      selected data indices
- `map_val`:  scalar magnetic anomaly map values
- `set_igrf`: if true, set the `igrf` field in `xyz`
- `TL_coef`:  Tolles-Lawson coefficients
- `terms`:    (optional) Tolles-Lawson terms to use {`:permanent`,`:induced`,`:eddy`}
- `Bt_scale`: (optional) scaling factor for induced and eddy current terms [nT]

**Returns:**
- `TL_aircraft`: `3` x `N` matrix of TL aircraft vector field
- `B_earth`:     `3` x `N` matrix of Earth vector field
"""
function calculate_imputed_TL_earth(xyz::Union{XYZ1,XYZ20,XYZ21}, ind,
                                    map_val, set_igrf::Bool, TL_coef;
                                    terms    = [:permanent,:induced,:eddy],
                                    Bt_scale = 50000)

    # get IGRF from model
    igrf_vec  = get_igrf(xyz,ind;frame=:body,norm_igrf=false,check_xyz=!set_igrf)
    set_igrf && (xyz.igrf[ind] .= norm.(igrf_vec))

    # obtain scalar map values with IGRF for this trajectory
    B_earth = map_val + norm.(igrf_vec) # no diurnal

    # impute vector quantity using scaled IGRF vector field
    B_earth = reduce(hcat, B_earth .* normalize.(igrf_vec))

    # time-derivative of vector field
    B_earth_dot = [fdm(B_earth[1,:]) fdm(B_earth[2,:]) fdm(B_earth[3,:])]'

    # Earth-only contribution to aircraft field
    (TL_coef_p,TL_coef_i,TL_coef_e) = TL_vec2mat(TL_coef,terms;Bt_scale=Bt_scale)
    TL_aircraft = get_TL_aircraft_vec(B_earth,B_earth_dot,TL_coef_p,TL_coef_i,TL_coef_e)

    return (TL_aircraft, B_earth)
end # function calculate_imputed_TL_earth

"""
    create_informed_xyz(xyz::Union{XYZ1,XYZ20,XYZ21}, ind, mapS::Union{MapS,MapSd,MapS3D},
                        use_mag::Symbol, use_vec::Symbol, TL_coef::Vector;
                        terms::Vector{Symbol} = [:permanent,:induced,:eddy],
                        disp_min = 100,
                        disp_max = 500,
                        Bt_disp  = 50,
                        Bt_scale = 50000)

Create knowledge-informed data from existing data. Given map information, an
`XYZ` structure with magnetometer readings, and a fitted Tolles-Lawson model,
this function creates a consistent, displaced `XYZ` structure representing what
the `use_mag` and `mag_1_c` would have collected had the entire flight been
laterally shifted by (`disp_min`,`disp_max`), assuming that the linear aircraft
model is reasonably accurate. It makes use of a Taylor expansion of the map
information to update the expected changes due to the alternative map location
*and* due to the imputed aircraft field. The aircraft "noise" is then carried
over into `use_mag` in the new `XYZ` data.

**Arguments:**
- `xyz`:      (original) `XYZ` flight data struct
- `ind`:      selected data indices
- `mapS`:     `MapS`, `MapSd`, or `MapS3D` scalar magnetic anomaly map struct
- `use_mag`:  scalar magnetometer to use {`:mag_1_uc`, etc.}
- `use_vec`:  vector magnetometer (fluxgate) to use for Tolles-Lawson `A` matrix {`:flux_a`, etc.}
- `TL_coef`:  Tolles-Lawson coefficients
- `terms`:    (optional) Tolles-Lawson terms to use {`:permanent`,`:induced`,`:eddy`}
- `disp_min`: (optional) minimum trajectory displacement [m]
- `disp_max`: (optional) maximum trajectory displacement [m]
- `Bt_disp`:  (optional) target total magnetic field magnitude displacement offset [nT]
- `Bt_scale`: (optional) scaling factor for induced and eddy current terms [nT]

**Returns:**
- `xyz_disp`: `XYZ` flight data struct with displaced trajectory and modified magnetometer readings
"""
function create_informed_xyz(xyz::Union{XYZ1,XYZ20,XYZ21}, ind, mapS::Union{MapS,MapSd,MapS3D},
                             use_mag::Symbol, use_vec::Symbol, TL_coef::Vector;
                             terms::Vector{Symbol} = [:permanent,:induced,:eddy],
                             disp_min = 100,
                             disp_max = 500,
                             Bt_disp  = 50,
                             Bt_scale = 50000)

    @assert any([:permanent,:p,:permanent3,:p3] .∈ (terms,)) "permanent terms are required"
    @assert any([:induced,:i,:induced6,:i6,:induced5,:i5,:induced3,:i3] .∈ (terms,)) "induced terms are required"
    @assert any([:eddy,:e,:eddy9,:e9,:eddy8,:e8,:eddy3,:e3] .∈ (terms,)) "eddy current terms are required"
    @assert !any([:fdm,:f,:fdm3,:f3,:bias,:b] .∈ (terms,)) "derivative and bias terms may not be used"

    N = length(TL_coef)
    A_test = create_TL_A([1.0],[1.0],[1.0];terms=terms)
    @assert N == length(A_test) "TL_coef does not agree with specified terms"

    traj = get_traj(xyz,ind)

    @assert map_check(mapS,traj) "trajectory must be inside the provided map"

    # map values along trajectory & map
    (map_val,itp_mapS) = get_map_val(mapS,traj;α=200,return_itp=true)

    # compute vector aircraft component and vector flux along trajectory
    set_igrf = false
    (TL_aircraft,B_earth) =
        calculate_imputed_TL_earth(xyz,ind,map_val,set_igrf,TL_coef,
                                   terms    = terms,
                                   Bt_scale = Bt_scale)

    # sample ~100 points along trajectory
    spacing = floor(Int, traj.N / 100)
    pts     = 1:spacing:traj.N

    # figure out direction from trajectory to middle of map
    mean_traj_lat_lon = [mean(traj.lon[pts]),mean(traj.lat[pts])]
    mean_map_lat_lon  = [(mapS.xx[1] + mapS.xx[end]) / 2,
                         (mapS.yy[1] + mapS.yy[end]) / 2]
    traj_to_map_mid   = mean_map_lat_lon - mean_traj_lat_lon

    # average x/y gradients [nT/rad] of sampled points
    avg_grad = mean(map((x,y) -> collect(gradient(itp_mapS,x,y)),
                    traj.lon[pts],traj.lat[pts]))
    dir_disp = normalize(avg_grad)

    # switch direction to go toward middle of map if necessary
    (dot(traj_to_map_mid,dir_disp) < 0) && (dir_disp *= -1)

    # convert displacement limits from [m] to [rad]
    disp_min = min(dn2dlat(disp_min,mean_traj_lat_lon[2]),
                   de2dlon(disp_min,mean_traj_lat_lon[2]))
    disp_max = max(dn2dlat(disp_max,mean_traj_lat_lon[2]),
                   de2dlon(disp_max,mean_traj_lat_lon[2]))

    # shoot for difference of Bt_disp [nT]
    dir_diriv = abs(dot(avg_grad,dir_disp)) # [nT/rad]
    disp_rad  = Bt_disp / dir_diriv # [rad]
    disp_rad  = clamp(disp_rad,disp_min,disp_max) # limit displacement range
    disp_ll   = disp_rad * dir_disp # set correct direction

    # copy and displace trajectory (uniformally; no acceleration changes!)
    xyz_disp = deepcopy(xyz)
    xyz_disp.traj.lon[ind] .+= disp_ll[1]
    xyz_disp.traj.lat[ind] .+= disp_ll[2]

    @assert map_check(mapS,xyz_disp.traj(ind)) "larger map needed, could not create trajectory"

    # map values along trajectory
    map_val_disp = get_map_val(mapS,xyz_disp.traj(ind);α=200)

    # calculate Earth's vector flux and TL component from that on new trajectory
    set_igrf = true
    (TL_aircraft_disp,B_earth_disp) =
        calculate_imputed_TL_earth(xyz_disp,ind,map_val_disp,set_igrf,TL_coef,
                                   terms    = terms,
                                   Bt_scale = Bt_scale)

    # calculate Earth-induced field that would occur along this trajectory
    ΔB_TL    = TL_aircraft_disp - TL_aircraft # known part from aircraft
    ΔB_earth = B_earth_disp     - B_earth     # known part from Earth
    ΔB       = ΔB_TL + ΔB_earth # total difference in vector field from different Earth locale and aircraft
    Δmap_val = map_val_disp - map_val # differnece in map values

    # update displaced vector magnetometer values used in learning
    flux = getfield(xyz_disp,use_vec)
    flux.x[ind] += ΔB[1,:]
    flux.y[ind] += ΔB[2,:]
    flux.z[ind] += ΔB[3,:]
    flux.t[ind]  = sqrt.(flux.x[ind].^2 .+ flux.y[ind].^2 .+ flux.z[ind].^2)

    # update displaced scalar magnetometer values used in learning
    ΔB_dot = dot.(eachcol(ΔB),[[x,y,z] for (x,y,z) in
                  zip(flux.x[ind],flux.y[ind],flux.z[ind])]) ./ flux.t[ind]

    getfield(xyz_disp,use_mag )[ind] += ΔB_dot
    getfield(xyz_disp,:mag_1_c)[ind] += Δmap_val

    return (xyz_disp)
end # function create_informed_xyz
