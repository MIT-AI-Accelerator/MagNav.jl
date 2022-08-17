"""
    get_XYZ0(xyz_file::String,
             traj_field::Symbol = :traj,
             ins_field::Symbol  = :ins_data;
             flight             = 1,
             line               = 1,
             dt                 = 0.1,
             silent::Bool       = false)

Get the minimum dataset required for MagNav from saved HDF5 or MAT file. 
Not all fields within the `XYZ0` flight data struct are required. The minimum 
data required in the HDF5 or MAT file includes: 
- `lat`, `lon`, `alt` (position)
- `mag_1_uc` OR `mag_1_c` (scalar magnetometer measurements)

If an HDF5 file is provided, the possible fields in the file are: 

|**Field**|**Type**|**Description**
|:--|:--|:--
|`dt`      |scalar | measurement time step [s]
|`tt`      |vector | time [s]
|`lat`     |vector | latitude  [deg]
|`lon`     |vector | longitude [deg]
|`alt`     |vector | altitude  [m]
|`vn`      |vector | north velocity [m/s]
|`ve`      |vector | east  velocity [m/s]
|`vd`      |vector | down  velocity [m/s]
|`fn`      |vector | north specific force [m/s]
|`fe`      |vector | east  specific force [m/s]
|`fd`      |vector | down  specific force [m/s]
|`Cnb`     |3x3xN  | direction cosine matrix (body to navigation) [-]
|`roll`    |vector | roll [deg]
|`pitch`   |vector | pitch [deg]
|`yaw`     |vector | yaw [deg]
|`flux_a_x`|vector | x-direction magnetic field [nT]
|`flux_a_y`|vector | y-direction magnetic field [nT]
|`flux_a_z`|vector | z-direction magnetic field [nT]
|`flux_a_t`|vector | total magnetic field [nT]
|`mag_1_uc`|vector | Mag 1 uncompensated (corrupted) scalar magnetometer measurements [nT]
|`mag_1_c` |vector | Mag 1 compensated (clean) scalar magnetometer measurements [nT]
|`flight`  |vector | flight number(s)
|`line`    |vector | line number(s), i.e. segments within `flight`
|`ins_dt`  |scalar | INS measurement time step [s]
|`ins_tt`  |vector | INS time [s]
|`ins_lat` |vector | INS latitude  [deg]
|`ins_lon` |vector | INS longitude [deg]
|`ins_alt` |vector | INS altitude  [m]
|`ins_vn`  |vector | INS north velocity [m/s]
|`ins_ve`  |vector | INS east  velocity [m/s]
|`ins_vd`  |vector | INS down  velocity [m/s]
|`ins_fn`  |vector | INS north specific force [m/s]
|`ins_fe`  |vector | INS east  specific force [m/s]
|`ins_fd`  |vector | INS down  specific force [m/s]
|`ins_Cnb` |3x3xN  | INS direction cosine matrix (body to navigation) [-]
|`ins_roll`|vector | INS roll [deg]
|`ins_pitch`|vector| INS pitch [deg]
|`ins_yaw` |vector | INS yaw [deg]
|`ins_P`   |17x17xN| INS covariance matrix, only relevant for simulated data, otherwise zeros [-]

If a MAT file is provided, the above fields may also be provided, but the 
non-INS fields should be within a `traj_field` struct and the INS fields should 
be within an `ins_field` struct and without `ins_` prefixes. This is the 
standard way the MATLAB-companion produces data.

**Arguments:**
- `xyz_file`:   path/name of HDF5 or MAT file containing flight data
- `traj_field`: (optional) trajectory struct field within MAT file to use, not relevant for HDF5 file
- `ins_field`:  (optional) INS struct field within MAT file to use, `:none` if unavailable, not relevant for HDF5 file
- `flight`:     (optional) flight number, only used if not in file
- `line`:       (optional) line number, i.e. segment within `flight`, only used if not in file
- `dt`:         (optional) measurement time step [s], only used if not in file
- `silent`:     (optional) if true, no print outs

**Returns:**
- `xyz`: `XYZ0` flight data struct
"""
function get_XYZ0(xyz_file::String,
                  traj_field::Symbol = :traj,
                  ins_field::Symbol  = :ins_data;
                  flight             = 1,
                  line               = 1,
                  dt                 = 0.1,
                  silent::Bool       = false)

    any(occursin.([".h5",".mat"],xyz_file)) || error("$xyz_file flight data file is invalid")

    traj = get_traj(xyz_file,traj_field;dt=dt,silent=silent)

    if occursin(".h5",xyz_file) # get data from HDF5 file

        xyz_data = h5open(xyz_file,"r") # read-only

        if any(isnan.(MagNav.read_check(xyz_data,:ins_lat,1,true))) | 
           any(isnan.(MagNav.read_check(xyz_data,:ins_lon,1,true))) |
           any(isnan.(MagNav.read_check(xyz_data,:ins_alt,1,true)))
            ins = create_ins(traj)
        else
            ins = get_ins(xyz_file,ins_field;dt=traj.dt,silent=silent)
        end

        # these fields might not be included
        if !any(isnan.(read_check(xyz_data,:flight,1,true)))
            flights = read_check(xyz_data ,:flight,1,true)
        end
        if !any(isnan.(read_check(xyz_data,:line,1,true)))
            lines = read_check(xyz_data ,:line,1,true)
        end
        mag_1_uc   = read_check(xyz_data,:mag_1_uc,1,true)
        mag_1_c    = read_check(xyz_data,:mag_1_c ,1,true)
        flux_a_x   = read_check(xyz_data,:flux_a_x,1,true)
        flux_a_y   = read_check(xyz_data,:flux_a_y,1,true)
        flux_a_z   = read_check(xyz_data,:flux_a_z,1,true)
        flux_a_t   = read_check(xyz_data,:flux_a_t,1,true)

        close(xyz_data)

    elseif occursin(".mat",xyz_file) # get data from MAT file

        if ins_field == :none
            ins = create_ins(traj)
        else
            ins = get_ins(xyz_file,ins_field;dt=traj.dt,silent=silent)
        end

        xyz_data = matopen(xyz_file,"r") do file
            read(file,string(traj_field))
        end

        # these fields might not be included
        haskey(xyz_data,"flight") && (flight = xyz_data["flight"])
        haskey(xyz_data,"line"  ) && (line   = xyz_data["line"])
        mag_1_uc = haskey(xyz_data,"mag_1_uc") ? xyz_data["mag_1_uc"] : NaN
        mag_1_c  = haskey(xyz_data,"mag_1_c" ) ? xyz_data["mag_1_c" ] : NaN
        flux_a_x = haskey(xyz_data,"flux_a_x") ? xyz_data["flux_a_x"] : NaN
        flux_a_y = haskey(xyz_data,"flux_a_y") ? xyz_data["flux_a_y"] : NaN
        flux_a_z = haskey(xyz_data,"flux_a_z") ? xyz_data["flux_a_z"] : NaN
        flux_a_t = haskey(xyz_data,"flux_a_t") ? xyz_data["flux_a_t"] : NaN
    end

    # ensure vectors
    mag_1_uc = length(mag_1_uc) > 1 ? vec(mag_1_uc) : mag_1_uc[1]
    mag_1_c  = length(mag_1_c ) > 1 ? vec(mag_1_c ) : mag_1_c[1]
    flux_a_x = length(flux_a_x) > 1 ? vec(flux_a_x) : flux_a_x[1]
    flux_a_y = length(flux_a_y) > 1 ? vec(flux_a_y) : flux_a_y[1]
    flux_a_z = length(flux_a_z) > 1 ? vec(flux_a_z) : flux_a_z[1]
    flux_a_t = length(flux_a_t) > 1 ? vec(flux_a_t) : flux_a_t[1]

    # if needed, create flux_a
    if any(isnan.([flux_a_x;flux_a_y;flux_a_z;]))
        silent || @info("creating compensated vector magnetometer data")
        flux_a = create_flux(traj)
    else
        if any(isnan.(flux_a_t))
            flux_a_t = sqrt.(flux_a_x.^2+flux_a_y.^2+flux_a_z.^2)
        end
        flux_a = MagV(flux_a_x,flux_a_y,flux_a_z,flux_a_t)
    end

    # if needed, create mag_1_uc
    if any(isnan.(mag_1_uc))
        silent || @info("creating uncompensated scalar magnetometer data")
        (mag_1_uc,_) = corrupt_mag(mag_1_c,flux_a;dt=dt)
    end

    # if needed, create mag_1_c
    if any(isnan.(mag_1_c))
        silent || @info("creating compensated scalar magnetometer data")
        A       = create_TL_A(flux_a)
        TL_coef = create_TL_coef(flux_a,mag_1_uc)
        mag_1_c = mag_1_uc - detrend(A*TL_coef;mean_only=true)
    end

    flights = length(flight) == 1 ? flight*one.(traj.lat) : flight
    lines   = length(line)   == 1 ? line  *one.(traj.lat) : line

    return XYZ0(traj, ins, flux_a, flights, lines, mag_1_c, mag_1_uc)
end # function get_XYZ0

"""
    get_traj(traj_file::String, field::Symbol=:traj; dt=0.1, silent::Bool=false)

Get trajectory data from saved HDF5 or MAT file.

**Arguments:**
- `traj_file`: path/name of HDF5 or MAT file containing trajectory data
- `field`:     (optional) struct field within MAT file to use, not relevant for HDF5 file
- `dt`:        (optional) measurement time step [s], only used if not in file
- `silent`:    (optional) if true, no print outs

**Returns:**
- `traj`: `Traj` trajectory struct
"""
function get_traj(traj_file::String, field::Symbol=:traj; dt=0.1, silent::Bool=false)

    silent || @info("reading in data: $traj_file")

    if occursin(".h5",traj_file) # get data from HDF5 file

        traj_data = h5open(traj_file,"r") # read-only

        # these fields are absolutely expected
        lat   = read_check(traj_data,:lat,1,true)
        lon   = read_check(traj_data,:lon,1,true)
        alt   = read_check(traj_data,:alt,1,true)

        # these fields might not be included (especially dt vs tt & Cnb vs RPY)
        if !isnan(read_check(traj_data,:dt,1,true)[1])
            dt = read_check(traj_data ,:dt,1,true)[1]
        end
        tt    = read_check(traj_data,:tt   ,1,true)
        vn    = read_check(traj_data,:vn   ,1,true)
        ve    = read_check(traj_data,:ve   ,1,true)
        vd    = read_check(traj_data,:vd   ,1,true)
        fn    = read_check(traj_data,:fn   ,1,true)
        fe    = read_check(traj_data,:fe   ,1,true)
        fd    = read_check(traj_data,:fd   ,1,true)
        Cnb   = read_check(traj_data,:Cnb  ,1,true)
        roll  = read_check(traj_data,:roll ,1,true)
        pitch = read_check(traj_data,:pitch,1,true)
        yaw   = read_check(traj_data,:yaw  ,1,true)

        close(traj_data)

    elseif occursin(".mat",traj_file) # get data from MAT file

        traj_data = matopen(traj_file,"r") do file
            read(file,string(field))
        end

        # these fields are absolutely expected
        lat   = traj_data["lat"]
        lon   = traj_data["lon"]
        alt   = traj_data["alt"]

        # these fields might not be included (especially dt vs tt & Cnb vs RPY)
        dt    = haskey(traj_data,"dt"   ) ? traj_data["dt"][1] : dt
        tt    = haskey(traj_data,"tt"   ) ? traj_data["tt"   ] : NaN
        vn    = haskey(traj_data,"vn"   ) ? traj_data["vn"   ] : NaN
        ve    = haskey(traj_data,"ve"   ) ? traj_data["ve"   ] : NaN
        vd    = haskey(traj_data,"vd"   ) ? traj_data["vd"   ] : NaN
        fn    = haskey(traj_data,"fn"   ) ? traj_data["fn"   ] : NaN
        fe    = haskey(traj_data,"fe"   ) ? traj_data["fe"   ] : NaN
        fd    = haskey(traj_data,"fd"   ) ? traj_data["fd"   ] : NaN
        Cnb   = haskey(traj_data,"Cnb"  ) ? traj_data["Cnb"  ] : NaN
        roll  = haskey(traj_data,"roll" ) ? traj_data["roll" ] : NaN
        pitch = haskey(traj_data,"pitch") ? traj_data["pitch"] : NaN
        yaw   = haskey(traj_data,"yaw"  ) ? traj_data["yaw"  ] : NaN

    else
        error("$traj_file trajectory file is invalid")
    end

    # ensure vectors
    lat   = length(lat  ) > 1 ? vec(lat  ) : lat[1]
    lon   = length(lon  ) > 1 ? vec(lon  ) : lon[1]
    alt   = length(alt  ) > 1 ? vec(alt  ) : alt[1]
    tt    = length(tt   ) > 1 ? vec(tt   ) : tt[1]
    vn    = length(vn   ) > 1 ? vec(vn   ) : vn[1]
    ve    = length(ve   ) > 1 ? vec(ve   ) : ve[1]
    vd    = length(vd   ) > 1 ? vec(vd   ) : vd[1]
    fn    = length(fn   ) > 1 ? vec(fn   ) : fn[1]
    fe    = length(fe   ) > 1 ? vec(fe   ) : fe[1]
    fd    = length(fd   ) > 1 ? vec(fd   ) : fd[1]
    roll  = length(roll ) > 1 ? vec(roll ) : roll[1]
    pitch = length(pitch) > 1 ? vec(pitch) : pitch[1]
    yaw   = length(yaw  ) > 1 ? vec(yaw  ) : yaw[1]

    # convert deg to rad
    lat   = deg2rad.(lat)
    lon   = deg2rad.(lon)
    roll  = deg2rad.(roll)
    pitch = deg2rad.(pitch)
    yaw   = deg2rad.(yaw)

    N = length(lat)

    # if needed, create tt
    if any(isnan.(tt))
        silent || @info("creating time data")
        tt = [LinRange(0:dt:dt*(N-1));]
    else
        dt = tt[2] - tt[1]
    end

    # if needed, create vn, ve, vd
    if any(isnan.([vn;ve;vd]))
        silent || @info("creating velocity data")
        lla2utm = UTMZfromLLA(WGS84)
        llas    = LLA.(rad2deg.(lat),rad2deg.(lon),alt)
        vn      =  fdm([lla2utm(lla).y for lla in llas]) / dt
        ve      =  fdm([lla2utm(lla).x for lla in llas]) / dt
        vd      = -fdm([lla2utm(lla).z for lla in llas]) / dt
    end

    # if needed, create fn, fe, fd
    if any(isnan.([fn;fe;fd]))
        silent || @info("creating specific force data")
        fn = fdm(vn) / dt
        fe = fdm(ve) / dt
        fd = fdm(vd) / dt .- g_earth
    end

    # if needed, create Cnb
    if any(isnan.(Cnb))
        if any(isnan.([roll;pitch;yaw]))
            silent || @info("creating direction cosine matrix data")
            Cnb = create_dcm(vn,ve,dt,:body2nav)
        else # use RPY to get Cnb
            Cnb = euler2dcm(roll,pitch,yaw,:body2nav)
        end
    end

    return Traj(N, dt, tt, lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb)
end # function get_traj

"""
    get_ins(ins_file::String, field::Symbol=:ins_data; dt=0.1, silent::Bool=false)

Get inertial navigation system data from saved HDF5 or MAT file.

**Arguments:**
- `ins_file`: path/name of HDF5 or MAT file containing INS data
- `field`:    (optional) struct field within MAT file to use, not relevant for HDF5 file
- `dt`:       (optional) measurement time step [s], only used if not in file
- `silent`:   (optional) if true, no print outs

**Returns:**
- `ins`: `INS` inertial navigation system struct
"""
function get_ins(ins_file::String, field::Symbol=:ins_data; dt=0.1, silent::Bool=false)

    silent || @info("reading in data: $ins_file")

    if occursin(".h5",ins_file) # get data from HDF5 file

        ins_data = h5open(ins_file,"r") # read-only

        # these fields are absolutely expected
        lat   = read_check(ins_data,:ins_lat,1,true)
        lon   = read_check(ins_data,:ins_lon,1,true)
        alt   = read_check(ins_data,:ins_alt,1,true)

        # these fields might not be included (especially dt vs tt & Cnb vs RPY)
        if !isnan(read_check(ins_data,:ins_dt,1,true)[1])
            dt = read_check(ins_data,:ins_dt,1,true)[1]
        end
        tt    = read_check(ins_data,:ins_tt   ,1,true)
        vn    = read_check(ins_data,:ins_vn   ,1,true)
        ve    = read_check(ins_data,:ins_ve   ,1,true)
        vd    = read_check(ins_data,:ins_vd   ,1,true)
        fn    = read_check(ins_data,:ins_fn   ,1,true)
        fe    = read_check(ins_data,:ins_fe   ,1,true)
        fd    = read_check(ins_data,:ins_fd   ,1,true)
        Cnb   = read_check(ins_data,:ins_Cnb  ,1,true)
        roll  = read_check(ins_data,:ins_roll ,1,true)
        pitch = read_check(ins_data,:ins_pitch,1,true)
        yaw   = read_check(ins_data,:ins_yaw  ,1,true)
        P     = read_check(ins_data,:ins_P    ,1,true)

        close(ins_data)

    elseif occursin(".mat",ins_file) # get data from MAT file

        ins_data = matopen(ins_file,"r") do file
            read(file,string(field))
        end

        # these fields are absolutely expected
        lat   = ins_data["lat"]
        lon   = ins_data["lon"]
        alt   = ins_data["alt"]

        # these fields might not be included (especially dt vs tt & Cnb vs RPY)
        dt    = haskey(ins_data,"dt"   ) ? ins_data["dt"][1] : dt
        tt    = haskey(ins_data,"tt"   ) ? ins_data["tt"   ] : NaN
        vn    = haskey(ins_data,"vn"   ) ? ins_data["vn"   ] : NaN
        ve    = haskey(ins_data,"ve"   ) ? ins_data["ve"   ] : NaN
        vd    = haskey(ins_data,"vd"   ) ? ins_data["vd"   ] : NaN
        fn    = haskey(ins_data,"fn"   ) ? ins_data["fn"   ] : NaN
        fe    = haskey(ins_data,"fe"   ) ? ins_data["fe"   ] : NaN
        fd    = haskey(ins_data,"fd"   ) ? ins_data["fd"   ] : NaN
        Cnb   = haskey(ins_data,"Cnb"  ) ? ins_data["Cnb"  ] : NaN
        roll  = haskey(ins_data,"roll" ) ? ins_data["roll" ] : NaN
        pitch = haskey(ins_data,"pitch") ? ins_data["pitch"] : NaN
        yaw   = haskey(ins_data,"yaw"  ) ? ins_data["yaw"  ] : NaN
        P     = haskey(ins_data,"P"    ) ? ins_data["P"    ] : NaN

    else
        error("$ins_file INS file is invalid")
    end

    # ensure vectors
    lat   = length(lat  ) > 1 ? vec(lat  ) : lat[1]
    lon   = length(lon  ) > 1 ? vec(lon  ) : lon[1]
    alt   = length(alt  ) > 1 ? vec(alt  ) : alt[1]
    tt    = length(tt   ) > 1 ? vec(tt   ) : tt[1]
    vn    = length(vn   ) > 1 ? vec(vn   ) : vn[1]
    ve    = length(ve   ) > 1 ? vec(ve   ) : ve[1]
    vd    = length(vd   ) > 1 ? vec(vd   ) : vd[1]
    fn    = length(fn   ) > 1 ? vec(fn   ) : fn[1]
    fe    = length(fe   ) > 1 ? vec(fe   ) : fe[1]
    fd    = length(fd   ) > 1 ? vec(fd   ) : fd[1]
    roll  = length(roll ) > 1 ? vec(roll ) : roll[1]
    pitch = length(pitch) > 1 ? vec(pitch) : pitch[1]
    yaw   = length(yaw  ) > 1 ? vec(yaw  ) : yaw[1]

    # convert deg to rad
    lat   = deg2rad.(lat)
    lon   = deg2rad.(lon)
    roll  = deg2rad.(roll)
    pitch = deg2rad.(pitch)
    yaw   = deg2rad.(yaw)

    N = length(lat)

    # if needed, create tt
    if any(isnan.(tt))
        silent || @info("creating INS time data")
        tt = [LinRange(0:dt:dt*(N-1));]
    else
        dt = tt[2] - tt[1]
    end

    # if needed, create vn, ve, vd
    if any(isnan.([vn;ve;vd]))
        silent || @info("creating INS velocity data")
        lla2utm = UTMZfromLLA(WGS84)
        llas    = LLA.(rad2deg.(lat),rad2deg.(lon),alt)
        vn      =  fdm([lla2utm(lla).y for lla in llas]) / dt
        ve      =  fdm([lla2utm(lla).x for lla in llas]) / dt
        vd      = -fdm([lla2utm(lla).z for lla in llas]) / dt
    end

    # if needed, create fn, fe, fd
    if any(isnan.([fn;fe;fd]))
        silent || @info("creating INS specific force data")
        fn = fdm(vn) / dt
        fe = fdm(ve) / dt
        fd = fdm(vd) / dt .- g_earth
    end

    # if needed, create Cnb
    if any(isnan.(Cnb))
        if any(isnan.([roll;pitch;yaw]))
            silent || @info("creating INS direction cosine matrix data")
            Cnb = create_dcm(vn,ve,dt,:body2nav)
        else # use RPY to get Cnb
            Cnb = euler2dcm(roll,pitch,yaw,:body2nav)
        end
    end

    # if needed, create P
    if any(isnan.(P))
        silent || @info("creating INS covariance matrix data (zeros)")
        P = zeros(1,1,N) # unknown
    end

    return INS(N, dt, tt, lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb, P)
end # function get_ins

"""
    get_ins(xyz::XYZ, ind=trues(xyz.traj.N);
            N_zero_ll::Int=0, t_zero_ll=0, err=0.0)

Get inertial navigation system data at specific indicies.

**Arguments:**
- `xyz`:       `XYZ` flight data struct
- `ind`:       (optional) selected data indices
- `N_zero_ll`: (optional) number of samples (instances) to zero INS lat/lon to truth (`xyz.traj`)
- `t_zero_ll`: (optional) length of time to zero INS lat/lon to truth (`xyz.traj`), will overwrite `N_zero_ll`
- `err`:       (optional) initial INS latitude and longitude error [m]

**Returns:**
- `ins`: `INS` inertial navigation system struct at specific indicies
"""
function get_ins(xyz::XYZ, ind=trues(xyz.ins.N);
                 N_zero_ll::Int=0, t_zero_ll=0, err=0.0)
    N_zero_ll = t_zero_ll > 0 ? round(Int,t_zero_ll/xyz.ins.dt) : N_zero_ll
    lat = xyz.traj.lat[ind][1:N_zero_ll]
    lon = xyz.traj.lon[ind][1:N_zero_ll]
    xyz.ins(ind;N_zero_ll=N_zero_ll,err=err,lat=lat,lon=lon)
end # function get_ins

function (ins::INS)(ind=trues(ins.N);
                    N_zero_ll::Int=0, err=0.0,
                    lat=zero(ins.lat[ind]),
                    lon=zero(ins.lon[ind]))

    ind = findall((1:ins.N) .∈ ((1:ins.N)[ind],))

    if N_zero_ll > 0
        (ins_lat,ins_lon) = zero_ins_ll(ins.lat[ind],ins.lon[ind],err,lat,lon)
    else
        (ins_lat,ins_lon) = (ins.lat[ind],ins.lon[ind])
    end

    return INS(length(ind) ,  ins.dt      , ins.tt[ind]   ,
               ins_lat     ,  ins_lon     , ins.alt[ind]  ,
               ins.vn[ind] ,  ins.ve[ind] , ins.vd[ind]   ,
               ins.fn[ind] ,  ins.fe[ind] , ins.fd[ind]   ,
               ins.Cnb[:,:,ind], ins.P[:,:,ind])
end # function (ins::INS)

"""
    zero_ins_ll(ins_lat, ins_lon, err=0.0, lat=ins_lat[1:1], lon=ins_lon[1:1])

Zero INS latitude and longitude starting position, with optional error.

**Arguments:**
- `ins_lat`: INS latitude  vector [rad]
- `ins_lon`: INS longitude vector [rad]
- `err`:     (optional) initial INS latitude and longitude error [m]
- `lat`:     (optional) best-guess (truth) initial latitude  [rad]
- `lon`:     (optional) best-guess (truth) initial longitude [rad]

**Returns:**
- `ins_lat`: zeroed INS latitude  vector [rad]
- `ins_lon`: zeroed INS longitude vector [rad]
"""
function zero_ins_ll(ins_lat, ins_lon, err=0.0, lat=ins_lat[1:1], lon=ins_lon[1:1])

    # check that INS vectors are the same length and get length
    N_lat = length(ins_lat)
    N_lon = length(ins_lon)
    N_lat == N_lon ? (N=N_lat) : error("N_ins_lat ≂̸ N_ins_lon")

    # check that truth vectors (or scalar) are the same length and get length
    N_zero_lat = length(lat)
    N_zero_lon = length(lon)
    N_zero_lat == N_zero_lon ? (N0=N_zero_lat) : error("N_lat ≂̸ N_lon")

    # avoid modifying original data (possibly in xyz struct)
    ins_lat = deepcopy(ins_lat)
    ins_lon = deepcopy(ins_lon)

    # correct 1:N0
    δlat = ins_lat[1:N0] - lat
    δlon = ins_lon[1:N0] - lon
    ins_lat[1:N0] .-= (δlat .- sign.(δlat) .* dn2dlat.(err,lat))
    ins_lon[1:N0] .-= (δlon .- sign.(δlon) .* de2dlon.(err,lat))

    # correct N0+1:end
    if N > N0
        δlat_end = δlat[end] .- sign.(δlat[end]) .* dn2dlat.(err,lat[end])
        δlon_end = δlon[end] .- sign.(δlon[end]) .* de2dlon.(err,lat[end])
        ins_lat[N0+1:end] .-= δlat_end
        ins_lon[N0+1:end] .-= δlon_end
    end

    return (ins_lat, ins_lon)
end # function zero_ins_ll

"""
    get_traj(xyz::XYZ, ind=trues(xyz.traj.N))

Get trajectory data at specific indicies.

**Arguments:**
- `xyz`: `XYZ` flight data struct
- `ind`: (optional) selected data indices

**Returns:**
- `traj`: `Traj` trajectory struct at specific indicies
"""
function get_traj(xyz::XYZ, ind=trues(xyz.traj.N))
    xyz.traj(ind)
end # function get_traj

function (traj::Traj)(ind=trues(traj.N))

    ind = findall((1:traj.N) .∈ ((1:traj.N)[ind],))

    return Traj(length(ind)   , traj.dt        , traj.tt[ind]   ,
                traj.lat[ind] , traj.lon[ind]  , traj.alt[ind]  ,
                traj.vn[ind]  , traj.ve[ind]   , traj.vd[ind]   ,
                traj.fn[ind]  , traj.fe[ind]   , traj.fd[ind]   ,
                traj.Cnb[:,:,ind])
end # function (traj::Traj)

function (flux::MagV)(ind=trues(length(flux.x)))
    return MagV(flux.x[ind], flux.y[ind], flux.z[ind], flux.t[ind])
end # function (flux::MagV)
