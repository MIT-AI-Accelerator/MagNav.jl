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
|`flux_a_x`|vector | Flux A x-direction magnetic field [nT]
|`flux_a_y`|vector | Flux A y-direction magnetic field [nT]
|`flux_a_z`|vector | Flux A z-direction magnetic field [nT]
|`flux_a_t`|vector | Flux A total magnetic field [nT]
|`flight`  |vector | flight number(s)
|`line`    |vector | line number(s), i.e., segments within `flight`
|`mag_1_c` |vector | Mag 1 compensated (clean) scalar magnetometer measurements [nT]
|`mag_1_uc`|vector | Mag 1 uncompensated (corrupted) scalar magnetometer measurements [nT]

If a MAT file is provided, the above fields may also be provided, but the
non-INS fields should be within the specified `traj_field` MAT struct and the
INS fields should be within the specified `ins_field` MAT struct and without
`ins_` prefixes. This is the standard way the MATLAB-companion outputs data.

**Arguments:**
- `xyz_file`:   path/name of flight data HDF5 or MAT file (`.h5` or `.mat` extension required)
- `traj_field`: (optional) trajectory struct field within MAT file to use, not relevant for HDF5 file
- `ins_field`:  (optional) INS struct field within MAT file to use, `:none` if unavailable, not relevant for HDF5 file
- `flight`:     (optional) flight number, only used if not in file
- `line`:       (optional) line number, i.e., segment within `flight`, only used if not in file
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

    @assert any(occursin.([".h5",".mat"],xyz_file)) "$xyz_file flight data file must have .h5 or .mat extension"

    traj = get_traj(xyz_file,traj_field;dt=dt,silent=silent)

    if occursin(".h5",xyz_file) # get data from HDF5 file

        xyz_data = h5open(xyz_file,"r") # read-only

        # if needed, create INS struct
        if any(isnan.(read_check(xyz_data,:ins_lat,1,true))) | 
           any(isnan.(read_check(xyz_data,:ins_lon,1,true))) |
           any(isnan.(read_check(xyz_data,:ins_alt,1,true)))
            ins = create_ins(traj)
        else
            ins = get_ins(xyz_file,ins_field;dt=traj.dt,silent=silent)
        end

        # these fields might not be included
        flight_  = read_check(xyz_data,:flight,1,true)
        line_    = read_check(xyz_data,:line  ,1,true)
        flight   = any(isnan.(flight_)) ? flight : flight_
        line     = any(isnan.(line_))   ? line   : line_
        mag_1_uc = read_check(xyz_data,:mag_1_uc,1,true)
        mag_1_c  = read_check(xyz_data,:mag_1_c ,1,true)

        close(xyz_data)

    elseif occursin(".mat",xyz_file) # get data from MAT file

        # if needed, create INS struct
        if ins_field == :none
            ins = create_ins(traj)
        else
            ins = get_ins(xyz_file,ins_field;dt=traj.dt,silent=silent)
        end

        xyz_data = matopen(xyz_file,"r") do file
            read(file,string(traj_field))
        end

        # these fields might not be included
        flight   = haskey(xyz_data,"flight"  ) ? xyz_data["flight"]   : flight
        line     = haskey(xyz_data,"line"    ) ? xyz_data["line"]     : line
        mag_1_uc = haskey(xyz_data,"mag_1_uc") ? xyz_data["mag_1_uc"] : NaN
        mag_1_c  = haskey(xyz_data,"mag_1_c" ) ? xyz_data["mag_1_c" ] : NaN

    end

    # ensure vectors
    flight   = length(flight  ) > 1 ? vec(flight  ) : one.(traj.lat)*flight[1]
    line     = length(line    ) > 1 ? vec(line    ) : one.(traj.lat)*line[1]
    mag_1_uc = length(mag_1_uc) > 1 ? vec(mag_1_uc) : one.(traj.lat)*mag_1_uc[1]
    mag_1_c  = length(mag_1_c ) > 1 ? vec(mag_1_c ) : one.(traj.lat)*mag_1_c[1]

    # if needed, create flux_a
    flux_a = get_flux(xyz_file,:flux_a,traj_field)
    if any(isnan.([flux_a.x;flux_a.y;flux_a.z;]))
        silent || @info("creating compensated vector magnetometer data")
        flux_a = create_flux(traj)
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

    return XYZ0(traj, ins, flux_a, flight, line, mag_1_c, mag_1_uc)
end # function get_XYZ0

"""
    get_XYZ1(xyz_file::String,
             traj_field::Symbol = :traj,
             ins_field::Symbol  = :ins_data;
             flight             = 1,
             line               = 1,
             dt                 = 0.1,
             silent::Bool       = false)

Get the minimum dataset required for MagNav from saved HDF5 or MAT file.
Not all fields within the `XYZ1` flight data struct are required. The minimum
data required in the HDF5 or MAT file includes:
- `lat`, `lon`, `alt` (position)
- `mag_1_uc` OR `mag_1_c` (scalar magnetometer measurements)

All other fields will be computed/simulated, except `flux_b`, `year`, `doy`,
`diurnal`, `igrf`, `mag_2_c`, `mag_3_c`, `mag_2_uc`, `mag_3_uc`, `aux_1`,
`aux_2`, and `aux_3`.

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
|`flux_a_x`|vector | Flux A x-direction magnetic field [nT]
|`flux_a_y`|vector | Flux A y-direction magnetic field [nT]
|`flux_a_z`|vector | Flux A z-direction magnetic field [nT]
|`flux_a_t`|vector | Flux A total magnetic field [nT]
|`flux_b_x`|vector | Flux B x-direction magnetic field [nT]
|`flux_b_y`|vector | Flux B y-direction magnetic field [nT]
|`flux_b_z`|vector | Flux B z-direction magnetic field [nT]
|`flux_b_t`|vector | Flux B total magnetic field [nT]
|`flight`  |vector | flight number(s)
|`line`    |vector | line number(s), i.e., segments within `flight`
|`year`    |vector | year
|`doy`     |vector | day of year
|`diurnal` |vector | measured diurnal, i.e., temporal variations or space weather effects [nT]
|`igrf`    |vector | International Geomagnetic Reference Field (IGRF), i.e., core field [nT]
|`mag_1_c` |vector | Mag 1 compensated (clean) scalar magnetometer measurements [nT]
|`mag_2_c` |vector | Mag 2 compensated (clean) scalar magnetometer measurements [nT]
|`mag_3_c` |vector | Mag 3 compensated (clean) scalar magnetometer measurements [nT]
|`mag_1_uc`|vector | Mag 1 uncompensated (corrupted) scalar magnetometer measurements [nT]
|`mag_2_uc`|vector | Mag 2 uncompensated (corrupted) scalar magnetometer measurements [nT]
|`mag_3_uc`|vector | Mag 3 uncompensated (corrupted) scalar magnetometer measurements [nT]
|`aux_1`   |vector | flexible-use auxiliary data 1
|`aux_2`   |vector | flexible-use auxiliary data 2
|`aux_3`   |vector | flexible-use auxiliary data 3

If a MAT file is provided, the above fields may also be provided, but the
non-INS fields should be within the specified `traj_field` MAT struct and the
INS fields should be within the specified `ins_field` MAT struct and without
`ins_` prefixes. This is the standard way the MATLAB-companion outputs data.

**Arguments:**
- `xyz_file`:   path/name of flight data HDF5 or MAT file (`.h5` or `.mat` extension required)
- `traj_field`: (optional) trajectory struct field within MAT file to use, not relevant for HDF5 file
- `ins_field`:  (optional) INS struct field within MAT file to use, `:none` if unavailable, not relevant for HDF5 file
- `flight`:     (optional) flight number, only used if not in file
- `line`:       (optional) line number, i.e., segment within `flight`, only used if not in file
- `dt`:         (optional) measurement time step [s], only used if not in file
- `silent`:     (optional) if true, no print outs

**Returns:**
- `xyz`: `XYZ1` flight data struct
"""
function get_XYZ1(xyz_file::String,
                  traj_field::Symbol = :traj,
                  ins_field::Symbol  = :ins_data;
                  flight             = 1,
                  line               = 1,
                  dt                 = 0.1,
                  silent::Bool       = false)

    @assert any(occursin.([".h5",".mat"],xyz_file)) "$xyz_file flight data file must have .h5 or .mat extension"

    xyz = get_XYZ0(xyz_file,traj_field,ins_field;
                   flight = flight,
                   line   = line,
                   dt     = dt,
                   silent = silent)

    # these fields can be extracted using XYZ0 functionality
    traj     = xyz.traj
    ins      = xyz.ins
    flux_a   = xyz.flux_a
    flight   = xyz.flight
    line     = xyz.line
    mag_1_c  = xyz.mag_1_c
    mag_1_uc = xyz.mag_1_uc

    flux_b = get_flux(xyz_file,:flux_b,traj_field)

    if occursin(".h5",xyz_file) # get data from HDF5 file

        xyz_data = h5open(xyz_file,"r") # read-only

        # these fields might not be included
        year     = read_check(xyz_data,:year,1,true)
        doy      = read_check(xyz_data,:doy ,1,true)
        diurnal  = read_check(xyz_data,:diurnal,1,true)
        igrf     = read_check(xyz_data,:igrf ,1,true)
        mag_2_c  = read_check(xyz_data,:mag_2_c,1,true)
        mag_3_c  = read_check(xyz_data,:mag_3_c ,1,true)
        mag_2_uc = read_check(xyz_data,:mag_2_uc,1,true)
        mag_3_uc = read_check(xyz_data,:mag_3_uc ,1,true)
        aux_1    = read_check(xyz_data,:aux_1,1,true)
        aux_2    = read_check(xyz_data,:aux_2,1,true)
        aux_3    = read_check(xyz_data,:aux_3 ,1,true)

        close(xyz_data)

    elseif occursin(".mat",xyz_file) # get data from MAT file

        xyz_data = matopen(xyz_file,"r") do file
            read(file,string(traj_field))
        end

        # these fields might not be included
        year     = haskey(xyz_data,"year"    ) ? xyz_data["year"    ] : NaN
        doy      = haskey(xyz_data,"doy"     ) ? xyz_data["doy"     ] : NaN
        diurnal  = haskey(xyz_data,"diurnal" ) ? xyz_data["diurnal" ] : NaN
        igrf     = haskey(xyz_data,"igrf"    ) ? xyz_data["igrf"    ] : NaN
        mag_2_c  = haskey(xyz_data,"mag_2_c" ) ? xyz_data["mag_2_c" ] : NaN
        mag_3_c  = haskey(xyz_data,"mag_3_c" ) ? xyz_data["mag_3_c" ] : NaN
        mag_2_uc = haskey(xyz_data,"mag_2_uc") ? xyz_data["mag_2_uc"] : NaN
        mag_3_uc = haskey(xyz_data,"mag_3_uc") ? xyz_data["mag_3_uc"] : NaN
        aux_1    = haskey(xyz_data,"aux_1"   ) ? xyz_data["aux_1"   ] : NaN
        aux_2    = haskey(xyz_data,"aux_2"   ) ? xyz_data["aux_2"   ] : NaN
        aux_3    = haskey(xyz_data,"aux_3"   ) ? xyz_data["aux_3"   ] : NaN
    end

    # ensure vectors
    year     = length(year    ) > 1 ? vec(year    ) : one.(traj.lat)*year[1]
    doy      = length(doy     ) > 1 ? vec(doy     ) : one.(traj.lat)*doy[1]
    diurnal  = length(diurnal ) > 1 ? vec(diurnal ) : one.(traj.lat)*diurnal[1]
    igrf     = length(igrf    ) > 1 ? vec(igrf    ) : one.(traj.lat)*igrf[1]
    mag_2_c  = length(mag_2_c ) > 1 ? vec(mag_2_c ) : one.(traj.lat)*mag_2_c[1]
    mag_3_c  = length(mag_3_c ) > 1 ? vec(mag_3_c ) : one.(traj.lat)*mag_3_c[1]
    mag_2_uc = length(mag_2_uc) > 1 ? vec(mag_2_uc) : one.(traj.lat)*mag_2_uc[1]
    mag_3_uc = length(mag_3_uc) > 1 ? vec(mag_3_uc) : one.(traj.lat)*mag_3_uc[1]
    aux_1    = length(aux_1   ) > 1 ? vec(aux_1   ) : one.(traj.lat)*aux_1[1]
    aux_2    = length(aux_2   ) > 1 ? vec(aux_2   ) : one.(traj.lat)*aux_2[1]
    aux_3    = length(aux_3   ) > 1 ? vec(aux_3   ) : one.(traj.lat)*aux_3[1]

    return XYZ1(traj, ins, flux_a, flux_b, flight, line, year, doy,
                diurnal, igrf, mag_1_c, mag_2_c, mag_3_c,
                mag_1_uc, mag_2_uc, mag_3_uc, aux_1, aux_2, aux_3)
end # function get_XYZ1

"""
    get_flux(flux_file::String,
             use_vec::Symbol = :flux_a,
             field::Symbol   = :traj)

Get vector magnetometer data from saved HDF5 or MAT file.

If an HDF5 file is provided, the possible fields in the file are:

|**Field**|**Type**|**Description**
|:--|:--|:--
|`use_vec`*`_x`|vector | x-direction magnetic field [nT]
|`use_vec`*`_y`|vector | y-direction magnetic field [nT]
|`use_vec`*`_z`|vector | z-direction magnetic field [nT]
|`use_vec`*`_t`|vector | total magnetic field [nT], optional

If a MAT file is provided, the above fields may also be provided, but they
should be within the specified `field` MAT struct. This is the standard way the
MATLAB-companion outputs data.

**Arguments:**
- `flux_file`: path/name of vector magnetometer data HDF5 or MAT file (`.h5` or `.mat` extension required)
- `field`:     (optional) struct field within MAT file to use, not relevant for HDF5 file
- `use_vec`:   (optional) vector magnetometer (fluxgate) to use

**Returns:**
- `flux`: `MagV` vector magnetometer measurement struct
"""
function get_flux(flux_file::String,
                  use_vec::Symbol = :flux_a,
                  field::Symbol   = :traj)

    @assert any(occursin.([".h5",".mat"],flux_file)) "$flux_file vector magnetometer data file must have .h5 or .mat extension"

    if occursin(".h5",flux_file) # get data from HDF5 file

        magv_data = h5open(flux_file,"r") # read-only

        x = read_check(magv_data,Symbol(use_vec,"_x"),1,true)
        y = read_check(magv_data,Symbol(use_vec,"_y"),1,true)
        z = read_check(magv_data,Symbol(use_vec,"_z"),1,true)
        t = read_check(magv_data,Symbol(use_vec,"_t"),1,true)

        close(magv_data)

    elseif occursin(".mat",flux_file) # get data from MAT file

        magv_data = matopen(flux_file,"r") do file
            read(file,string(field))
        end

        x = haskey(magv_data,"$(use_vec)_x") ? magv_data["$(use_vec)_x"] : NaN
        y = haskey(magv_data,"$(use_vec)_y") ? magv_data["$(use_vec)_y"] : NaN
        z = haskey(magv_data,"$(use_vec)_z") ? magv_data["$(use_vec)_z"] : NaN
        t = haskey(magv_data,"$(use_vec)_t") ? magv_data["$(use_vec)_t"] : NaN
    end

    # ensure vectors
    l = ones(eltype(x),maximum(length.([x,y,z,t]))) 
    x = length(x) > 1 ? vec(x) : one.(l)*x[1]
    y = length(y) > 1 ? vec(y) : one.(l)*y[1]
    z = length(z) > 1 ? vec(z) : one.(l)*z[1]
    t = length(t) > 1 ? vec(t) : one.(l)*t[1]

    # in case total field not filled, but others are
    any(isnan.(t)) && (t = sqrt.(x.^2 + y.^2 + z.^2))

    return MagV(x, y, z, t)
end # function get_flux

get_MagV = get_magv = get_flux

"""
    get_traj(traj_file::String, field::Symbol=:traj; dt=0.1, silent::Bool=false)

Get trajectory data from saved HDF5 or MAT file. The only required fields are
`lat`, `lon`, and `alt` (position).

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

If a MAT file is provided, the above fields may also be provided, but they
should be within the specified `field` MAT struct. This is the standard way the
MATLAB-companion outputs data.

**Arguments:**
- `traj_file`: path/name of trajectory data HDF5 or MAT file (`.h5` or `.mat` extension required)
- `field`:     (optional) struct field within MAT file to use, not relevant for HDF5 file
- `dt`:        (optional) measurement time step [s], only used if not in file
- `silent`:    (optional) if true, no print outs

**Returns:**
- `traj`: `Traj` trajectory struct
"""
function get_traj(traj_file::String, field::Symbol=:traj; dt=0.1, silent::Bool=false)

    @assert any(occursin.([".h5",".mat"],traj_file)) "$traj_file trajectory data file must have .h5 or .mat extension"

    silent || @info("reading in data: $traj_file")

    if occursin(".h5",traj_file) # get data from HDF5 file

        traj_data = h5open(traj_file,"r") # read-only

        # these fields are absolutely expected
        lat   = read_check(traj_data,:lat,1,true)
        lon   = read_check(traj_data,:lon,1,true)
        alt   = read_check(traj_data,:alt,1,true)

        # these fields might not be included (especially dt vs tt & Cnb vs RPY)
        dt_   = read_check(traj_data,:dt,1,true)[1]
        dt    = isnan(dt_) ? dt : dt_
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

    end

    # ensure vectors
    lat   = vec(lat) # assume lat always exists as vector
    lon   = length(lon  ) > 1 ? vec(lon  ) : one.(lat)*lon[1]
    alt   = length(alt  ) > 1 ? vec(alt  ) : one.(lat)*alt[1]
    tt    = length(tt   ) > 1 ? vec(tt   ) : one.(lat)*tt[1]
    vn    = length(vn   ) > 1 ? vec(vn   ) : one.(lat)*vn[1]
    ve    = length(ve   ) > 1 ? vec(ve   ) : one.(lat)*ve[1]
    vd    = length(vd   ) > 1 ? vec(vd   ) : one.(lat)*vd[1]
    fn    = length(fn   ) > 1 ? vec(fn   ) : one.(lat)*fn[1]
    fe    = length(fe   ) > 1 ? vec(fe   ) : one.(lat)*fe[1]
    fd    = length(fd   ) > 1 ? vec(fd   ) : one.(lat)*fd[1]
    roll  = length(roll ) > 1 ? vec(roll ) : one.(lat)*roll[1]
    pitch = length(pitch) > 1 ? vec(pitch) : one.(lat)*pitch[1]
    yaw   = length(yaw  ) > 1 ? vec(yaw  ) : one.(lat)*yaw[1]

    # convert deg to rad
    lat   = deg2rad.(lat)
    lon   = deg2rad.(lon)
    roll  = deg2rad.(roll)
    pitch = deg2rad.(pitch)
    yaw   = deg2rad.(yaw)

    N = length(lat)

    # if needed, create tt, otherwise get dt from tt
    if any(isnan.(tt))
        silent || @info("creating time data")
        tt = round.(0:dt:dt*(N-1),digits=9)
    else
        dt = round(tt[2] - tt[1],digits=9)
    end

    # if needed, create vn, ve, vd
    if any(isnan.([vn;ve;vd]))
        silent || @info("creating velocity data")
        (zone_utm,is_north) = utm_zone(mean(rad2deg.(lat)),mean(rad2deg.(lon)))
        lla2utm = UTMfromLLA(zone_utm,is_north,WGS84)
        utms    = lla2utm.(LLA.(rad2deg.(lat),rad2deg.(lon),alt))
        vn      =  fdm([utm.y for utm in utms]) / dt
        ve      =  fdm([utm.x for utm in utms]) / dt
        vd      = -fdm([utm.z for utm in utms]) / dt
    end

    # if needed, create fn, fe, fd
    if any(isnan.([fn;fe;fd]))
        silent || @info("creating specific force data")
        fn = fdm(vn) / dt
        fe = fdm(ve) / dt
        fd = fdm(vd) / dt .- g_earth
    end

    # if needed, create Cnb, otherwise get Cnb from RPY
    if any(isnan.(Cnb))
        if any(isnan.([roll;pitch;yaw]))
            silent || @info("creating direction cosine matrix data")
            Cnb = create_dcm(vn,ve,dt,:body2nav)
        else
            Cnb = euler2dcm(roll,pitch,yaw,:body2nav)
        end
    end

    return Traj(N, dt, tt, lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb)
end # function get_traj

"""
    get_ins(ins_file::String, field::Symbol=:ins_data; dt=0.1, silent::Bool=false)

Get inertial navigation system data from saved HDF5 or MAT file. The only
required fields are `ins_lat`, `ins_lon`, and `ins_alt` (position).

If an HDF5 file is provided, the possible fields in the file are:

|**Field**|**Type**|**Description**
|:--|:--|:--
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

If a MAT file is provided, the above fields may also be provided, but they
should be within the specified `field` MAT struct and without `ins_`
prefixes. This is the standard way the MATLAB-companion outputs data.

**Arguments:**
- `ins_file`: path/name of INS data HDF5 or MAT file (`.h5` or `.mat` extension required)
- `field`:    (optional) struct field within MAT file to use, not relevant for HDF5 file
- `dt`:       (optional) measurement time step [s], only used if not in file
- `silent`:   (optional) if true, no print outs

**Returns:**
- `ins`: `INS` inertial navigation system struct
"""
function get_ins(ins_file::String, field::Symbol=:ins_data; dt=0.1, silent::Bool=false)

    @assert any(occursin.([".h5",".mat"],ins_file)) "$ins_file INS data file must have .h5 or .mat extension"

    silent || @info("reading in data: $ins_file")

    if occursin(".h5",ins_file) # get data from HDF5 file

        ins_data = h5open(ins_file,"r") # read-only

        # these fields are absolutely expected
        lat   = read_check(ins_data,:ins_lat,1,true)
        lon   = read_check(ins_data,:ins_lon,1,true)
        alt   = read_check(ins_data,:ins_alt,1,true)

        # these fields might not be included (especially dt vs tt & Cnb vs RPY)
        dt_   = read_check(ins_data,:ins_dt,1,true)[1]
        dt    = isnan(dt_) ? dt : dt_
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

    end

    # ensure vectors
    lat   = vec(lat) # assume lat always exists as vector
    lon   = length(lon  ) > 1 ? vec(lon  ) : one.(lat)*lon[1]
    alt   = length(alt  ) > 1 ? vec(alt  ) : one.(lat)*alt[1]
    tt    = length(tt   ) > 1 ? vec(tt   ) : one.(lat)*tt[1]
    vn    = length(vn   ) > 1 ? vec(vn   ) : one.(lat)*vn[1]
    ve    = length(ve   ) > 1 ? vec(ve   ) : one.(lat)*ve[1]
    vd    = length(vd   ) > 1 ? vec(vd   ) : one.(lat)*vd[1]
    fn    = length(fn   ) > 1 ? vec(fn   ) : one.(lat)*fn[1]
    fe    = length(fe   ) > 1 ? vec(fe   ) : one.(lat)*fe[1]
    fd    = length(fd   ) > 1 ? vec(fd   ) : one.(lat)*fd[1]
    roll  = length(roll ) > 1 ? vec(roll ) : one.(lat)*roll[1]
    pitch = length(pitch) > 1 ? vec(pitch) : one.(lat)*pitch[1]
    yaw   = length(yaw  ) > 1 ? vec(yaw  ) : one.(lat)*yaw[1]

    # convert deg to rad
    lat   = deg2rad.(lat)
    lon   = deg2rad.(lon)
    roll  = deg2rad.(roll)
    pitch = deg2rad.(pitch)
    yaw   = deg2rad.(yaw)

    N = length(lat)

    # if needed, create tt, otherwise get dt from tt
    if any(isnan.(tt))
        silent || @info("creating INS time data")
        tt = round.(0:dt:dt*(N-1),digits=9)
    else
        dt = round(tt[2] - tt[1],digits=9)
    end

    # if needed, create vn, ve, vd
    if any(isnan.([vn;ve;vd]))
        silent || @info("creating INS velocity data")
        (zone_utm,is_north) = utm_zone(mean(rad2deg.(lat)),mean(rad2deg.(lon)))
        lla2utm = UTMfromLLA(zone_utm,is_north,WGS84)
        utms    = lla2utm.(LLA.(rad2deg.(lat),rad2deg.(lon),alt))
        vn      =  fdm([utm.y for utm in utms]) / dt
        ve      =  fdm([utm.x for utm in utms]) / dt
        vd      = -fdm([utm.z for utm in utms]) / dt
    end

    # if needed, create fn, fe, fd
    if any(isnan.([fn;fe;fd]))
        silent || @info("creating INS specific force data")
        fn = fdm(vn) / dt
        fe = fdm(ve) / dt
        fd = fdm(vd) / dt .- g_earth
    end

    # if needed, create Cnb, otherwise get Cnb from RPY
    if any(isnan.(Cnb))
        if any(isnan.([roll;pitch;yaw]))
            silent || @info("creating INS direction cosine matrix data")
            Cnb = create_dcm(vn,ve,dt,:body2nav)
        else
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

Get inertial navigation system data at specific indices, possibly zeroed.

**Arguments:**
- `xyz`:       `XYZ` flight data struct
- `ind`:       (optional) selected data indices
- `N_zero_ll`: (optional) number of samples (instances) to zero INS lat/lon to truth (`xyz.traj`)
- `t_zero_ll`: (optional) length of time to zero INS lat/lon to truth (`xyz.traj`), will overwrite `N_zero_ll`
- `err`:       (optional) initial INS latitude and longitude error [m]

**Returns:**
- `ins`: `INS` inertial navigation system struct at `ind`
"""
function get_ins(xyz::XYZ, ind=trues(xyz.ins.N);
                 N_zero_ll::Int=0, t_zero_ll=0, err=0.0)
    N_zero_ll = t_zero_ll > 0 ? round(Int,t_zero_ll/xyz.ins.dt) : N_zero_ll
    lat = xyz.traj.lat[ind][1:N_zero_ll]
    lon = xyz.traj.lon[ind][1:N_zero_ll]
    xyz.ins(ind;N_zero_ll=N_zero_ll,err=err,lat=lat,lon=lon)
end # function get_ins

get_INS = get_ins

"""
    (ins::INS)(ind=trues(ins.N);
               N_zero_ll::Int=0, t_zero_ll=0, err=0.0,
               lat=zero(ins.lat[ind]),
               lon=zero(ins.lon[ind]))

Get inertial navigation system data at specific indices, possibly zeroed.

**Arguments:**
- `ins`:       `INS` inertial navigation system struct
- `ind`:       (optional) selected data indices
- `N_zero_ll`: (optional) number of samples (instances) to zero INS lat/lon to truth (`xyz.traj`)
- `t_zero_ll`: (optional) length of time to zero INS lat/lon to truth (`xyz.traj`), will overwrite `N_zero_ll`
- `err`:       (optional) initial INS latitude and longitude error [m]
- `lat`:       (optional) best-guess (truth) initial latitude  [rad]
- `lon`:       (optional) best-guess (truth) initial longitude [rad]

**Returns:**
- `ins`: `INS` inertial navigation system struct at `ind`
"""
function (ins::INS)(ind=trues(ins.N);
                    N_zero_ll::Int=0, t_zero_ll=0, err=0.0,
                    lat=zero(ins.lat[ind]),
                    lon=zero(ins.lon[ind]))

    ind = findall((1:ins.N) .∈ ((1:ins.N)[ind],))

    # set first N_zero_ll INS lat & lon to specified lat & lon
    N_zero_ll = t_zero_ll > 0 ? round(Int,t_zero_ll/ins.dt) : N_zero_ll
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
end # function INS

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

Get trajectory data at specific indices.

**Arguments:**
- `xyz`: `XYZ` flight data struct
- `ind`: (optional) selected data indices

**Returns:**
- `traj`: `Traj` trajectory struct at `ind`
"""
function get_traj(xyz::XYZ, ind=trues(xyz.traj.N))
    xyz.traj(ind)
end # 

get_Traj = get_traj

"""
    (traj::Traj)(ind=trues(traj.N))

Get trajectory data at specific indices.

**Arguments:**
- `traj`: `Traj` trajectory struct
- `ind`:  (optional) selected data indices

**Returns:**
- `traj`: `Traj` trajectory struct at `ind`
"""
function (traj::Traj)(ind=trues(traj.N))

    ind = findall((1:traj.N) .∈ ((1:traj.N)[ind],))

    return Traj(length(ind)   , traj.dt        , traj.tt[ind]   ,
                traj.lat[ind] , traj.lon[ind]  , traj.alt[ind]  ,
                traj.vn[ind]  , traj.ve[ind]   , traj.vd[ind]   ,
                traj.fn[ind]  , traj.fe[ind]   , traj.fd[ind]   ,
                traj.Cnb[:,:,ind])
end # function Traj

"""
    (flux::MagV)(ind=trues(length(flux.x)))

Get vector magnetometer measurements at specific indices.

**Arguments:**
- `flux`: `MagV` vector magnetometer measurement struct
- `ind`:  (optional) selected data indices

**Returns:**
- `flux`: `MagV` vector magnetometer measurement struct at `ind`
"""
function (flux::MagV)(ind=trues(length(flux.x)))
    return MagV(flux.x[ind], flux.y[ind], flux.z[ind], flux.t[ind])
end # function MagV

"""
    get_XYZ20(xyz_h5::String; tt_sort::Bool=true, silent::Bool=false)

Get `XYZ20` flight data from saved HDF5 file. Based on SGL 2020 data fields.

**Arguments:**
- `xyz_h5`:  path/name of flight data HDF5 file (`.h5` extension optional)
- `tt_sort`: (optional) if true, sort data by time (instead of line)
- `silent`:  (optional) if true, no print outs

**Returns:**
- `xyz`: `XYZ20` flight data struct
"""
function get_XYZ20(xyz_h5::String; tt_sort::Bool=true, silent::Bool=false)

    xyz_h5 = add_extension(xyz_h5,".h5")

    fields = :fields20

    silent || @info("reading in data: $xyz_h5")

    xyz = h5open(xyz_h5,"r") # read-only
    N   = maximum([length(read(xyz,k)) for k in keys(xyz)])
    d   = Dict{Symbol,Any}()
    ind = tt_sort ? sortperm(read_check(xyz,:tt,N,silent)) : trues(N)

    for field in xyz_fields(fields)
        field != :ignore && push!(d,field=>read_check(xyz,field,N,silent)[ind])
    end

    close(xyz)

    dt = N > 1 ? round(d[:tt][2] - d[:tt][1]; digits=9) : 0.1

    # using [rad] exclusively
    for field in [:lat,:lon,:ins_roll,:ins_pitch,:ins_yaw,
                  :roll_rate,:pitch_rate,:yaw_rate]
        push!(d,field => deg2rad.(d[field]))
    end

    # provided IGRF for convenience
    push!(d,:igrf => d[:mag_1_dc] - d[:mag_1_igrf])

    # trajectory velocities and specific forces from position
    push!(d,:vn =>  fdm(d[:utm_y]) / dt)
    push!(d,:ve =>  fdm(d[:utm_x]) / dt)
    push!(d,:vd => -fdm(d[:utm_z]) / dt)
    push!(d,:fn =>  fdm(d[:vn])    / dt)
    push!(d,:fe =>  fdm(d[:ve])    / dt)
    push!(d,:fd =>  fdm(d[:vd])    / dt .- g_earth)

    # Cnb direction cosine matrix (body to navigation) from yaw, pitch, roll
    push!(d,:Cnb     => zeros(3,3,N)) # unknown
    push!(d,:ins_Cnb => euler2dcm(d[:ins_roll],d[:ins_pitch],d[:ins_yaw],:body2nav))
    push!(d,:ins_P   => zeros(1,1,N)) # unknown

    # INS velocities in NED direction
    push!(d,:ins_ve => -d[:ins_vw])
    push!(d,:ins_vd => -d[:ins_vu])

    # INS specific forces from measurements, rotated wander angle (CW for NED)
    ins_f = zeros(N,3)
    for i = 1:N
        ins_f[i,:] = euler2dcm(0,0,-d[:ins_wander][i],:body2nav) * 
                     [d[:ins_acc_x][i],-d[:ins_acc_y][i],-d[:ins_acc_z][i]]
    end

    push!(d,:ins_fn => ins_f[:,1])
    push!(d,:ins_fe => ins_f[:,2])
    push!(d,:ins_fd => ins_f[:,3])

    # INS specific forces from finite differences
    # push!(d,:ins_fn => fdm(-d[:ins_vn]) / dt)
    # push!(d,:ins_fe => fdm(-d[:ins_ve]) / dt)
    # push!(d,:ins_fd => fdm(-d[:ins_vd]) / dt .- g_earth)

    return XYZ20(Traj(N,  dt, d[:tt], d[:lat], d[:lon], d[:utm_z], d[:vn],
                      d[:ve], d[:vd], d[:fn] , d[:fe] , d[:fd]   , d[:Cnb]),
                 INS( N, dt, d[:tt], d[:ins_lat], d[:ins_lon], d[:ins_alt],
                      d[:ins_vn]   , d[:ins_ve] , d[:ins_vd] , d[:ins_fn] ,
                      d[:ins_fe]   , d[:ins_fd] , d[:ins_Cnb], d[:ins_P]),
                 MagV(d[:flux_a_x], d[:flux_a_y], d[:flux_a_z], d[:flux_a_t]),
                 MagV(d[:flux_b_x], d[:flux_b_y], d[:flux_b_z], d[:flux_b_t]),
                 MagV(d[:flux_c_x], d[:flux_c_y], d[:flux_c_z], d[:flux_c_t]),
                 MagV(d[:flux_d_x], d[:flux_d_y], d[:flux_d_z], d[:flux_d_t]),
                 d[:flight]    , d[:line]      , d[:year]      , d[:doy]       ,
                 d[:utm_x]     , d[:utm_y]     , d[:utm_z]     , d[:msl]       ,
                 d[:baro]      , d[:diurnal]   , d[:igrf]      , d[:mag_1_c]   ,
                 d[:mag_1_lag] , d[:mag_1_dc]  , d[:mag_1_igrf], d[:mag_1_uc]  ,
                 d[:mag_2_uc]  , d[:mag_3_uc]  , d[:mag_4_uc]  , d[:mag_5_uc]  ,
                 d[:mag_6_uc]  , d[:ogs_mag]   , d[:ogs_alt]   , d[:ins_wander],
                 d[:ins_roll]  , d[:ins_pitch] , d[:ins_yaw]   , d[:roll_rate] ,
                 d[:pitch_rate], d[:yaw_rate]  , d[:ins_acc_x] , d[:ins_acc_y] ,
                 d[:ins_acc_z] , d[:lgtl_acc]  , d[:ltrl_acc]  , d[:nrml_acc]  ,
                 d[:pitot_p]   , d[:static_p]  , d[:total_p]   , d[:cur_com_1] ,
                 d[:cur_ac_hi] , d[:cur_ac_lo] , d[:cur_tank]  , d[:cur_flap]  ,
                 d[:cur_strb]  , d[:cur_srvo_o], d[:cur_srvo_m], d[:cur_srvo_i],
                 d[:cur_heat]  , d[:cur_acpwr] , d[:cur_outpwr], d[:cur_bat_1] ,
                 d[:cur_bat_2] , d[:vol_acpwr] , d[:vol_outpwr], d[:vol_bat_1] ,
                 d[:vol_bat_2] , d[:vol_res_p] , d[:vol_res_n] , d[:vol_back_p],
                 d[:vol_back_n], d[:vol_gyro_1], d[:vol_gyro_2], d[:vol_acc_p] ,
                 d[:vol_acc_n] , d[:vol_block] , d[:vol_back]  , d[:vol_srvo]  ,
                 d[:vol_cabt]  , d[:vol_fan]   )
end # function get_XYZ20

"""
    get_XYZ20(xyz_160_h5::String, xyz_h5::String; silent::Bool=false)

Get 160 Hz (partial) `XYZ20` flight data from saved HDF5 file and 
combine with 10 Hz `XYZ20` flight data from another saved HDF5 file. 
Data is time sorted to ensure data is aligned.

**Arguments:**
- `xyz_160_h5`: path/name of 160 Hz flight data HDF5 file (`.h5` extension optional)
- `xyz_h5`:     path/name of 10  Hz flight data HDF5 file (`.h5` extension optional)
- `silent`:     (optional) if true, no print outs

**Returns:**
- `xyz`: `XYZ20` flight data struct
"""
function get_XYZ20(xyz_160_h5::String, xyz_h5::String; silent::Bool=false)

    xyz_160_h5 = add_extension(xyz_160_h5,".h5")
    xyz_h5     = add_extension(xyz_h5    ,".h5")

    fields = :fields160

    silent || @info("reading in data: $xyz_160_h5")

    xyz = h5open(xyz_160_h5,"r") # read-only
    N   = maximum([length(read(xyz,k)) for k in keys(xyz)])
    d   = Dict{Symbol,Any}()
    ind = sortperm(read_check(xyz,:tt,N,silent))

    for field in xyz_fields(fields)
        field != :ignore && push!(d,field=>read_check(xyz,field,N,silent)[ind])
    end

    close(xyz)

    xyz = get_XYZ20(xyz_h5;tt_sort=true,silent=silent)

    xyz.mag_1_uc .= d[:mag_1_uc]
    xyz.mag_2_uc .= d[:mag_2_uc]
    xyz.mag_3_uc .= d[:mag_3_uc]
    xyz.mag_4_uc .= d[:mag_4_uc]
    xyz.mag_5_uc .= d[:mag_5_uc]
    xyz.mag_6_uc .= d[:mag_6_uc]
    xyz.flux_a.x .= d[:flux_a_x]
    xyz.flux_a.y .= d[:flux_a_y]
    xyz.flux_a.z .= d[:flux_a_z]
    xyz.flux_a.t .= d[:flux_a_t]
    xyz.flux_b.x .= d[:flux_b_x]
    xyz.flux_b.y .= d[:flux_b_y]
    xyz.flux_b.z .= d[:flux_b_z]
    xyz.flux_b.t .= d[:flux_b_t]
    xyz.flux_c.x .= d[:flux_c_x]
    xyz.flux_c.y .= d[:flux_c_y]
    xyz.flux_c.z .= d[:flux_c_z]
    xyz.flux_c.t .= d[:flux_c_t]
    xyz.flux_d.x .= d[:flux_d_x]
    xyz.flux_d.y .= d[:flux_d_y]
    xyz.flux_d.z .= d[:flux_d_z]
    xyz.flux_d.t .= d[:flux_d_t]

    return (xyz)
end # function get_XYZ20

"""
    get_XYZ21(xyz_h5::String; tt_sort::Bool=true, silent::Bool=false)

Get `XYZ21` flight data from saved HDF5 file. Based on SGL 2021 data fields.

**Arguments:**
- `xyz_h5`:  path/name of flight data HDF5 file (`.h5` extension optional)
- `tt_sort`: (optional) if true, sort data by time (instead of line)
- `silent`:  (optional) if true, no print outs

**Returns:**
- `xyz`: `XYZ21` flight data struct
"""
function get_XYZ21(xyz_h5::String; tt_sort::Bool=true, silent::Bool=false)

    xyz_h5 = add_extension(xyz_h5,".h5")

    fields = :fields21

    silent || @info("reading in data: $xyz_h5")

    xyz = h5open(xyz_h5,"r") # read-only
    N   = maximum([length(read(xyz,k)) for k in keys(xyz)])
    d   = Dict{Symbol,Any}()
    ind = tt_sort ? sortperm(read_check(xyz,:tt,N,silent)) : trues(N)

    for field in xyz_fields(fields)
        field != :ignore && push!(d,field=>read_check(xyz,field,N,silent)[ind])
    end

    close(xyz)

    dt = N > 1 ? round(d[:tt][2] - d[:tt][1]; digits=9) : 0.1

    # using [rad] exclusively
    for field in [:lat,:lon,:ins_roll,:ins_pitch,:ins_yaw]
        push!(d,field => deg2rad.(d[field]))
    end

    # provided IGRF for convenience
    push!(d,:igrf => d[:mag_1_dc] - d[:mag_1_igrf])

    # trajectory velocities and specific forces from position
    push!(d,:vn =>  fdm(d[:utm_y]) / dt)
    push!(d,:ve =>  fdm(d[:utm_x]) / dt)
    push!(d,:vd => -fdm(d[:utm_z]) / dt)
    push!(d,:fn =>  fdm(d[:vn])    / dt)
    push!(d,:fe =>  fdm(d[:ve])    / dt)
    push!(d,:fd =>  fdm(d[:vd])    / dt .- g_earth)

    # Cnb direction cosine matrix (body to navigation) from yaw, pitch, roll
    push!(d,:Cnb     => zeros(3,3,N)) # unknown
    push!(d,:ins_Cnb => euler2dcm(d[:ins_roll],d[:ins_pitch],d[:ins_yaw],:body2nav))
    push!(d,:ins_P   => zeros(1,1,N)) # unknown

    # INS velocities in NED direction
    push!(d,:ins_ve => -d[:ins_vw])
    push!(d,:ins_vd => -d[:ins_vu])

    # INS specific forces from measurements, rotated wander angle (CW for NED)
    ins_f = zeros(N,3)
    for i = 1:N
        ins_f[i,:] = euler2dcm(0,0,-d[:ins_wander][i],:body2nav) * 
                     [d[:ins_acc_x][i],-d[:ins_acc_y][i],-d[:ins_acc_z][i]]
    end

    push!(d,:ins_fn => ins_f[:,1])
    push!(d,:ins_fe => ins_f[:,2])
    push!(d,:ins_fd => ins_f[:,3])

    # INS specific forces from finite differences
    # push!(d,:ins_fn => fdm(-d[:ins_vn]) / dt)
    # push!(d,:ins_fe => fdm(-d[:ins_ve]) / dt)
    # push!(d,:ins_fd => fdm(-d[:ins_vd]) / dt .- g_earth)

    return XYZ21(Traj(N,  dt, d[:tt], d[:lat], d[:lon], d[:utm_z], d[:vn],
                      d[:ve], d[:vd], d[:fn] , d[:fe] , d[:fd]   , d[:Cnb]),
                 INS( N, dt, d[:tt], d[:ins_lat], d[:ins_lon], d[:ins_alt],
                      d[:ins_vn]   , d[:ins_ve] , d[:ins_vd] , d[:ins_fn] ,
                      d[:ins_fe]   , d[:ins_fd] , d[:ins_Cnb], d[:ins_P]),
                 MagV(d[:flux_a_x], d[:flux_a_y], d[:flux_a_z], d[:flux_a_t]),
                 MagV(d[:flux_b_x], d[:flux_b_y], d[:flux_b_z], d[:flux_b_t]),
                 MagV(d[:flux_c_x], d[:flux_c_y], d[:flux_c_z], d[:flux_c_t]),
                 MagV(d[:flux_d_x], d[:flux_d_y], d[:flux_d_z], d[:flux_d_t]),
                 d[:flight]    , d[:line]      , d[:year]      , d[:doy]       ,
                 d[:utm_x]     , d[:utm_y]     , d[:utm_z]     , d[:msl]       ,
                 d[:baro]      , d[:diurnal]   , d[:igrf]      , d[:mag_1_c]   ,
                 d[:mag_1_uc]  , d[:mag_2_uc]  , d[:mag_3_uc]  , d[:mag_4_uc]  ,
                 d[:mag_5_uc]  , d[:cur_com_1] , d[:cur_ac_hi] , d[:cur_ac_lo] ,
                 d[:cur_tank]  , d[:cur_flap]  , d[:cur_strb]  , d[:vol_block] ,
                 d[:vol_back]  , d[:vol_cabt]  , d[:vol_fan]   )
end # function get_XYZ21

"""
    get_XYZ(flight::Symbol, df_flight::DataFrame; tt_sort::Bool=true,
            reorient_vec::Bool=false, silent::Bool=false)

Get `XYZ` flight data from saved HDF5 file via DataFrame lookup.

**Arguments:**
- `flight`:       flight name (e.g., `:Flt1001`)
- `df_flight`:    lookup table (DataFrame) of flight data HDF5 files
- `tt_sort`:      (optional) if true, sort data by time (instead of line)
- `reorient_vec`: (optional) if true, align vector magnetometer with body frame
- `silent`:       (optional) if true, no print outs

**Returns:**
- `xyz`: `XYZ` flight data struct
"""
function get_XYZ(flight::Symbol, df_flight::DataFrame; tt_sort::Bool=true,
                 reorient_vec::Bool=false, silent::Bool=false)

    ind = findfirst(df_flight.flight .== flight)

    if df_flight.xyz_type[ind] == :XYZ20
        xyz = get_XYZ20(df_flight.xyz_h5[ind];tt_sort=tt_sort,silent=silent)
    elseif df_flight.xyz_type[ind] == :XYZ21
        xyz = get_XYZ21(df_flight.xyz_h5[ind];tt_sort=false,silent=silent)
    end

    reorient_vec && xyz_reorient_vec!(xyz)

    return (xyz)
end # function get_XYZ

get_xyz = get_XYZ

"""
    read_check(xyz::HDF5.File, field::Symbol, N::Int=1, silent::Bool=false)

Internal helper function to check for NaNs or missing data (returned as NaNs)
in opened flight data HDF5 file. Prints out warning for any field that
contains NaNs.

**Arguments:**
- `xyz`:    opened flight data HDF5 file
- `field`:  data field to read
- `N`:      number of samples (instances)
- `silent`: (optional) if true, no print outs

**Returns:**
- `val`: data returned for `field`
"""
function read_check(xyz::HDF5.File, field::Symbol, N::Int=1, silent::Bool=false)
    field = string.(field)
    if field in keys(xyz)
        val = read(xyz,field)
        !any(isnan.(val)) || silent || @info("$field field contains NaNs")
    else
        val = zeros(N)*NaN
        silent || @info("$field field contains NaNs")
    end
    return (val)
end # function read_check

"""
    xyz_reorient_vec!(xyz::Union{XYZ1,XYZ20,XYZ21})

Internal helper function to reorient all vector magnetometer data to best
align with the IGRF direction in the body frame. Operates in place on passed-in
`XYZ` flight data.

**Arguments:**
- `xyz`: `XYZ` flight data struct

**Returns:**
- `nothing`: `xyz` is mutated with reoriented vector magnetometer data
"""
function xyz_reorient_vec!(xyz::Union{XYZ1,XYZ20,XYZ21})
    for use_vec in field_check(xyz,MagV)
        flux = getfield(xyz,use_vec) # get vector magnetometer data
        if any(isnan,flux.t) | any(flux.t.≈0)
            @info("found NaNs, not reorienting $use_vec")
        else
            # get start time of flight (seconds past midnight) and compute IGRF directions
            ind            = trues(length(flux.x)) # do for whole flight
            norm_igrf_vecs = get_igrf(xyz,ind;frame=:body,norm_igrf=true)

            # compute optimal rotation matrix for this flight
            igrf_matrix = permutedims(reduce(hcat,norm_igrf_vecs))
            flux_matrix = permutedims(reduce(hcat,normalize.([ [x,y,z] for (x,y,z) in zip(flux.x,flux.y,flux.z) ])))
            R = get_optimal_rotation_matrix(flux_matrix,igrf_matrix)

            # correct vector magnetometer data
            for (i,Bt) in enumerate(flux.t)
                  new_flux = Bt*R*flux_matrix[i,:]
                  flux.x[i] = new_flux[1]
                  flux.y[i] = new_flux[2]
                  flux.z[i] = new_flux[3]
            end
        end
    end
end # function xyz_reorient_vec!
