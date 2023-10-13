"""
`MagNav`: airborne `Mag`netic anomaly `Nav`igation
"""
module MagNav
    using ArchGDAL, Flux, ForwardDiff, HDF5, LazyArtifacts
    using LinearAlgebra, Plots, TOML, ZipFile, Zygote
    using BSON: bson, load
    using DataFrames: DataFrame, combine, groupby, order, sort
    using DelimitedFiles: readdlm, writedlm
    using Distributions: MvNormal, Normal, Uniform
    using DSP: Bandpass, Butterworth, Highpass, Lowpass
    using DSP: digitalfilter, fft, fftfreq, filtfilt, hamming, ifft
    using DSP: pow2db, rms, spectrogram, welch_pgram
    using ExponentialUtilities: exponential!
    using Flux: @functor, Chain, DataLoader, Dense, destructure, flatten, mse
    using FluxOptTools: optfuns
    using GLMNet: glmnetcv
    using Geodesy: LLA, LLAfromUTM, UTM, UTMfromLLA, WGS84, utm_zone
    using GlobalSensitivity: Morris, gsa
    using Interpolations: BSpline, BSplineInterpolation, Cubic, Line
    using Interpolations: Linear, OnGrid, Quadratic, ScaledInterpolation
    using Interpolations: interpolate, linear_interpolation, scale
    using IterTools: ncycle
    using KernelFunctions: PolynomialKernel, kernelmatrix
    using MAT: matopen
    using MLJLinearModels: ElasticNetRegression, fit
    using NearestNeighbors: KDTree, knn, nn
    using Optim: LBFGS, Options, only_fg!, optimize
    using Parameters: @unpack, @with_kw
    using Pkg.Artifacts: @artifact_str
    using Plots: annotate!, contourf!, heatmap, mm, plot, plot!, scatter
    using PrecompileTools
    using Random: rand, randn, randperm, seed!, shuffle
    using RecipesBase: @recipe
    using SatelliteToolboxGeomagneticField: igrf, igrfd
    using ShapML: shap
    using SpecialFunctions: gamma, gamma_inc, gamma_inc_inv
    using Statistics: cor, cov, mean, median, std, var
    using StatsBase: autocor, skewness
    using Zygote: Params, gradient, refresh

    project_toml = normpath(joinpath(@__DIR__,"../Project.toml"))

    """
        magnav_version :: VersionNumber

    MagNav.jl version as recorded in Project.toml.
    """
    const magnav_version = VersionNumber(TOML.parse(open(project_toml))["version"])

    """
        const num_mag_max = 6

    Maximum number of scalar & vector magnetometers (each)
    """
    const num_mag_max = 6

    """
        const e_earth = 0.0818191908426

    First eccentricity of Earth [-]
    """
    const e_earth = 0.0818191908426

    """
        const g_earth = 9.80665

    Gravity of Earth [m/s^2]
    """
    const g_earth = 9.80665

    """
        const r_earth = 6378137

    WGS-84 radius of Earth [m]
    """
    const r_earth = 6378137

    """
        const ω_earth = 7.2921151467e-5

    Rotation rate of Earth [rad/s]
    """
    const ω_earth = 7.2921151467e-5

    """
        const usgs

    RBG file for the standard non-linear, pseudo-log, color scale used by the
    USGS. Same as the color scale for the "Bouguer gravity anomaly map."

    Reference: https://mrdata.usgs.gov/magnetic/namag.png
    """
    const usgs = joinpath(artifact"util_files","util_files","color_scale_usgs.csv")

    """
        const icon_circle

    Point icon for optional use in path2kml(;points=true)
    """
    const icon_circle = joinpath(artifact"util_files","util_files","icon_circle.dae")

    """
        const silent_debug :: Bool

    Internal flag. If true, no verbose print outs.
    """
    const silent_debug = true

    """
        sgl_fields()

    Data fields in SGL flight data collections, contains:
    - `fields_sgl_2020.csv`
    - `fields_sgl_2021.csv`
    - `fields_sgl_160.csv`
    """
    sgl_fields() = joinpath(artifact"sgl_fields","sgl_fields")

    """
        sgl_2020_train()

    Flight data from SGL 2020 data collection - training portion, contains:
    - `Flt1002_train.h5`
    - `Flt1003_train.h5`
    - `Flt1004_train.h5`
    - `Flt1005_train.h5`
    - `Flt1006_train.h5`
    - `Flt1007_train.h5`
    - `Flt1008_train.h5`
    - `Flt1009_train.h5`
    """
    sgl_2020_train() = joinpath(artifact"sgl_2020_train","sgl_2020_train")

    """
        const emag2

    Earth Magnetic Anomaly Grid (2-arc-minute resolution)
    """
    const emag2 = joinpath(artifact"EMAG2","EMAG2.h5")

    """
        const emm720

    Enhanced Magnetic Model (vector magnetic anomaly map)
    """
    const emm720 = joinpath(artifact"EMM720_World","EMM720_World.h5")

    """
        const namad

    North American Magnetic Anomaly Map
    """
    const namad = joinpath(artifact"NAMAD_305","NAMAD_305.h5")

    """
        ottawa_area_maps()

    Magnetic anomaly maps near Ottawa, Ontario, Canada, contains:
    - `Eastern_395.h5`:   Eastern Ontario at 395 m HAE (`SEE NOTE`)
    - `Eastern_drape.h5`: Eastern Ontario on drape (`SEE NOTE`)
    - `Eastern_plot.h5`:  Eastern Ontario on drape, empty map areas left unfilled, used for plotting true survey area
    - `Renfrew_395.h5`:   Renfrew at 395 m HAE (`SEE NOTE`)
    - `Renfrew_555.h5`:   Renfrew at 555 m HAE (`SEE NOTE`)
    - `Renfrew_drape.h5`: Renfrew on drape (`SEE NOTE`)
    - `Renfrew_plot.h5`:  Renfrew on drape, empty map areas left unfilled, used for plotting true survey area
    - `HighAlt_5181.h5`:  High Altitude mini-survey (within Renfrew)
    - `Perth_800.h5`:     Perth mini-survey (within Eastern Ontario)

    `NOTE`: Missing map data within `Eastern_395.h5`, `Eastern_drape.h5`,
    `Renfrew_395.h5`, `Renfrew_555.h5`, and `Renfrew_drape.h5` has been filled
    in (using k nearest neighbors) so that the grids are completely filled.
    Care must be taken to not navigate in the filled-in areas, as this is not
    real data and only done for more accurate upward continuation of the maps.
    Use the `map_check` function on `Eastern_plot.h5` or `Renfrew_plot.h5` with
    the desired flight path data to determine if the `Eastern` and/or `Renfrew`
    maps may be used without navigating into any filled-in (artificial) areas.
    """
    ottawa_area_maps() = joinpath(artifact"ottawa_area_maps","ottawa_area_maps")

    """
        ottawa_area_maps_gxf()

    GXF versions of small magnetic anomaly maps near Ottawa, Ontario, Canada, contains:
    - `HighAlt_Mag.gxf`: High Altitude mini-survey (within Renfrew)
    - `Perth_Mag.gxf`:   Perth mini-survey (within Eastern Ontario)
    """
    ottawa_area_maps_gxf() = joinpath(artifact"ottawa_area_maps_gxf","ottawa_area_maps_gxf")

    """
        Map{T2 <: AbstractFloat}

    Abstract type `Map` for a magnetic anomaly map.
    """
    abstract type Map{T2 <: AbstractFloat} end

    """
        MapS{T2 <: AbstractFloat} <: Map{T2}

    Scalar magnetic anomaly map struct. Subtype of `Map`.

    |**Field**|**Type**|**Description**
    |:--|:--|:--
    |`map`|Matrix{`T2`}| `ny` x `nx` scalar magnetic anomaly map [nT]
    |`xx` |Vector{`T2`}| `nx` map x-direction (longitude) coordinates [rad]
    |`yy` |Vector{`T2`}| `ny` map y-direction (latitude)  coordinates [rad]
    |`alt`|`T2`        | map altitude [m]
    """
    struct MapS{T2 <: AbstractFloat} <: Map{T2}
        map :: Matrix{T2}
        xx  :: Vector{T2}
        yy  :: Vector{T2}
        alt :: T2
    end # struct MapS

    """
        MapSd{T2 <: AbstractFloat} <: Map{T2}

    Scalar magnetic anomaly map struct used to store an additional altitude
    map for drape maps. Subtype of `Map`.

    |**Field**|**Type**|**Description**
    |:--|:--|:--
    |`map`|Matrix{`T2`}| `ny` x `nx` scalar magnetic anomaly map [nT]
    |`xx` |Vector{`T2`}| `nx` map x-direction (longitude) coordinates [rad]
    |`yy` |Vector{`T2`}| `ny` map y-direction (latitude)  coordinates [rad]
    |`alt`|Matrix{`T2`}| `ny` x `nx` altitude map [m]
    """
    struct MapSd{T2 <: AbstractFloat} <: Map{T2}
        map :: Matrix{T2}
        xx  :: Vector{T2}
        yy  :: Vector{T2}
        alt :: Matrix{T2}
    end # struct MapSd

    """
        MapS3D{T2 <: AbstractFloat} <: Map{T2}

    3D (multi-level) scalar magnetic anomaly map struct. Subtype of `Map`.

    |**Field**|**Type**|**Description**
    |:--|:--|:--
    |`map`|Array{`T2,3`}| `ny` x `nx` x `nz` 3D (multi-level) scalar magnetic anomaly map [nT]
    |`xx` |Vector{`T2`} | `nx` map x-direction (longitude) coordinates [rad]
    |`yy` |Vector{`T2`} | `ny` map y-direction (latitude)  coordinates [rad]
    |`alt`|Vector{`T2`} | `nz` map altitude levels [m]
    """
    struct MapS3D{T2 <: AbstractFloat} <: Map{T2}
        map :: Array{T2,3}
        xx  :: Vector{T2}
        yy  :: Vector{T2}
        alt :: Vector{T2}
    end # struct MapS3D

    """
        MapV{T2 <: AbstractFloat} <: Map{T2}

    Vector magnetic anomaly map struct. Subtype of `Map`.

    |**Field**|**Type**|**Description**
    |:--|:--|:--
    |`mapX`|Matrix{`T2`}| `ny` x `nx` x-direction magnetic anomaly map [nT]
    |`mapY`|Matrix{`T2`}| `ny` x `nx` y-direction magnetic anomaly map [nT]
    |`mapZ`|Matrix{`T2`}| `ny` x `nx` z-direction magnetic anomaly map [nT]
    |`xx`  |Vector{`T2`}| `nx` map x-direction (longitude) coordinates [rad]
    |`yy`  |Vector{`T2`}| `ny` map y-direction (latitude)  coordinates [rad]
    |`alt` |`T2`        | map altitude [m]
    """
    struct MapV{T2 <: AbstractFloat} <: Map{T2}
        mapX :: Matrix{T2}
        mapY :: Matrix{T2}
        mapZ :: Matrix{T2}
        xx   :: Vector{T2}
        yy   :: Vector{T2}
        alt  :: T2
    end # struct MapV

    """
        MagV{T2 <: AbstractFloat}

    Vector magnetometer measurement struct.

    |**Field**|**Type**|**Description**
    |:--|:--|:--
    |`x`|Vector{`T2`}| x-direction magnetic field [nT]
    |`y`|Vector{`T2`}| y-direction magnetic field [nT]
    |`z`|Vector{`T2`}| z-direction magnetic field [nT]
    |`t`|Vector{`T2`}| total magnetic field [nT]
    """
    struct MagV{T2 <: AbstractFloat}
        x :: Vector{T2}
        y :: Vector{T2}
        z :: Vector{T2}
        t :: Vector{T2}
    end # struct MagV

    """
        const mapS_null

    Null scalar magnetic anomaly map used as default map for optional arguments.
    """
    const mapS_null = MapS(zeros(1,1),[0.0],[0.0],0.0)

    """
        Path{T1 <: Signed, T2 <: AbstractFloat} <: Path{T1, T2}

    Abstract type `Path` for a flight path.
    """
    abstract type Path{T1 <: Signed, T2 <: AbstractFloat} end

    """
        Traj{T1 <: Signed, T2 <: AbstractFloat}

    Trajectory struct, i.e., GPS or other truth flight data. Subtype of `Path`.

    |**Field**|**Type**|**Description**
    |:--|:--|:--
    |`N`  |`T1`         | number of samples (instances)
    |`dt` |`T2`         | measurement time step [s]
    |`tt` |Vector{`T2`} | time [s]
    |`lat`|Vector{`T2`} | latitude  [rad]
    |`lon`|Vector{`T2`} | longitude [rad]
    |`alt`|Vector{`T2`} | altitude  [m]
    |`vn` |Vector{`T2`} | north velocity [m/s]
    |`ve` |Vector{`T2`} | east  velocity [m/s]
    |`vd` |Vector{`T2`} | down  velocity [m/s]
    |`fn` |Vector{`T2`} | north specific force [m/s]
    |`fe` |Vector{`T2`} | east  specific force [m/s]
    |`fd` |Vector{`T2`} | down  specific force [m/s]
    |`Cnb`|Array{`T2,3`}| `3` x `3` x `N` direction cosine matrix (body to navigation) [-]
    """
    struct Traj{T1 <: Signed, T2 <: AbstractFloat} <: Path{T1, T2}
        N   :: T1
        dt  :: T2
        tt  :: Vector{T2}
        lat :: Vector{T2}
        lon :: Vector{T2}
        alt :: Vector{T2}
        vn  :: Vector{T2}
        ve  :: Vector{T2}
        vd  :: Vector{T2}
        fn  :: Vector{T2}
        fe  :: Vector{T2}
        fd  :: Vector{T2}
        Cnb :: Array{T2,3}
    end # struct Traj

    """
        INS{T1 <: Signed, T2 <: AbstractFloat} <: Path{T1, T2}

    Inertial navigation system (INS) struct. Subtype of `Path`.

    |**Field**|**Type**|**Description**
    |:--|:--|:--
    |`N`  |`T1`         | number of samples (instances)
    |`dt` |`T2`         | measurement time step [s]
    |`tt` |Vector{`T2`} | time [s]
    |`lat`|Vector{`T2`} | latitude  [rad]
    |`lon`|Vector{`T2`} | longitude [rad]
    |`alt`|Vector{`T2`} | altitude  [m]
    |`vn` |Vector{`T2`} | north velocity [m/s]
    |`ve` |Vector{`T2`} | east  velocity [m/s]
    |`vd` |Vector{`T2`} | down  velocity [m/s]
    |`fn` |Vector{`T2`} | north specific force [m/s]
    |`fe` |Vector{`T2`} | east  specific force [m/s]
    |`fd` |Vector{`T2`} | down  specific force [m/s]
    |`Cnb`|Array{`T2,3`}| `3` x `3` x `N` direction cosine matrix (body to navigation) [-]
    |`P`  |Array{`T2,3`}| `17` x `17` x `N` covariance matrix, only relevant for simulated data, otherwise zeros [-]
    """
    struct INS{T1 <: Signed, T2 <: AbstractFloat} <: Path{T1, T2}
        N   :: T1
        dt  :: T2
        tt  :: Vector{T2}
        lat :: Vector{T2}
        lon :: Vector{T2}
        alt :: Vector{T2}
        vn  :: Vector{T2}
        ve  :: Vector{T2}
        vd  :: Vector{T2}
        fn  :: Vector{T2}
        fe  :: Vector{T2}
        fd  :: Vector{T2}
        Cnb :: Array{T2,3}
        P   :: Array{T2,3}
    end # struct INS

    """
        XYZ{T1 <: Signed, T2 <: AbstractFloat}

    Abstract type `XYZ` for flight data. Simplest subtype is `XYZ0`.
    """
    abstract type XYZ{T1 <: Signed, T2 <: AbstractFloat} end

    """
        XYZ0{T1 <: Signed, T2 <: AbstractFloat} <: XYZ{T1, T2}

    Subtype of `XYZ` containing the minimum dataset required for MagNav.

    |**Field**|**Type**|**Description**
    |:--|:--|:--
    |`traj`    |Traj{`T1`,`T2`}| trajectory struct
    |`ins`     |INS{`T1`,`T2`} | inertial navigation system struct
    |`flux_a`  |MagV{`T2`}     | Flux A vector magnetometer measurement struct
    |`flight`  |Vector{`T2`}   | flight number(s)
    |`line`    |Vector{`T2`}   | line number(s), i.e., segments within `flight`
    |`mag_1_c` |Vector{`T2`}   | Mag 1 compensated (clean) scalar magnetometer measurements [nT]
    |`mag_1_uc`|Vector{`T2`}   | Mag 1 uncompensated (corrupted) scalar magnetometer measurements [nT]
    """
    struct XYZ0{T1 <: Signed, T2 <: AbstractFloat} <: XYZ{T1, T2}
        traj     :: Traj{T1,T2}
        ins      :: INS{T1,T2}
        flux_a   :: MagV{T2}
        flight   :: Vector{T2}
        line     :: Vector{T2}
        mag_1_c  :: Vector{T2}
        mag_1_uc :: Vector{T2}
    end # struct XYZ0

    """
        XYZ1{T1 <: Signed, T2 <: AbstractFloat} <: XYZ{T1, T2}

    Subtype of `XYZ` containing a flexible dataset for future use. Feed in
    NaNs for any unused fields (e.g., `aux_3`) when creating struct.

    |**Field**|**Type**|**Description**
    |:--|:--|:--
    |`traj`    |Traj{`T1`,`T2`}| trajectory struct
    |`ins`     |INS{`T1`,`T2`} | inertial navigation system struct
    |`flux_a`  |MagV{`T2`}     | Flux A vector magnetometer measurement struct
    |`flux_b`  |MagV{`T2`}     | Flux B vector magnetometer measurement struct
    |`flight`  |Vector{`T2`}   | flight number(s)
    |`line`    |Vector{`T2`}   | line number(s), i.e., segments within `flight`
    |`year`    |Vector{`T2`}   | year
    |`doy`     |Vector{`T2`}   | day of year
    |`diurnal` |Vector{`T2`}   | measured diurnal, i.e., temporal variations or space weather effects [nT]
    |`igrf`    |Vector{`T2`}   | International Geomagnetic Reference Field (IGRF), i.e., core field [nT]
    |`mag_1_c` |Vector{`T2`}   | Mag 1 compensated (clean) scalar magnetometer measurements [nT]
    |`mag_2_c` |Vector{`T2`}   | Mag 2 compensated (clean) scalar magnetometer measurements [nT]
    |`mag_3_c` |Vector{`T2`}   | Mag 3 compensated (clean) scalar magnetometer measurements [nT]
    |`mag_1_uc`|Vector{`T2`}   | Mag 1 uncompensated (corrupted) scalar magnetometer measurements [nT]
    |`mag_2_uc`|Vector{`T2`}   | Mag 2 uncompensated (corrupted) scalar magnetometer measurements [nT]
    |`mag_3_uc`|Vector{`T2`}   | Mag 3 uncompensated (corrupted) scalar magnetometer measurements [nT]
    |`aux_1`   |Vector{`T2`}   | flexible-use auxiliary data 1
    |`aux_2`   |Vector{`T2`}   | flexible-use auxiliary data 2
    |`aux_3`   |Vector{`T2`}   | flexible-use auxiliary data 3
    """
    struct XYZ1{T1 <: Signed, T2 <: AbstractFloat} <: XYZ{T1, T2}
        traj     :: Traj{T1,T2}
        ins      :: INS{T1,T2}
        flux_a   :: MagV{T2}
        flux_b   :: MagV{T2}
        flight   :: Vector{T2}
        line     :: Vector{T2}
        year     :: Vector{T2}
        doy      :: Vector{T2}
        diurnal  :: Vector{T2}
        igrf     :: Vector{T2}
        mag_1_c  :: Vector{T2}
        mag_2_c  :: Vector{T2}
        mag_3_c  :: Vector{T2}
        mag_1_uc :: Vector{T2}
        mag_2_uc :: Vector{T2}
        mag_3_uc :: Vector{T2}
        aux_1    :: Vector{T2}
        aux_2    :: Vector{T2}
        aux_3    :: Vector{T2}
    end # struct XYZ1

    """
        XYZ20{T1 <: Signed, T2 <: AbstractFloat} <: XYZ{T1, T2}

    Subtype of `XYZ` for SGL 2020 datasets.

    |**Field**|**Type**|**Description**
    |:--|:--|:--
    |`traj`      |Traj{`T1`,`T2`}| trajectory struct
    |`ins`       |INS{`T1`,`T2`} | inertial navigation system struct
    |`flux_a`    |MagV{`T2`}     | Flux A vector magnetometer measurement struct
    |`flux_b`    |MagV{`T2`}     | Flux B vector magnetometer measurement struct
    |`flux_c`    |MagV{`T2`}     | Flux C vector magnetometer measurement struct
    |`flux_d`    |MagV{`T2`}     | Flux D vector magnetometer measurement struct
    |`flight`    |Vector{`T2`}   | flight number(s)
    |`line`      |Vector{`T2`}   | line number(s), i.e., segments within `flight`
    |`year`      |Vector{`T2`}   | year
    |`doy`       |Vector{`T2`}   | day of year
    |`utm_x`     |Vector{`T2`}   | x-coordinate, WGS-84 UTM zone 18N [m]
    |`utm_y`     |Vector{`T2`}   | y-coordinate, WGS-84 UTM zone 18N [m]
    |`utm_z`     |Vector{`T2`}   | z-coordinate, GPS altitude above WGS-84 ellipsoid [m]
    |`msl`       |Vector{`T2`}   | z-coordinate, GPS altitude above EGM2008 Geoid [m]
    |`baro`      |Vector{`T2`}   | barometric altimeter [m]
    |`diurnal`   |Vector{`T2`}   | measured diurnal, i.e., temporal variations or space weather effects [nT]
    |`igrf`      |Vector{`T2`}   | International Geomagnetic Reference Field (IGRF), i.e., core field [nT]
    |`mag_1_c`   |Vector{`T2`}   | Mag 1 compensated (clean) scalar magnetometer measurements [nT]
    |`mag_1_lag` |Vector{`T2`}   | Mag 1 lag-corrected scalar magnetometer measurements [nT]
    |`mag_1_dc`  |Vector{`T2`}   | Mag 1 diurnal-corrected scalar magnetometer measurements [nT]
    |`mag_1_igrf`|Vector{`T2`}   | Mag 1 IGRF & diurnal-corrected scalar magnetometer measurements [nT]
    |`mag_1_uc`  |Vector{`T2`}   | Mag 1 uncompensated (corrupted) scalar magnetometer measurements [nT]
    |`mag_2_uc`  |Vector{`T2`}   | Mag 2 uncompensated (corrupted) scalar magnetometer measurements [nT]
    |`mag_3_uc`  |Vector{`T2`}   | Mag 3 uncompensated (corrupted) scalar magnetometer measurements [nT]
    |`mag_4_uc`  |Vector{`T2`}   | Mag 4 uncompensated (corrupted) scalar magnetometer measurements [nT]
    |`mag_5_uc`  |Vector{`T2`}   | Mag 5 uncompensated (corrupted) scalar magnetometer measurements [nT]
    |`mag_6_uc`  |Vector{`T2`}   | Mag 6 uncompensated (corrupted) scalar magnetometer measurements [nT]
    |`ogs_mag`   |Vector{`T2`}   | OGS survey diurnal-corrected, levelled, magnetic field [nT]
    |`ogs_alt`   |Vector{`T2`}   | OGS survey, GPS altitude (WGS-84) [m]
    |`ins_wander`|Vector{`T2`}   | INS-computed wander angle (ccw from north) [rad]
    |`ins_roll`  |Vector{`T2`}   | INS-computed aircraft roll [deg]
    |`ins_pitch` |Vector{`T2`}   | INS-computed aircraft pitch [deg]
    |`ins_yaw`   |Vector{`T2`}   | INS-computed aircraft yaw [deg]
    |`roll_rate` |Vector{`T2`}   | avionics-computed roll rate [deg/s]
    |`pitch_rate`|Vector{`T2`}   | avionics-computed pitch rate [deg/s]
    |`yaw_rate`  |Vector{`T2`}   | avionics-computed yaw rate [deg/s]
    |`ins_acc_x` |Vector{`T2`}   | INS x-acceleration [m/s^2]
    |`ins_acc_y` |Vector{`T2`}   | INS y-acceleration [m/s^2]
    |`ins_acc_z` |Vector{`T2`}   | INS z-acceleration [m/s^2]
    |`lgtl_acc`  |Vector{`T2`}   | avionics-computed longitudinal (forward) acceleration [g]
    |`ltrl_acc`  |Vector{`T2`}   | avionics-computed lateral (starboard) acceleration [g]
    |`nrml_acc`  |Vector{`T2`}   | avionics-computed normal (vertical) acceleration [g]
    |`pitot_p`   |Vector{`T2`}   | avionics-computed pitot pressure [kPa]
    |`static_p`  |Vector{`T2`}   | avionics-computed static pressure [kPa]
    |`total_p`   |Vector{`T2`}   | avionics-computed total pressure [kPa]
    |`cur_com_1` |Vector{`T2`}   | current sensor: aircraft radio 1 [A]
    |`cur_ac_hi` |Vector{`T2`}   | current sensor: air conditioner fan high [A]
    |`cur_ac_lo` |Vector{`T2`}   | current sensor: air conditioner fan low [A]
    |`cur_tank`  |Vector{`T2`}   | current sensor: cabin fuel pump [A]
    |`cur_flap`  |Vector{`T2`}   | current sensor: flap motor [A]
    |`cur_strb`  |Vector{`T2`}   | current sensor: strobe lights [A]
    |`cur_srvo_o`|Vector{`T2`}   | current sensor: INS outer servo [A]
    |`cur_srvo_m`|Vector{`T2`}   | current sensor: INS middle servo [A]
    |`cur_srvo_i`|Vector{`T2`}   | current sensor: INS inner servo [A]
    |`cur_heat`  |Vector{`T2`}   | current sensor: INS heater [A]
    |`cur_acpwr` |Vector{`T2`}   | current sensor: aircraft power [A]
    |`cur_outpwr`|Vector{`T2`}   | current sensor: system output power [A]
    |`cur_bat_1` |Vector{`T2`}   | current sensor: battery 1 [A]
    |`cur_bat_2` |Vector{`T2`}   | current sensor: battery 2 [A]
    |`vol_acpwr` |Vector{`T2`}   | voltage sensor: aircraft power [V]
    |`vol_outpwr`|Vector{`T2`}   | voltage sensor: system output power [V]
    |`vol_bat_1` |Vector{`T2`}   | voltage sensor: battery 1 [V]
    |`vol_bat_2` |Vector{`T2`}   | voltage sensor: battery 2 [V]
    |`vol_res_p` |Vector{`T2`}   | voltage sensor: resolver board (+) [V]
    |`vol_res_n` |Vector{`T2`}   | voltage sensor: resolver board (-) [V]
    |`vol_back_p`|Vector{`T2`}   | voltage sensor: backplane (+) [V]
    |`vol_back_n`|Vector{`T2`}   | voltage sensor: backplane (-) [V]
    |`vol_gyro_1`|Vector{`T2`}   | voltage sensor: gyroscope 1 [V]
    |`vol_gyro_2`|Vector{`T2`}   | voltage sensor: gyroscope 2 [V]
    |`vol_acc_p` |Vector{`T2`}   | voltage sensor: INS accelerometers (+) [V]
    |`vol_acc_n` |Vector{`T2`}   | voltage sensor: INS accelerometers (-) [V]
    |`vol_block` |Vector{`T2`}   | voltage sensor: block [V]
    |`vol_back`  |Vector{`T2`}   | voltage sensor: backplane [V]
    |`vol_srvo`  |Vector{`T2`}   | voltage sensor: servos [V]
    |`vol_cabt`  |Vector{`T2`}   | voltage sensor: cabinet [V]
    |`vol_fan`   |Vector{`T2`}   | voltage sensor: cooling fan [V]
    """
    struct XYZ20{T1 <: Signed, T2 <: AbstractFloat} <: XYZ{T1, T2}
        traj       :: Traj{T1,T2}
        ins        :: INS{T1,T2}
        flux_a     :: MagV{T2}
        flux_b     :: MagV{T2}
        flux_c     :: MagV{T2}
        flux_d     :: MagV{T2}
        flight     :: Vector{T2}
        line       :: Vector{T2}
        year       :: Vector{T2}
        doy        :: Vector{T2}
        utm_x      :: Vector{T2}
        utm_y      :: Vector{T2}
        utm_z      :: Vector{T2}
        msl        :: Vector{T2}
        baro       :: Vector{T2}
        diurnal    :: Vector{T2}
        igrf       :: Vector{T2}
        mag_1_c    :: Vector{T2}
        mag_1_lag  :: Vector{T2}
        mag_1_dc   :: Vector{T2}
        mag_1_igrf :: Vector{T2}
        mag_1_uc   :: Vector{T2}
        mag_2_uc   :: Vector{T2}
        mag_3_uc   :: Vector{T2}
        mag_4_uc   :: Vector{T2}
        mag_5_uc   :: Vector{T2}
        mag_6_uc   :: Vector{T2}
        ogs_mag    :: Vector{T2}
        ogs_alt    :: Vector{T2}
        ins_wander :: Vector{T2}
        ins_roll   :: Vector{T2}
        ins_pitch  :: Vector{T2}
        ins_yaw    :: Vector{T2}
        roll_rate  :: Vector{T2}
        pitch_rate :: Vector{T2}
        yaw_rate   :: Vector{T2}
        ins_acc_x  :: Vector{T2}
        ins_acc_y  :: Vector{T2}
        ins_acc_z  :: Vector{T2}
        lgtl_acc   :: Vector{T2}
        ltrl_acc   :: Vector{T2}
        nrml_acc   :: Vector{T2}
        pitot_p    :: Vector{T2}
        static_p   :: Vector{T2}
        total_p    :: Vector{T2}
        cur_com_1  :: Vector{T2}
        cur_ac_hi  :: Vector{T2}
        cur_ac_lo  :: Vector{T2}
        cur_tank   :: Vector{T2}
        cur_flap   :: Vector{T2}
        cur_strb   :: Vector{T2}
        cur_srvo_o :: Vector{T2}
        cur_srvo_m :: Vector{T2}
        cur_srvo_i :: Vector{T2}
        cur_heat   :: Vector{T2}
        cur_acpwr  :: Vector{T2}
        cur_outpwr :: Vector{T2}
        cur_bat_1  :: Vector{T2}
        cur_bat_2  :: Vector{T2}
        vol_acpwr  :: Vector{T2}
        vol_outpwr :: Vector{T2}
        vol_bat_1  :: Vector{T2}
        vol_bat_2  :: Vector{T2}
        vol_res_p  :: Vector{T2}
        vol_res_n  :: Vector{T2}
        vol_back_p :: Vector{T2}
        vol_back_n :: Vector{T2}
        vol_gyro_1 :: Vector{T2}
        vol_gyro_2 :: Vector{T2}
        vol_acc_p  :: Vector{T2}
        vol_acc_n  :: Vector{T2}
        vol_block  :: Vector{T2}
        vol_back   :: Vector{T2}
        vol_srvo   :: Vector{T2}
        vol_cabt   :: Vector{T2}
        vol_fan    :: Vector{T2}
    end # struct XYZ20

    """
        XYZ21{T1 <: Signed, T2 <: AbstractFloat} <: XYZ{T1, T2}

    Subtype of `XYZ` for SGL 2021 datasets.

    |**Field**|**Type**|**Description**
    |:--|:--|:--
    |`traj`      |Traj{`T1`,`T2`}| trajectory struct
    |`ins`       |INS{`T1`,`T2`} | inertial navigation system struct
    |`flux_a`    |MagV{`T2`}     | Flux A vector magnetometer measurement struct
    |`flux_b`    |MagV{`T2`}     | Flux B vector magnetometer measurement struct
    |`flux_c`    |MagV{`T2`}     | Flux C vector magnetometer measurement struct
    |`flux_d`    |MagV{`T2`}     | Flux D vector magnetometer measurement struct
    |`flight`    |Vector{`T2`}   | flight number(s)
    |`line`      |Vector{`T2`}   | line number(s), i.e., segments within `flight`
    |`year`      |Vector{`T2`}   | year
    |`doy`       |Vector{`T2`}   | day of year
    |`utm_x`     |Vector{`T2`}   | x-coordinate, WGS-84 UTM zone 18N [m]
    |`utm_y`     |Vector{`T2`}   | y-coordinate, WGS-84 UTM zone 18N [m]
    |`utm_z`     |Vector{`T2`}   | z-coordinate, GPS altitude above WGS-84 ellipsoid [m]
    |`msl`       |Vector{`T2`}   | z-coordinate, GPS altitude above EGM2008 Geoid [m]
    |`baro`      |Vector{`T2`}   | barometric altimeter [m]
    |`diurnal`   |Vector{`T2`}   | measured diurnal, i.e., temporal variations or space weather effects [nT]
    |`igrf`      |Vector{`T2`}   | International Geomagnetic Reference Field (IGRF), i.e., core field [nT]
    |`mag_1_c`   |Vector{`T2`}   | Mag 1 compensated (clean) scalar magnetometer measurements [nT]
    |`mag_1_uc`  |Vector{`T2`}   | Mag 1 uncompensated (corrupted) scalar magnetometer measurements [nT]
    |`mag_2_uc`  |Vector{`T2`}   | Mag 2 uncompensated (corrupted) scalar magnetometer measurements [nT]
    |`mag_3_uc`  |Vector{`T2`}   | Mag 3 uncompensated (corrupted) scalar magnetometer measurements [nT]
    |`mag_4_uc`  |Vector{`T2`}   | Mag 4 uncompensated (corrupted) scalar magnetometer measurements [nT]
    |`mag_5_uc`  |Vector{`T2`}   | Mag 5 uncompensated (corrupted) scalar magnetometer measurements [nT]
    |`cur_com_1` |Vector{`T2`}   | current sensor: aircraft radio 1 [A]
    |`cur_ac_hi` |Vector{`T2`}   | current sensor: air conditioner fan high [A]
    |`cur_ac_lo` |Vector{`T2`}   | current sensor: air conditioner fan low [A]
    |`cur_tank`  |Vector{`T2`}   | current sensor: cabin fuel pump [A]
    |`cur_flap`  |Vector{`T2`}   | current sensor: flap motor [A]
    |`cur_strb`  |Vector{`T2`}   | current sensor: strobe lights [A]
    |`vol_block` |Vector{`T2`}   | voltage sensor: block [V]
    |`vol_back`  |Vector{`T2`}   | voltage sensor: backplane [V]
    |`vol_cabt`  |Vector{`T2`}   | voltage sensor: cabinet [V]
    |`vol_fan`   |Vector{`T2`}   | voltage sensor: cooling fan [V]
    """
    struct XYZ21{T1 <: Signed, T2 <: AbstractFloat} <: XYZ{T1, T2}
        traj       :: Traj{T1,T2}
        ins        :: INS{T1,T2}
        flux_a     :: MagV{T2}
        flux_b     :: MagV{T2}
        flux_c     :: MagV{T2}
        flux_d     :: MagV{T2}
        flight     :: Vector{T2}
        line       :: Vector{T2}
        year       :: Vector{T2}
        doy        :: Vector{T2}
        utm_x      :: Vector{T2}
        utm_y      :: Vector{T2}
        utm_z      :: Vector{T2}
        msl        :: Vector{T2}
        baro       :: Vector{T2}
        diurnal    :: Vector{T2}
        igrf       :: Vector{T2}
        mag_1_c    :: Vector{T2}
        mag_1_uc   :: Vector{T2}
        mag_2_uc   :: Vector{T2}
        mag_3_uc   :: Vector{T2}
        mag_4_uc   :: Vector{T2}
        mag_5_uc   :: Vector{T2}
        cur_com_1  :: Vector{T2}
        cur_ac_hi  :: Vector{T2}
        cur_ac_lo  :: Vector{T2}
        cur_tank   :: Vector{T2}
        cur_flap   :: Vector{T2}
        cur_strb   :: Vector{T2}
        vol_block  :: Vector{T2}
        vol_back   :: Vector{T2}
        vol_cabt   :: Vector{T2}
        vol_fan    :: Vector{T2}
    end # struct XYZ21

    """
        FILTres{T2 <: AbstractFloat}

    Filter results struct.

    |**Field**|**Type**|**Description**
    |:--|:--|:--
    |`x`|Matrix{`T2`} | filtered states, i.e., E(x_t   y_1,..,y_t)
    |`P`|Array{`T2,3`}| non-linear covariance matrix
    |`r`|Matrix{`T2`} | measurement residuals [nT]
    |`c`|`Bool`       | if true, filter converged
    """
    struct FILTres{T2 <: AbstractFloat}
        x :: Matrix{T2}
        P :: Array{T2,3}
        r :: Matrix{T2}
        c :: Bool
    end # struct FILTres

    """
        CRLBout{T2 <: AbstractFloat}

    Cramér–Rao lower bound extracted output struct.

    |**Field**|**Type**|**Description**
    |:--|:--|:--
    |`lat_std` |Vector{`T2`}| latitude  1-σ [rad]
    |`lon_std` |Vector{`T2`}| longitude 1-σ [rad]
    |`alt_std` |Vector{`T2`}| altitude  1-σ [m]
    |`vn_std`  |Vector{`T2`}| north velocity 1-σ [m/s]
    |`ve_std`  |Vector{`T2`}| east  velocity 1-σ [m/s]
    |`vd_std`  |Vector{`T2`}| down  velocity 1-σ [m/s]
    |`tn_std`  |Vector{`T2`}| north tilt (attitude) 1-σ [rad]
    |`te_std`  |Vector{`T2`}| east  tilt (attitude) 1-σ [rad]
    |`td_std`  |Vector{`T2`}| down  tilt (attitude) 1-σ [rad]
    |`fogm_std`|Vector{`T2`}| FOGM 1-σ [nT]
    |`n_std`   |Vector{`T2`}| northing 1-σ [m]
    |`e_std`   |Vector{`T2`}| easting  1-σ [m]
    """
    struct CRLBout{T2 <: AbstractFloat}
        lat_std  :: Vector{T2}
        lon_std  :: Vector{T2}
        alt_std  :: Vector{T2}
        vn_std   :: Vector{T2}
        ve_std   :: Vector{T2}
        vd_std   :: Vector{T2}
        tn_std   :: Vector{T2}
        te_std   :: Vector{T2}
        td_std   :: Vector{T2}
        fogm_std :: Vector{T2}
        n_std    :: Vector{T2}
        e_std    :: Vector{T2}
    end # struct CRLBout

    """
        INSout{T2 <: AbstractFloat}

    Inertial navigation system extracted output struct.

    |**Field**|**Type**|**Description**
    |:--|:--|:--
    |`lat_std`|Vector{`T2`}| latitude  1-σ [rad]
    |`lon_std`|Vector{`T2`}| longitude 1-σ [rad]
    |`alt_std`|Vector{`T2`}| altitude  1-σ [m]
    |`n_std`  |Vector{`T2`}| northing  1-σ [m]
    |`e_std`  |Vector{`T2`}| easting   1-σ [m]
    |`lat_err`|Vector{`T2`}| latitude  error [rad]
    |`lon_err`|Vector{`T2`}| longitude error [rad]
    |`alt_err`|Vector{`T2`}| altitude  error [m]
    |`n_err`  |Vector{`T2`}| northing  error [m]
    |`e_err`  |Vector{`T2`}| easting   error [m]
    """
    struct INSout{T2 <: AbstractFloat}
        lat_std :: Vector{T2}
        lon_std :: Vector{T2}
        alt_std :: Vector{T2}
        n_std   :: Vector{T2}
        e_std   :: Vector{T2}
        lat_err :: Vector{T2}
        lon_err :: Vector{T2}
        alt_err :: Vector{T2}
        n_err   :: Vector{T2}
        e_err   :: Vector{T2}
    end # struct INSout

    """
        FILTout{T1 <: Signed, T2 <: AbstractFloat} <: Path{T1, T2}

    Filter extracted output struct. Subtype of `Path`.

    |**Field**|**Type**|**Description**
    |:--|:--|:--
    |`N`       |`T1`        | number of samples (instances)
    |`dt`      |`T2`        | measurement time step [s]
    |`tt`      |Vector{`T2`}| time [s]
    |`lat`     |Vector{`T2`}| latitude  [rad]
    |`lon`     |Vector{`T2`}| longitude [rad]
    |`alt`     |Vector{`T2`}| altitude  [m]
    |`vn`      |Vector{`T2`}| north velocity [m/s]
    |`ve`      |Vector{`T2`}| east  velocity [m/s]
    |`vd`      |Vector{`T2`}| down  velocity [m/s]
    |`tn`      |Vector{`T2`}| north tilt (attitude) [rad]
    |`te`      |Vector{`T2`}| east  tilt (attitude) [rad]
    |`td`      |Vector{`T2`}| down  tilt (attitude) [rad]
    |`ha`      |Vector{`T2`}| barometer aiding altitude [m]
    |`ah`      |Vector{`T2`}| barometer aiding vertical acceleration [m/s^2]
    |`ax`      |Vector{`T2`}| x accelerometer [m/s^2]
    |`ay`      |Vector{`T2`}| y accelerometer [m/s^2]
    |`az`      |Vector{`T2`}| z accelerometer [m/s^2]
    |`gx`      |Vector{`T2`}| x gyroscope [rad/s]
    |`gy`      |Vector{`T2`}| y gyroscope [rad/s]
    |`gz`      |Vector{`T2`}| z gyroscope [rad/s]
    |`fogm`    |Vector{`T2`}| FOGM catch-all [nT]
    |`lat_std` |Vector{`T2`}| latitude  1-σ [rad]
    |`lon_std` |Vector{`T2`}| longitude 1-σ [rad]
    |`alt_std` |Vector{`T2`}| altitude  1-σ [m]
    |`vn_std`  |Vector{`T2`}| north velocity 1-σ [m/s]
    |`ve_std`  |Vector{`T2`}| east  velocity 1-σ [m/s]
    |`vd_std`  |Vector{`T2`}| down  velocity 1-σ [m/s]
    |`tn_std`  |Vector{`T2`}| north tilt (attitude) 1-σ [rad]
    |`te_std`  |Vector{`T2`}| east  tilt (attitude) 1-σ [rad]
    |`td_std`  |Vector{`T2`}| down  tilt (attitude) 1-σ [rad]
    |`ha_std`  |Vector{`T2`}| barometer aiding altitude 1-σ [m]
    |`ah_std`  |Vector{`T2`}| barometer aiding vertical acceleration 1-σ [m/s^2]
    |`ax_std`  |Vector{`T2`}| x accelerometer 1-σ [m/s^2]
    |`ay_std`  |Vector{`T2`}| y accelerometer 1-σ [m/s^2]
    |`az_std`  |Vector{`T2`}| z accelerometer 1-σ [m/s^2]
    |`gx_std`  |Vector{`T2`}| x gyroscope 1-σ [rad/s]
    |`gy_std`  |Vector{`T2`}| y gyroscope 1-σ [rad/s]
    |`gz_std`  |Vector{`T2`}| z gyroscope 1-σ [rad/s]
    |`fogm_std`|Vector{`T2`}| FOGM catch-all 1-σ [nT]
    |`n_std`   |Vector{`T2`}| northing 1-σ [m]
    |`e_std`   |Vector{`T2`}| easting  1-σ [m]
    |`lat_err` |Vector{`T2`}| latitude  error [rad]
    |`lon_err` |Vector{`T2`}| longitude error [rad]
    |`alt_err` |Vector{`T2`}| altitude  error [m]
    |`vn_err`  |Vector{`T2`}| north velocity error [m/s]
    |`ve_err`  |Vector{`T2`}| east  velocity error [m/s]
    |`vd_err`  |Vector{`T2`}| down  velocity error [m/s]
    |`tn_err`  |Vector{`T2`}| north tilt (attitude) error [rad]
    |`te_err`  |Vector{`T2`}| east  tilt (attitude) error [rad]
    |`td_err`  |Vector{`T2`}| down  tilt (attitude) error [rad]
    |`n_err`   |Vector{`T2`}| northing error [m]
    |`e_err`   |Vector{`T2`}| easting  error [m]
    """
    struct FILTout{T1 <: Signed, T2 <: AbstractFloat} <: Path{T1, T2}
        N        :: T1
        dt       :: T2
        tt       :: Vector{T2}
        lat      :: Vector{T2}
        lon      :: Vector{T2}
        alt      :: Vector{T2}
        vn       :: Vector{T2}
        ve       :: Vector{T2}
        vd       :: Vector{T2}
        tn       :: Vector{T2}
        te       :: Vector{T2}
        td       :: Vector{T2}
        ha       :: Vector{T2}
        ah       :: Vector{T2}
        ax       :: Vector{T2}
        ay       :: Vector{T2}
        az       :: Vector{T2}
        gx       :: Vector{T2}
        gy       :: Vector{T2}
        gz       :: Vector{T2}
        fogm     :: Vector{T2}
        lat_std  :: Vector{T2}
        lon_std  :: Vector{T2}
        alt_std  :: Vector{T2}
        vn_std   :: Vector{T2}
        ve_std   :: Vector{T2}
        vd_std   :: Vector{T2}
        tn_std   :: Vector{T2}
        te_std   :: Vector{T2}
        td_std   :: Vector{T2}
        ha_std   :: Vector{T2}
        ah_std   :: Vector{T2}
        ax_std   :: Vector{T2}
        ay_std   :: Vector{T2}
        az_std   :: Vector{T2}
        gx_std   :: Vector{T2}
        gy_std   :: Vector{T2}
        gz_std   :: Vector{T2}
        fogm_std :: Vector{T2}
        n_std    :: Vector{T2}
        e_std    :: Vector{T2}
        lat_err  :: Vector{T2}
        lon_err  :: Vector{T2}
        alt_err  :: Vector{T2}
        vn_err   :: Vector{T2}
        ve_err   :: Vector{T2}
        vd_err   :: Vector{T2}
        tn_err   :: Vector{T2}
        te_err   :: Vector{T2}
        td_err   :: Vector{T2}
        n_err    :: Vector{T2}
        e_err    :: Vector{T2}
    end # struct FILTout

    """
        CompParams

    Abstract type `CompParams` for aeromagnetic compensation parameters.
    """
    abstract type CompParams end

    """
        LinCompParams <: CompParams

    Linear aeromagnetic compensation parameters struct. Subtype of `CompParams`.

    To see default parameters, type `LinCompParams()`.

    **General Parameters:**

    |**Parameter**|**Description**
    |:--|:--
    |`version`         | MagNav.jl version used to generate this struct
    |`features_setup`  | vector of features to include, not used for `model_type = :TL, :mod_TL, :map_TL`
    |`features_no_norm`| vector of features to not normalize, not used for `model_type = :TL, :mod_TL, :map_TL`
    |`model_type`      | aeromagnetic compensation model type (`see below`)
    |`y_type`          | `y` target type (`see below`)
    |`use_mag`         | scalar magnetometer to use {`:mag_1_uc`, etc.}, only used for `y_type = :c, :d, :e`
    |`use_vec`         | vector magnetometer (fluxgate) to use for "external" Tolles-Lawson `A` matrix {`:flux_a`, etc.}, not used for `model_type = :elasticnet, :plsr`
    |`data_norms`      | tuple of data normalizations, e.g., `(A_bias,A_scale,x_bias,x_scale,y_bias,y_scale)`
    |`model`           | linear model coefficients
    |`terms`           | Tolles-Lawson terms to use for Tolles-Lawson `A` matrix (or matrices) within `x` data matrix {`:permanent`,`:induced`,`:eddy`}, not used for `model_type = :TL, :mod_TL, :map_TL`
    |`terms_A`         | Tolles-Lawson terms to use for "external" Tolles-Lawson `A` matrix {`:permanent`,`:induced`,`:eddy`,`:bias`}, not used for `model_type = :elasticnet, :plsr`
    |`sub_diurnal`     | if true, subtract diurnal from scalar magnetometer measurements
    |`sub_igrf`        | if true, subtract IGRF from scalar magnetometer measurements
    |`bpf_mag`         | if true, bpf scalar magnetometer measurements in `x` data matrix, not used for `model_type = :TL, :mod_TL, :map_TL`
    |`reorient_vec`    | if true, align vector magnetometers (fluxgates) with body frame
    |`norm_type_A`     | normalization for "external" Tolles-Lawson `A` matrix, not used for `model_type = :elasticnet, :plsr` (`see below`)
    |`norm_type_x`     | normalization for `x` data matrix, not used for `model_type = :TL, :mod_TL, :map_TL` (`see below`)
    |`norm_type_y`     | normalization for `y` target vector (`see below`)

    - `model_type` options:
        - `:TL`         = classical Tolles-Lawson
        - `:mod_TL`     = modified  Tolles-Lawson
        - `:map_TL`     = map-based Tolles-Lawson
        - `:elasticnet` = elastic net (ridge regression and/or Lasso)
        - `:plsr`:      = partial least squares regression (PLSR)

    - `y_type` options:
        - `:a` = anomaly field  #1, compensated tail stinger total field scalar magnetometer measurements
        - `:b` = anomaly field  #2, interpolated `magnetic anomaly map` values
        - `:c` = aircraft field #1, difference between uncompensated cabin total field scalar magnetometer measurements and interpolated `magnetic anomaly map` values
        - `:d` = aircraft field #2, difference between uncompensated cabin and compensated tail stinger total field scalar magnetometer measurements
        - `:e` = BPF'd total field, bandpass filtered uncompensated cabin total field scalar magnetometer measurements

    - `norm_type` options:
        - `:standardize` = Z-score normalization
        - `:normalize`   = min-max normalization
        - `:scale`       = scale by maximum absolute value, bias = 0
        - `:none`        = scale by 1, bias = 0

    **Linear Model-Specific Parameters:**

    |**Parameter**|**Description**
    |:--|:--
    |`k_plsr`| number of components, only used for `model_type = :plsr`
    |`λ_TL`  | ridge parameter, only used for `model_type = :TL, :mod_TL, :map_TL`
    """
    @with_kw struct LinCompParams <: CompParams
        version          :: String          = string(magnav_version)
        features_setup   :: Vector{Symbol}  = [:mag_1_uc,:TL_A_flux_a]
        features_no_norm :: Vector{Symbol}  = Symbol[]
        model_type       :: Symbol          = :plsr
        y_type           :: Symbol          = :d
        use_mag          :: Symbol          = :mag_1_uc
        use_vec          :: Symbol          = :flux_a
        data_norms       :: Tuple{Matrix{Float64},Matrix{Float64},Vector{Float64},Vector{Float64}} = (zeros(1,1),zeros(1,1),[0.0],[0.0])
        model            :: Tuple{Vector{Float64},Float64} = ([0.0],0.0)
        terms            :: Vector{Symbol}  = [:permanent,:induced,:eddy]
        terms_A          :: Vector{Symbol}  = [:permanent,:induced,:eddy,:bias]
        sub_diurnal      :: Bool            = false
        sub_igrf         :: Bool            = false
        bpf_mag          :: Bool            = false
        reorient_vec     :: Bool            = false
        norm_type_A      :: Symbol          = :none
        norm_type_x      :: Symbol          = :none
        norm_type_y      :: Symbol          = :none
        k_plsr           :: Int64           = 18
        λ_TL             :: Float64         = 0.025
    end # struct LinCompParams

    """
        NNCompParams <: CompParams

    Neural network-based aeromagnetic compensation parameters struct.
    Subtype of `CompParams`.

    To see default parameters, type `NNCompParams()`.

    **General Parameters:**

    |**Parameter**|**Description**
    |:--|:--
    |`version`         | MagNav.jl version used to generate this struct
    |`features_setup`  | vector of features to include
    |`features_no_norm`| vector of features to not normalize
    |`model_type`      | aeromagnetic compensation model type (`see below`)
    |`y_type`          | `y` target type (`see below`)
    |`use_mag`         | scalar magnetometer to use {`:mag_1_uc`, etc.}, only used for `y_type = :c, :d, :e`
    |`use_vec`         | vector magnetometer (fluxgate) to use for "external" Tolles-Lawson `A` matrix {`:flux_a`, etc.}, not used for `model_type = :m1`
    |`data_norms`      | tuple of data normalizations, e.g., `(A_bias,A_scale,v_scale,x_bias,x_scale,y_bias,y_scale)`
    |`model`           | neural network model
    |`terms`           | Tolles-Lawson terms to use for Tolles-Lawson `A` matrix (or matrices) within `x` data matrix {`:permanent`,`:induced`,`:eddy`}
    |`terms_A`         | Tolles-Lawson terms to use for "external" Tolles-Lawson `A` matrix {`:permanent`,`:induced`,`:eddy`,`:bias`}, not used for `model_type = :m1`
    |`sub_diurnal`     | if true, subtract diurnal from scalar magnetometer measurements
    |`sub_igrf`        | if true, subtract IGRF from scalar magnetometer measurements
    |`bpf_mag`         | if true, bpf scalar magnetometer measurements in `x` data matrix
    |`reorient_vec`    | if true, align vector magnetometers (fluxgates) with body frame
    |`norm_type_A`     | normalization for "external" Tolles-Lawson `A` matrix, only used for `model_type = :m2*` (`see below`)
    |`norm_type_x`     | normalization for `x` data matrix (`see below`)
    |`norm_type_y`     | normalization for `y` target vector (`see below`)

    - `model_type` options are broken into 3 architectures, with `1` being a standard feedforward neural network and `2,3` being used in conjunction with Tolles-Lawson
        - `:m1`   = standard feedforward neural network (NN)
        - `:m2a`  = NN determines Tolles-Lawson (TL) coefficients
        - `:m2b`  = NN determines additive correction to classical TL
        - `:m2c`  = NN determines additive correction to classical TL, TL coefficients tuned as well
        - `:m2d`  = NN determines additive correction to each TL coefficient
        - `:m3tl` = no NN, TL coefficients fine-tuned via SGD, without Taylor expansion for `y_type` :b and :c (for testing)
        - `:m3s`  = NN determines scalar correction to TL, using expanded TL vector terms for explainability
        - `:m3v`  = NN determines vector correction to TL, using expanded TL vector terms for explainability
        - `:m3sc` = `:m3s` with curriculum learning based on TL error
        - `:m3vc` = `:m3v` with curriculum learning based on TL error
        - `:m3w`  = `:m3s` with window NN for temporal dataset
        - `:m3tf` = `:m3s` with time series transformer for temporal dataset

    - `y_type` options:
        - `:a` = anomaly field  #1, compensated tail stinger total field scalar magnetometer measurements
        - `:b` = anomaly field  #2, interpolated `magnetic anomaly map` values
        - `:c` = aircraft field #1, difference between uncompensated cabin total field scalar magnetometer measurements and interpolated `magnetic anomaly map` values
        - `:d` = aircraft field #2, difference between uncompensated cabin and compensated tail stinger total field scalar magnetometer measurements
        - `:e` = BPF'd total field, bandpass filtered uncompensated cabin total field scalar magnetometer measurements

    - `norm_type` options:
        - `:standardize` = Z-score normalization
        - `:normalize`   = min-max normalization
        - `:scale`       = scale by maximum absolute value, bias = 0
        - `:none`        = scale by 1, bias = 0

    **Neural Network-Based Model-Specific Parameters:**

    |**Parameter**|**Description**
    |:--|:--
    |`TL_coef`     | Tolles-Lawson coefficients, not used for `model_type = :m1, :m2a`
    |`η_adam`      | learning rate for Adam optimizer
    |`epoch_adam`  | number of epochs for Adam optimizer
    |`epoch_lbfgs` | number of epochs for LBFGS optimizer
    |`hidden`      | hidden layers & nodes (e.g., `[8,8]` for 2 hidden layers, 8 nodes each)
    |`activation`  | activation function (`see below`)
    |`batchsize`   | mini-batch size
    |`frac_train`  | fraction of training data used for training (remainder for validation), only used for Adam optimizer
    |`α_sgl`       | Lasso (`α_sgl=0`) vs group Lasso (`α_sgl=1`) balancing parameter {0:1}
    |`λ_sgl`       | sparse group Lasso parameter, typically ~1e-5 (if nonzero)
    |`k_pca`       | number of components for pre-processing with PCA + whitening, `-1` to ignore
    |`drop_fi`     | if true, perform drop-column feature importance
    |`drop_fi_bson`| path/name of drop-column feature importance data BSON file to save/load (`.bson` extension optional)
    |`drop_fi_csv` | path/name of drop-column feature importance data CSV file to save (`.csv` extension optional)
    |`perm_fi`     | if true, perform permutation feature importance
    |`perm_fi_csv` | path/name of permutation feature importance data CSV file to save (`.csv` extension optional)

    - `activation` options can be visualized by running `plot_activation()`
        - `relu`  = rectified linear unit
        - `σ`     = sigmoid (logistic function)
        - `swish` = self-gated
        - `tanh`  = hyperbolic tan
    """
    @with_kw struct NNCompParams <: CompParams
        version          :: String          = string(magnav_version)
        features_setup   :: Vector{Symbol}  = [:mag_1_uc,:TL_A_flux_a]
        features_no_norm :: Vector{Symbol}  = Symbol[]
        model_type       :: Symbol          = :m1
        y_type           :: Symbol          = :d
        use_mag          :: Symbol          = :mag_1_uc
        use_vec          :: Symbol          = :flux_a
        data_norms       :: Tuple{Matrix{Float32},Matrix{Float32},Matrix{Float32},Matrix{Float32},Matrix{Float32},Vector{Float32},Vector{Float32}} = (zeros(Float32,1,1),zeros(Float32,1,1),zeros(Float32,1,1),zeros(Float32,1,1),zeros(Float32,1,1),[0f0],[0f0])
        model            :: Chain           = Chain()
        terms            :: Vector{Symbol}  = [:permanent,:induced,:eddy]
        terms_A          :: Vector{Symbol}  = [:permanent,:induced,:eddy,:bias]
        sub_diurnal      :: Bool            = false
        sub_igrf         :: Bool            = false
        bpf_mag          :: Bool            = false
        reorient_vec     :: Bool            = false
        norm_type_A      :: Symbol          = :none
        norm_type_x      :: Symbol          = :standardize
        norm_type_y      :: Symbol          = :standardize
        TL_coef          :: Vector{Float64} = zeros(19)
        η_adam           :: Float64         = 0.001
        epoch_adam       :: Int64           = 5
        epoch_lbfgs      :: Int64           = 0
        hidden           :: Vector{Int64}   = [8]
        activation       :: Function        = swish
        batchsize        :: Int64           = 2048
        frac_train       :: Float64         = 14/17
        α_sgl            :: Float64         = 1.0
        λ_sgl            :: Float64         = 0.0
        k_pca            :: Int64           = -1
        drop_fi          :: Bool            = false
        drop_fi_bson     :: String          = "drop_fi"
        drop_fi_csv      :: String          = "drop_fi"
        perm_fi          :: Bool            = false
        perm_fi_csv      :: String          = "perm_fi"
    end # struct NNCompParams

    """
        EKF_RT

    Real-time (RT) extended Kalman filter (EKF) struct, mutable.

    |**Field**|**Type**|**Description**
    |:--|:--|:--
    |`P`       |Matrix{`Float64`}| non-linear covariance matrix
    |`Qd`      |Matrix{`Float64`}| discrete time process/system noise matrix
    |`R`       |Float64          | measurement (white) noise variance
    |`baro_tau`|Float64          | barometer time constant [s]
    |`acc_tau` |Float64          | accelerometer time constant [s]
    |`gyro_tau`|Float64          | gyroscope time constant [s]
    |`fogm_tau`|Float64          | FOGM catch-all time constant [s]
    |`date`    |Float64          | measurement date for IGRF [yr]
    |`core`    |Bool             | if true, include core magnetic field in measurement
    |`nx`      |Int64            | total state dimension
    |`ny`      |Int64            | measurement dimension
    |`t`       |Float64          | time [s]
    |`x`       |Vector{`Float64`}| filtered states, i.e., E(x_t   y_1,..,y_t)
    |`r`       |Vector{`Float64`}| measurement residual
    """
    @with_kw mutable struct EKF_RT
        P        :: Matrix{Float64} = create_P0()
        Qd       :: Matrix{Float64} = create_Qd()
        R        :: Float64         = 1.0
        baro_tau :: Float64         = 3600.0
        acc_tau  :: Float64         = 3600.0
        gyro_tau :: Float64         = 3600.0
        fogm_tau :: Float64         = 600.0
        date     :: Float64         = get_years(2020,185)
        core     :: Bool            = false
        nx       :: Int64           = size(P,1)
        ny       :: Int64           = 1
        t        :: Float64         = -1
        x        :: Vector{Float64} = zeros(nx)
        r        :: Vector{Float64} = zeros(ny)
    end # mutable struct EKF_RT

    """
        Map_Cache

    Map cache struct, mutable.

    |**Field**|**Type**|**Description**
    |:--|:--|:--
    |`maps`          |Vector{`MapS`{`Float64`}}| vector of `MapS` scalar magnetic anomaly map structs, not filled-in
    |`maps_filled`   |Vector{`MapS`{`Float64`}}| vector of `MapS` scalar magnetic anomaly map structs, filled-in
    |`map_sort_ind`  |Vector{`Int64`}          | `maps` indices sorted by altitude
    |`fallback`      |`MapS`{`Float64`}        | fallback `MapS` scalar magnetic anomaly map struct, filled-in
    |`map_cache`     |Dict                     | `maps`     cache of scalar map interpolation functions at multiple altitudes
    |`fallback_cache`|Dict                     | `fallback` cache of scalar map interpolation functions at multiple altitudes
    |`dz`            |Real                     | step size between map altitude levels [m]
    """
    @with_kw mutable struct Map_Cache
        maps           :: Vector{MapS{Float64}}
        maps_filled    :: Vector{MapS{Float64}} = [map_fill(map_trim(m)) for m in maps]
        map_sort_ind   :: Vector{Int64}         = sortperm([m.alt for m in maps])
        fallback       :: MapS{Float64}         = map_fill(map_trim(get_map(namad)))
        map_cache      :: Dict                  = Dict{Tuple{Int64,Int64},Any}()
        fallback_cache :: Dict                  = Dict{Int64, Any}()
        dz             :: Real                  = 100
    end # mutable struct Map_Cache

    include("analysis_util.jl")
    include("baseline_plots.jl")
    include("compensation.jl")
    include("create_XYZ.jl")
    include("dcm.jl")
    include("ekf_&_crlb.jl")
    include("ekf_online_nn.jl")
    include("ekf_online.jl")
    include("eval_filt.jl")
    include("get_map.jl")
    include("get_XYZ.jl")
    include("google_earth.jl")
    include("map_fft.jl")
    include("map_functions.jl")
    include("model_functions.jl")
    include("mpf.jl")
    include("nekf.jl")
    include("tolles_lawson.jl")
    include("xyz2h5.jl")

    export
    LinCompParams,NNCompParams,EKF_RT,Map_Cache,
    dn2dlat,de2dlon,dlat2dn,dlon2de,linreg,detrend,get_bpf,bpf_data,bpf_data!,
    get_x,get_y,get_Axy,get_nn_m,sparse_group_lasso,err_segs,
    norm_sets,denorm_sets,get_ind,chunk_data,predict_rnn_full,
    predict_rnn_windowed,krr,eval_shapley,plot_shapley,eval_gsa,
    get_IGRF,get_igrf,project_body_field_to_2d_igrf,get_optimal_rotation_matrix,
    get_days_per_year,get_years,filter_events!,filter_events, gif_animation_m3,
    plot_basic,plot_activation,plot_mag,plot_mag_c,
    plot_PSD,plot_spectrogram,plot_frequency,plot_correlation,
    comp_train,comp_test,comp_m2bc_test,comp_m3_test,comp_train_test,
    create_XYZ0,create_traj,create_ins,create_mag_c,corrupt_mag,create_flux,
    create_informed_xyz,
    euler2dcm,dcm2euler,
    ekf,crlb,
    ekf_online_nn,ekf_online_nn_setup,
    ekf_online,ekf_online_setup,
    run_filt,eval_results,eval_crlb,eval_ins,eval_filt,
    plot_filt!,plot_filt,plot_filt_err,plot_mag_map,plot_mag_map_err,
    get_autocor,plot_autocor,gif_ellipse,
    get_map,save_map,get_comp_params,save_comp_params,
    get_XYZ20,get_XYZ21,get_XYZ,get_xyz,get_XYZ0,get_XYZ1,
    get_flux,get_magv,get_MagV,get_traj,get_Traj,get_ins,get_INS,
    map2kmz,path2kml,
    upward_fft,vector_fft,downward_L,psd,
    map_interpolate,map_itp,map_get_gxf,map_trim,
    map_correct_igrf!,map_correct_igrf,map_fill!,map_fill,
    map_chessboard!,map_chessboard,map_utm2lla!,map_utm2lla,map_gxf2h5,
    plot_map!,plot_map,plot_path!,plot_path,plot_events!,map_check,get_map_val,
    get_cached_map,map_border,map_border_sort,map_resample,map_combine,
    create_model,fogm,
    mpf,
    nekf,nekf_train,
    create_TL_A,create_TL_coef,fdm,
    xyz2h5

    # #* tried various combinations of function calls, always worse, WIP
    # @setup_workload begin
    #     mapS = MapS(ones(3,3),[0.1:0.1:0.3;],[0.1:0.1:0.3;],1000.0)
    #     xyz  = create_XYZ0(mapS;alt=mapS.alt,t=5,N_waves=0,silent=true)
    #     @compile_workload begin
    #         itp_mapS = map_itp(mapS)
    #         ind      = get_ind(xyz)
    #         terms    = [:permanent3,:induced3]
    #         TL_coef  = create_TL_coef(xyz.flux_a,xyz.mag_1_c;terms=terms)
    #         for model_type in [:m1,:m2c,:m3v]
    #             comp_params = NNCompParams(model_type = model_type,
    #                                        terms      = terms,
    #                                        terms_A    = terms,
    #                                        TL_coef    = TL_coef,
    #                                        epoch_adam = 1,
    #                                        batchsize  = 5)
    #             comp_train_test(comp_params,xyz,xyz,ind,ind;silent=true)
    #         end
    #         ekf(xyz.ins,xyz.mag_1_c,itp_mapS;R=2.0)
    #     end
    # end

end # module MagNav
