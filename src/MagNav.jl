module MagNav

using Pkg.Artifacts: @artifact_str
using DSP, HDF5, LazyArtifacts, LinearAlgebra, MAT, Statistics

data_dir() = joinpath(artifact"sgl_2020_train","sgl_2020_train")

struct XYZ
    N::Int64
    DT::Float64

    LINE::Vector{Float64}
    FLT::Vector{Float64}
    TIME::Vector{Float64}
    UTM_X::Vector{Float64}
    UTM_Y::Vector{Float64}
    UTM_Z::Vector{Float64}
    MSL_Z::Vector{Float64}
    LAT::Vector{Float64}
    LONG::Vector{Float64}

    BARO::Vector{Float64}
    RADAR::Vector{Float64}
    TOPO::Vector{Float64}
    DEM::Vector{Float64}
    DRAPE::Vector{Float64}

    PITCH::Vector{Float64}
    ROLL::Vector{Float64}
    AZIMUTH::Vector{Float64}

    DIURNAL::Vector{Float64}

    COMPMAG1::Vector{Float64}
    LAGMAG1::Vector{Float64}
    DCMAG1::Vector{Float64}
    IGRFMAG1::Vector{Float64}
    UNCOMPMAG1::Vector{Float64}

    UNCOMPMAG2::Vector{Float64}
    UNCOMPMAG3::Vector{Float64}
    UNCOMPMAG4::Vector{Float64}
    UNCOMPMAG5::Vector{Float64}

    FLUXA_X::Vector{Float64}
    FLUXA_Y::Vector{Float64}
    FLUXA_Z::Vector{Float64}
    FLUXA_TOT::Vector{Float64}

    FLUXB_X::Vector{Float64}
    FLUXB_Y::Vector{Float64}
    FLUXB_Z::Vector{Float64}
    FLUXB_TOT::Vector{Float64}

    FLUXC_X::Vector{Float64}
    FLUXC_Y::Vector{Float64}
    FLUXC_Z::Vector{Float64}
    FLUXC_TOT::Vector{Float64}

    FLUXD_X::Vector{Float64}
    FLUXD_Y::Vector{Float64}
    FLUXD_Z::Vector{Float64}
    FLUXD_TOT::Vector{Float64}

    OGS_MAG::Vector{Float64}
    OGS_HGT::Vector{Float64}

    INS_ACC_X::Vector{Float64}
    INS_ACC_Y::Vector{Float64}
    INS_ACC_Z::Vector{Float64}
    INS_WANDER::Vector{Float64}

    INS_LAT::Vector{Float64}
    INS_LON::Vector{Float64}
    INS_HGT::Vector{Float64}
    INS_VEL_N::Vector{Float64}
    INS_VEL_W::Vector{Float64}
    INS_VEL_V::Vector{Float64}

    PITCHRT::Vector{Float64}
    ROLLRT::Vector{Float64}
    YAWRT::Vector{Float64}

    LONG_ACC::Vector{Float64}
    LAT_ACC::Vector{Float64}
    NORM_ACC::Vector{Float64}

    TRUE_AS::Vector{Float64}
    PITOT_P::Vector{Float64}
    STATIC_P::Vector{Float64}
    TOT_P::Vector{Float64}

    CUR_COM1::Vector{Float64}
    CUR_ACHi::Vector{Float64}
    CUR_ACLo::Vector{Float64}
    CUR_TANK::Vector{Float64}
    CUR_FLAP::Vector{Float64}
    CUR_STRB::Vector{Float64}
    CUR_SRVO_O::Vector{Float64}
    CUR_SRVO_M::Vector{Float64}
    CUR_SRVO_I::Vector{Float64}
    CUR_IHTR::Vector{Float64}
    CUR_ACPWR::Vector{Float64}
    CUR_OUTPWR::Vector{Float64}
    CUR_BAT1::Vector{Float64}
    CUR_BAT2::Vector{Float64}

    V_ACPWR::Vector{Float64}
    V_OUTPWR::Vector{Float64}
    V_BAT1::Vector{Float64}
    V_BAT2::Vector{Float64}
    V_RESp::Vector{Float64}
    V_RESn::Vector{Float64}
    V_BACKp::Vector{Float64}
    V_BACKn::Vector{Float64}
    V_GYRO1::Vector{Float64}
    V_GYRO2::Vector{Float64}
    V_ACCp::Vector{Float64}
    V_ACCn::Vector{Float64}
    V_BLOCK::Vector{Float64}
    V_BACK::Vector{Float64}
    V_SERVO::Vector{Float64}
    V_CABT::Vector{Float64}
    V_FAN::Vector{Float64}
end

include("delta_lat_lon.jl")
include("get_flight_data.jl")
include("helpers.jl")
include("TL.jl")

export
delta_lat,delta_lon,delta_north,delta_east,
get_flight_data,
detrend,
create_TL_A,create_TL_coef,fdm

end # module MagNav
