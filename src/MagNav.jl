module MagNav

using Pkg.Artifacts: @artifact_str

using BenchmarkTools, DSP, FFTW, ForwardDiff, HDF5
using Interpolations, LinearAlgebra, Plots, StaticArrays, Statistics

data_dir() = joinpath(artifact"flight_data", "flight_data")

struct mapS
    map::Matrix{Float64}
    xx::Vector{Float64}
    yy::Vector{Float64}
    alt::Float64
    dn::Float64
    de::Float64
end

struct mapV
    mapX::Matrix{Float64}
    mapY::Matrix{Float64}
    mapZ::Matrix{Float64}
    xx::Vector{Float64}
    yy::Vector{Float64}
    alt::Float64
    dn::Float64
    de::Float64
end

struct xyz
    N::Int64
    dt::Float64

    tie_line::Vector{Float64}
    flight::Vector{Float64}
    tt::Vector{Float64}
    utmX::Vector{Float64}
    utmY::Vector{Float64}
    utmZ::Vector{Float64}
    alt::Vector{Float64}
    lat::Vector{Float64}
    lon::Vector{Float64}

    baro::Vector{Float64}
    radar::Vector{Float64}
    topo::Vector{Float64}
    dem::Vector{Float64}
    drape::Vector{Float64}

    ins_pitch::Vector{Float64}
    ins_roll::Vector{Float64}
    ins_azim::Vector{Float64}

    diurnal::Vector{Float64}

    mag_1_c::Vector{Float64}
    mag_1_lag::Vector{Float64}
    mag_1_dc::Vector{Float64}
    mag_1_igrf::Vector{Float64}
    mag_1_uc::Vector{Float64}

    mag_2_uc::Vector{Float64}
    mag_3_uc::Vector{Float64}
    mag_4_uc::Vector{Float64}
    mag_5_uc::Vector{Float64}

    flux_b_x::Vector{Float64}
    flux_b_y::Vector{Float64}
    flux_b_z::Vector{Float64}
    flux_b_t::Vector{Float64}

    flux_c_x::Vector{Float64}
    flux_c_y::Vector{Float64}
    flux_c_z::Vector{Float64}
    flux_c_t::Vector{Float64}

    flux_d_x::Vector{Float64}
    flux_d_y::Vector{Float64}
    flux_d_z::Vector{Float64}
    flux_d_t::Vector{Float64}

    ogs_mag::Vector{Float64}
    ogs_alt::Vector{Float64}

    ins_acc_x::Vector{Float64}
    ins_acc_y::Vector{Float64}
    ins_acc_z::Vector{Float64}
    ins_wander::Vector{Float64}

    ins_lat::Vector{Float64}
    ins_lon::Vector{Float64}
    ins_alt::Vector{Float64}
    ins_vn::Vector{Float64}
    ins_vw::Vector{Float64}
    ins_vu::Vector{Float64}

    pitch_rt::Vector{Float64}
    roll_rt::Vector{Float64}
    yaw_rt::Vector{Float64}

    lon_acc::Vector{Float64}
    lat_acc::Vector{Float64}
    alt_acc::Vector{Float64}

    true_as::Vector{Float64}
    pitot_p::Vector{Float64}
    static_p::Vector{Float64}
    total_p::Vector{Float64}

    cur_com_1::Vector{Float64}
    cur_ac_hi::Vector{Float64}
    cur_ac_lo::Vector{Float64}
    cur_tank::Vector{Float64}
    cur_flap::Vector{Float64}
    cur_strb::Vector{Float64}
    cur_srvo_o::Vector{Float64}
    cur_srvo_m::Vector{Float64}
    cur_srvo_i::Vector{Float64}
    cur_heat::Vector{Float64}
    cur_acpwr::Vector{Float64}
    cur_outpwr::Vector{Float64}
    cur_bat_1::Vector{Float64}
    cur_bat_2::Vector{Float64}

    vol_acpwr::Vector{Float64}
    vol_outpwr::Vector{Float64}
    vol_bat_1::Vector{Float64}
    vol_bat_2::Vector{Float64}
    vol_res_p::Vector{Float64}
    vol_res_n::Vector{Float64}
    vol_back_p::Vector{Float64}
    vol_back_n::Vector{Float64}
    vol_gyro_1::Vector{Float64}
    vol_gyro_2::Vector{Float64}
    vol_acc_p::Vector{Float64}
    vol_acc_n::Vector{Float64}
    vol_block::Vector{Float64}
    vol_back::Vector{Float64}
    vol_servo::Vector{Float64}
    vol_cabt::Vector{Float64}
    vol_fan::Vector{Float64}
end

include("create_TL_Amat.jl")
include("create_TL_coef.jl")
include("delta_lat_lon.jl")
include("fft_maps.jl")
include("gen_interp_map.jl")
include("get_flight_data.jl")
include("get_map_data.jl")
include("helpers.jl")

export
create_TL_Amat,central_fdm,
create_TL_coef,
delta_lat,delta_lon,delta_north,delta_east,
upward_fft,create_K,
gen_interp_map,
get_flight_data,
get_map_data,
detrend,map_grad,
mean

end # module
