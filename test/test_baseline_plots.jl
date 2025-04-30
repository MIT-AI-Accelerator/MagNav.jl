using MagNav, Test, MAT
using DSP: hamming
using Plots: Plot

test_file = joinpath(@__DIR__,"test_data/test_data_ins.mat")
ins_data  = matopen(test_file,"r") do file
    read(file,"ins_data")
end

test_file = joinpath(@__DIR__,"test_data/test_data_traj.mat")
traj_data = matopen(test_file,"r") do file
    read(file,"traj")
end

ins_lat  = deg2rad.(vec(ins_data["lat"]))
ins_lon  = deg2rad.(vec(ins_data["lon"]))
ins_alt  = vec(ins_data["alt"])
ins_vn   = vec(ins_data["vn"])
ins_ve   = vec(ins_data["ve"])
ins_vd   = vec(ins_data["vd"])
ins_fn   = vec(ins_data["fn"])
ins_fe   = vec(ins_data["fe"])
ins_fd   = vec(ins_data["fd"])
ins_Cnb  = ins_data["Cnb"]

tt       = vec(traj_data["tt"])
lat      = deg2rad.(vec(traj_data["lat"]))
lon      = deg2rad.(vec(traj_data["lon"]))
alt      = vec(traj_data["alt"])
vn       = vec(traj_data["vn"])
ve       = vec(traj_data["ve"])
vd       = vec(traj_data["vd"])
fn       = vec(traj_data["fn"])
fe       = vec(traj_data["fe"])
fd       = vec(traj_data["fd"])
Cnb      = traj_data["Cnb"]
mag_1_c  = vec(traj_data["mag_1_c"])
mag_1_uc = vec(traj_data["mag_1_uc"])
flux_a_x = vec(traj_data["flux_a_x"])
flux_a_y = vec(traj_data["flux_a_y"])
flux_a_z = vec(traj_data["flux_a_z"])
flux_a_t = sqrt.(flux_a_x.^2+flux_a_y.^2+flux_a_z.^2)
N        = length(lat)
dt       = tt[2] - tt[1]

ins_P  = zeros(1,1,N)
val    = one.(lat)
traj   = MagNav.Traj(N,dt,tt,lat,lon,alt,vn,ve,vd,fn,fe,fd,Cnb)
ins    = MagNav.INS( N,dt,tt,ins_lat,ins_lon,ins_alt,ins_vn,ins_ve,ins_vd,
                     ins_fn,ins_fe,ins_fd,ins_Cnb,ins_P)
flux_a = MagNav.MagV(flux_a_x,flux_a_y,flux_a_z,flux_a_t)
xyz    = MagNav.XYZ0("test",traj,ins,flux_a,val,val,val,val,val,val,mag_1_c,mag_1_uc)

ind = trues(N)
ind[51:end] .= false

show_plot = false

@testset "plot_basic tests" begin
    @test plot_basic(tt,mag_1_c;show_plot) isa Plot
    @test plot_basic(tt,mag_1_c,ind;
                     lab       = "mag_1_c",
                     xlab      = "time [min]",
                     ylab      = "magnetic field [nT]",
                     show_plot = false) isa Plot
end

@testset "plot_activation tests" begin
    @test plot_activation(;show_plot) isa Plot
    @test plot_activation([:relu,:swish];
                           plot_deriv = true,
                           show_plot  = false,
                           save_plot  = false) isa Plot
end

@testset "plot_mag tests" begin
    @test plot_mag(xyz;show_plot) isa Plot
    @test plot_mag(xyz;use_mags=[:comp_mags],show_plot) isa Plot
    @test plot_mag(xyz;use_mags=[:flux_a],show_plot) isa Plot
    @test plot_mag(xyz;use_mags=[:flight],show_plot) isa Plot
    @test plot_mag(xyz;
                   ind          = ind,
                   detrend_data = true,
                   use_mags     = [:mag_1_c,:mag_1_uc],
                   vec_terms    = [:all],
                   ylim         = (-300,300),
                   dpi          = 100,
                   show_plot    = false,
                   save_plot    = false) isa Plot
    @test plot_mag(xyz;
                   ind          = ind,
                   detrend_data = true,
                   use_mags     = [:comp_mags],
                   vec_terms    = [:all],
                   ylim         = (-1,1),
                   dpi          = 100,
                   show_plot    = false,
                   save_plot    = false) isa Plot
    @test plot_mag(xyz;
                   ind          = ind,
                   detrend_data = true,
                   use_mags     = [:flux_a],
                   vec_terms    = [:all],
                   ylim         = (-1000,1000),
                   dpi          = 100,
                   show_plot    = false,
                   save_plot    = false) isa Plot
    @test plot_mag(xyz;
                   ind          = ind,
                   detrend_data = true,
                   use_mags     = [:flight],
                   vec_terms    = [:all],
                   ylim         = (-1,1),
                   dpi          = 100,
                   show_plot    = false,
                   save_plot    = false) isa Plot
end

@testset "plot_mag_c tests" begin
    @test plot_mag_c(xyz,xyz;show_plot) isa Plot
    @test_throws ErrorException plot_mag_c(xyz,xyz;use_mags=[:test],show_plot)
    @test plot_mag_c(xyz,xyz;
                     ind           = .!ind,
                     ind_comp      = ind,
                     detrend_data  = false,
                     λ             = 0.0025,
                     terms         = [:p,:i,:e,:b],
                     pass1         = 0.2,
                     pass2         = 0.8,
                     fs            = 1.0,
                     use_mags      = [:mag_1_uc],
                     use_vec       = :flux_a,
                     plot_diff     = true,
                     plot_mag_1_uc = false,
                     plot_mag_1_c  = false,
                     dpi           = 100,
                     ylim          = (-50,50),
                     show_plot     = false,
                     save_plot     = false) isa Plot
end

@testset "plot_PSD tests" begin
    @test MagNav.plot_PSD(mag_1_c;show_plot) isa Plot
    @test MagNav.plot_PSD(mag_1_c,1;
                          window    = hamming,
                          dpi       = 100,
                          show_plot = false,
                          save_plot = false) isa Plot
end

@testset "plot_spectrogram tests" begin
    @test MagNav.plot_spectrogram(mag_1_c;show_plot) isa Plot
    @test MagNav.plot_spectrogram(mag_1_c,1;
                                  window    = hamming,
                                  dpi       = 100,
                                  show_plot = false,
                                  save_plot = false) isa Plot
end

@testset "plot_frequency tests" begin
    @test plot_frequency(xyz;show_plot) isa Plot
    @test plot_frequency(xyz;
                         ind          = ind,
                         field        = :mag_1_c,
                         freq_type    = :spec,
                         detrend_data = false,
                         window       = hamming,
                         dpi          = 100,
                         show_plot    = false,
                         save_plot    = false) isa Plot
end

@testset "plot_correlation tests" begin
    @test plot_correlation(xyz;show_plot) isa Plot
    @test plot_correlation(xyz,:mag_1_uc,:mag_1_c,ind;
                           lim       = 0.5,
                           dpi       = 100,
                           show_plot = false,
                           save_plot = false,
                           silent    = true) isa Plot
    @test plot_correlation(mag_1_uc,mag_1_c;show_plot) isa Plot
    @test plot_correlation(mag_1_uc,mag_1_c,:mag_1_uc,:mag_1_c;
                           lim       = 0.5,
                           dpi       = 100,
                           show_plot = false,
                           save_plot = false,
                           silent    = true) isa Plot
    @test plot_correlation(xyz;lim=Inf,show_plot) isa Nothing
end

feat_set  = [:mag_1_c,:mag_1_uc,:TL_A_flux_a,:flight]

@testset "plot_correlation_matrix tests" begin
    @test plot_correlation_matrix(xyz,ind;Nmax=10      ,show_plot) isa Plot
    @test plot_correlation_matrix(xyz,ind,feat_set[1:2];show_plot) isa Plot
    @test plot_correlation_matrix(xyz,ind,feat_set[3:3];show_plot) isa Plot
    @test plot_correlation_matrix(xyz,ind,feat_set[2:3];show_plot) isa Plot
    @test plot_correlation_matrix(xyz,ind,feat_set[1:3];show_plot) isa Plot
    @test_throws AssertionError plot_correlation_matrix(xyz,ind,feat_set[1:1];show_plot)
    @test_throws AssertionError plot_correlation_matrix(xyz,ind,feat_set[1:4];show_plot)
end
