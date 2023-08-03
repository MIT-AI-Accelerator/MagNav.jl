using MagNav, Test, MAT, Plots

test_file = joinpath(@__DIR__,"test_data/test_data_ins.mat")
ins_data  = matopen(test_file,"r") do file
    read(file,"ins_data")
end

test_file = joinpath(@__DIR__,"test_data/test_data_map.mat")
map_data  = matopen(test_file,"r") do file
    read(file,"map_data")
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

map_map  = map_data["map"]
map_xx   = deg2rad.(vec(map_data["xx"]))
map_yy   = deg2rad.(vec(map_data["yy"]))
map_alt  = map_data["alt"]

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

traj = MagNav.Traj(N,dt,tt,lat,lon,alt,vn,ve,vd,fn,fe,fd,Cnb)
ins  = MagNav.INS( N,dt,tt,ins_lat,ins_lon,ins_alt,ins_vn,ins_ve,ins_vd,
                   ins_fn,ins_fe,ins_fd,ins_Cnb,zeros(1,1,1))
ins2 = MagNav.INS( N,dt,tt,ins_lat,ins_lon,ins_alt,ins_vn,ins_ve,ins_vd,
                   ins_fn,ins_fe,ins_fd,ins_Cnb,ones(3,3,N))

mapS       = MagNav.MapS(map_map,map_xx,map_yy,map_alt)
itp_mapS   = map_interpolate(mapS,:linear) # linear to match MATLAB
itp_mapS3D = map_interpolate(upward_fft(mapS,[mapS.alt,mapS.alt+5]),:linear)
map_val    = itp_mapS.(lon,lat)

@testset "run_filt tests" begin
    @test typeof(run_filt(traj,ins,mag_1_c,itp_mapS,:ekf;
                          extract  = true,
                          run_crlb = true))  <: Tuple{MagNav.CRLBout,MagNav.INSout,MagNav.FILTout}
    @test typeof(run_filt(traj,ins,mag_1_c,itp_mapS,:ekf;
                          extract  = true,
                          run_crlb = false)) <: MagNav.FILTout
    @test typeof(run_filt(traj,ins,mag_1_c,itp_mapS,:mpf;
                          extract  = false,
                          run_crlb = true))  <: Tuple{MagNav.FILTres,Array}
          Tuple{MagNav.CRLBout,MagNav.INSout,MagNav.FILTout}
    @test typeof(run_filt(traj,ins,mag_1_c,itp_mapS,:mpf;
                          extract  = false,
                          run_crlb = false)) <: MagNav.FILTres
    @test_throws ErrorException typeof(run_filt(traj,ins,mag_1_c,itp_mapS,:test))
    @test typeof(run_filt(traj,ins,mag_1_c,itp_mapS,[:ekf,:mpf])) <: Nothing
end

(filt_res,crlb_P) = run_filt(traj,ins,mag_1_c,itp_mapS,:ekf;extract=false)

@testset "eval_results tests" begin
    @test typeof(eval_results(traj,ins,filt_res,crlb_P)) <: 
          Tuple{MagNav.CRLBout,MagNav.INSout,MagNav.FILTout}
    @test typeof(eval_crlb(traj,crlb_P)) <: MagNav.CRLBout
    @test typeof(eval_ins(traj,ins2)) <: MagNav.INSout
    @test typeof(eval_filt(traj,ins2,filt_res)) <: MagNav.FILTout
end

(crlb_out,ins_out,filt_out) = eval_results(traj,ins,filt_res,crlb_P)

p1 = plot();

@testset "plot_filt tests" begin
    @test_nowarn plot_filt!(p1,traj,ins,filt_out;vel_plot=true,show_plot=false);
    @test_nowarn plot_filt(traj,ins,filt_out;vel_plot=true,show_plot=false);
    @test_nowarn plot_filt_err(traj,filt_out,crlb_out;vel_plot=true,show_plot=false);
end

@testset "plot_filt tests" begin
    @test_nowarn plot_mag_map(traj,mag_1_c,itp_mapS  ;order=:magmap);
    @test_nowarn plot_mag_map(traj,mag_1_c,itp_mapS3D;order=:mapmag);
    @test_throws ErrorException plot_mag_map(traj,mag_1_c,itp_mapS;order=:test);
    @test typeof(plot_mag_map_err(traj,mag_1_c,itp_mapS  )) <: Plots.Plot
    @test typeof(plot_mag_map_err(traj,mag_1_c,itp_mapS3D)) <: Plots.Plot
end

@testset "plot_autocor tests" begin
    @test typeof(plot_autocor(mag_1_c-map_val,dt,1)) <: Plots.Plot
end

@testset "chisq tests" begin
    @test MagNav.chisq_pdf(0) ≈ 0
    @test MagNav.chisq_cdf(0) ≈ 0
    @test MagNav.chisq_q(  0) ≈ 0
    @test 0.4 < MagNav.chisq_pdf(0.5) < 0.6
    @test 0.4 < MagNav.chisq_cdf(0.5) < 0.6
    @test 0.4 < MagNav.chisq_q(  0.5) < 0.6
end

P = crlb_P[1:2,1:2,:]

p1 = plot();

ellipse_gif = joinpath(@__DIR__,"conf_ellipse")

@testset "ellipse tests" begin
    ENV["GKSwstype"] = "100"
    @test typeof(MagNav.points_ellipse(P[:,:,1])) <: Tuple{Vector,Vector}
    @test_nowarn MagNav.conf_ellipse!(p1,P[:,:,1])
    @test_nowarn MagNav.conf_ellipse!(p1,P[:,:,1];plot_eigax=true)
    @test_nowarn MagNav.conf_ellipse(P[:,:,1])
    @test typeof(MagNav.units_ellipse(P;conf_units=:deg)) <: Array
    @test typeof(MagNav.units_ellipse(P;conf_units=:rad)) <: Array
    @test typeof(MagNav.units_ellipse(filt_res,filt_out;conf_units=:ft)) <: Array
    @test typeof(MagNav.units_ellipse(filt_res,filt_out;conf_units=:m )) <: Array
    @test_throws ErrorException MagNav.units_ellipse(filt_res,filt_out;conf_units=:test)
    @test typeof(gif_ellipse(P,ellipse_gif)) <: Plots.AnimatedGif
    @test typeof(gif_ellipse(filt_res,filt_out,ellipse_gif)) <: Plots.AnimatedGif
    @test typeof(gif_ellipse(filt_res,filt_out,ellipse_gif,mapS)) <: Plots.AnimatedGif
end

ellipse_gif = MagNav.add_extension(ellipse_gif,".gif")
rm(ellipse_gif)
