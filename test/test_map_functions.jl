using MagNav, Test, MAT, DataFrames, Geodesy, Plots
using MagNav: MapS, MapSd, MapS3D, MapV

test_file = joinpath(@__DIR__,"test_data/test_data_grid.mat")
grid_data = matopen(test_file,"r") do file
    read(file,"grid_data")
end

map_file  = joinpath(@__DIR__,"test_data/test_data_map.mat")
mapS      = get_map(map_file)
itp_mapS  = map_interpolate(mapS,:linear) # linear to match MATLAB

traj_file = joinpath(@__DIR__,"test_data/test_data_traj.mat")
traj      = get_traj(traj_file,:traj,silent=true)

gxf_file  = MagNav.ottawa_area_maps_gxf()*"/HighAlt_Mag.gxf"
(map_map,map_xx,map_yy) = map_get_gxf(gxf_file)
mapP = get_map(MagNav.ottawa_area_maps()*"/HighAlt_5181.h5")

mapV = get_map(MagNav.emm720)
mapV = map_trim(mapV,traj)

mapSd  = MapSd(mapS.map,mapS.xx,mapS.yy,mapS.alt*one.(mapS.map))
mapS3D = upward_fft(mapS,[mapS.alt,mapS.alt+5])

@testset "map_interpolate tests" begin
    @test itp_mapS(traj.lon[1],traj.lat[1]) ≈ grid_data["itp_map"]
    @test_nowarn map_interpolate(mapS  ,:linear)
    @test_nowarn map_interpolate(mapSd ,:quad)
    @test_nowarn map_interpolate(mapS3D,:cubic)
    @test_throws ErrorException map_interpolate(mapS,:test)
    @test_nowarn map_interpolate(mapV,:X,:linear)
    @test_nowarn map_interpolate(mapV,:Y,:quad)
    @test_nowarn map_interpolate(mapV,:Z,:cubic)
    @test_throws ErrorException map_interpolate(mapV,:test)
    @test_throws ErrorException map_interpolate(mapV,:X,:test)
    typeof(mapS3D(mapS3D.alt[1]  -1)) <: MapS
    typeof(mapS3D(mapS3D.alt[end]+1)) <: MapS
    @test mapS3D(mapS3D.alt[1]).map ≈ mapS.map
end

@testset "map_get_gxf tests" begin
    @test_nowarn map_get_gxf(gxf_file)
end

@testset "map_params tests" begin
    @test_nowarn MagNav.map_params(mapS)
    @test_nowarn MagNav.map_params(mapV)
end

@testset "map_lla_lim tests" begin
    @test_nowarn MagNav.map_lla_lim(map_xx,map_yy,mapP.alt,
                                    1,length(map_xx),1,length(map_yy))
end

@testset "map_trim tests" begin
    @test map_trim(map_map,map_xx,map_yy,mapP.alt) == (68:91,52:75)
    @test_throws ErrorException map_trim(map_map,map_xx,map_yy,mapP.alt;
                                         map_units=:test)
    @test map_trim(mapS  ).map  ≈ mapS.map
    @test map_trim(mapSd ).map  ≈ mapSd.map
    @test map_trim(mapS3D).map  ≈ mapS3D.map
    @test map_trim(mapV  ).mapX ≈ mapV.mapX
end

add_igrf_date = get_years(2013,293)

@testset "map_correct_igrf tests" begin
    @test map_correct_igrf(mapS.map,mapS.alt,mapS.xx,mapS.yy;
                           add_igrf_date=add_igrf_date,map_units=:rad) == 
          map_correct_igrf(mapS.map,mapS.alt,rad2deg.(mapS.xx),rad2deg.(mapS.yy);
                           add_igrf_date=add_igrf_date,map_units=:deg)
    @test_throws ErrorException map_correct_igrf(mapS.map,mapS.alt,mapS.xx,mapS.yy;
                                                 add_igrf_date=add_igrf_date,map_units=:test)
    @test typeof(map_correct_igrf(mapS ;add_igrf_date=add_igrf_date)) <: MapS
    @test typeof(map_correct_igrf(mapSd;add_igrf_date=add_igrf_date)) <: MapSd
end

@testset "map_fill tests" begin
    @test typeof(map_fill(mapS.map,mapS.xx,mapS.yy)) <: Matrix
    @test typeof(map_fill(mapS  )) <: MapS
    @test typeof(map_fill(mapSd )) <: MapSd
    @test typeof(map_fill(mapS3D)) <: MapS3D
end

@testset "map_chessboard tests" begin
    mapSd.alt[1,1] = mapS.alt+200
    mapSd.alt[2,2] = mapS.alt-601
    @test typeof(map_chessboard(mapSd,mapS.alt;dz=200)) <: MapS
    @test typeof(map_chessboard(mapSd,mapS.alt;down_cont=false,dz=200)) <: MapS
end

lla2utm  = UTMZfromLLA(WGS84)
utm_temp = lla2utm.(LLA.(rad2deg.(mapS.yy),rad2deg.(mapS.xx),mapS.alt))
mapUTM   = MapS( mapS.map,[utm_temp[i].x for i in eachindex(mapS.xx)],
                          [utm_temp[i].y for i in eachindex(mapS.yy)],mapS.alt)
mapUTMd  = MapSd(mapS.map,[utm_temp[i].x for i in eachindex(mapS.xx)],
                          [utm_temp[i].y for i in eachindex(mapS.yy)],mapSd.alt)

@testset "map_utm2lla tests" begin
    @test typeof(map_utm2lla(mapUTM.map,mapUTM.xx,mapUTM.yy,mapUTM.alt)[1]) <: Matrix
    @test typeof(map_utm2lla(mapUTM )) <: MapS
    @test typeof(map_utm2lla(mapUTMd)) <: MapSd
end

@testset "map_gxf2h5 tests" begin
    @test typeof(map_gxf2h5(gxf_file,5181;get_lla=true ,save_h5=false)) <: MapS
    @test typeof(map_gxf2h5(gxf_file,5181;get_lla=false,save_h5=false)) <: MapS
    @test typeof(map_gxf2h5(gxf_file,gxf_file,5181;
                            up_cont=false,get_lla=true ,save_h5=false)) <: MapSd
    @test typeof(map_gxf2h5(gxf_file,gxf_file,5181;
                            up_cont=false,get_lla=false,save_h5=false)) <: MapSd
    @test typeof(map_gxf2h5(gxf_file,gxf_file,120;
                            up_cont=true ,get_lla=true ,save_h5=false)) <: MapS
    @test typeof(map_gxf2h5(gxf_file,gxf_file,120;
                            up_cont=true ,get_lla=false,save_h5=false)) <: MapS
end

p1 = plot();

@testset "plot_map tests" begin
    @test typeof(plot_map!(p1,mapS)) <: Plots.Plot
    @test typeof(plot_map(mapS;plot_units=:deg)) <: Plots.Plot
    @test typeof(plot_map(mapS;plot_units=:rad)) <: Plots.Plot
    @test typeof(plot_map(mapS;plot_units=:m  )) <: Plots.Plot
    @test typeof(plot_map(mapS.map,rad2deg.(mapS.xx),rad2deg.(mapS.yy);
                          map_units=:deg,plot_units=:deg)) <: Plots.Plot
    @test typeof(plot_map(mapS.map,rad2deg.(mapS.xx),rad2deg.(mapS.yy);
                          map_units=:deg,plot_units=:rad)) <: Plots.Plot
    @test typeof(plot_map(mapS.map,rad2deg.(mapS.xx),rad2deg.(mapS.yy);
                          map_units=:deg,plot_units=:m  )) <: Plots.Plot
    @test_throws ErrorException plot_map(mapS;map_units=:test)
    @test_throws ErrorException plot_map(mapS;plot_units=:test)
end

@testset "map_cs tests" begin
    for map_color in [:usgs,:gray,:gray1,:gray2,:plasma,:magma]
        @test typeof(MagNav.map_cs(map_color)) <: Plots.ColorGradient
    end
end

p1 = plot_path(traj;show_plot=false);

@testset "plot_path tests" begin
    @test_nowarn plot_path!(p1,traj;show_plot=false,path_color=:black)
    @test typeof(plot_path(traj;Nmax=50,show_plot=false)) <: Plots.Plot
end

p1 = plot_basic(traj.tt,traj.lat);
df_event = DataFrame(flight=:test,tt=49.5*60,event="test")

@testset "plot_events! tests" begin
    @test_nowarn plot_events!(p1,df_event.tt[1]/60,df_event.event[1])
    @test_nowarn plot_events!(p1,df_event,df_event.flight[1])
end

@testset "map_check tests" begin
    @test all(map_check([mapS,mapSd,mapS3D,mapV],traj)) == true
end

@testset "get_map_val tests" begin
    @test [get_map_val(mapS  ,traj.lat[1],traj.lon[1],traj.alt[1]),
           get_map_val(mapSd ,traj.lat[1],traj.lon[1],traj.alt[1]),
           get_map_val(mapS3D,traj.lat[1],traj.lon[1],traj.alt[1])] == 
           get_map_val([mapS,mapSd,mapS3D],traj,1)
end

@testset "map_border tests" begin
    @test_nowarn map_border(map_map,map_xx,map_yy;inner=true,sort_border=true)
    @test_nowarn map_border(mapS  ;inner=true ,sort_border=true)
    @test_nowarn map_border(mapSd ;inner=true ,sort_border=false)
    @test_nowarn map_border(mapS3D;inner=false,sort_border=false)
end

ind = [1,100]

@testset "map_resample tests" begin
    @test map_resample(mapS,mapS.xx[ind],mapS.yy[ind]).map ≈ mapS.map[ind,ind]
    @test_nowarn map_resample(mapS,mapS)
end

xx_lim = extrema(mapS.xx) .+ (-0.01,0.01)
yy_lim = extrema(mapS.yy) .+ (-0.01,0.01)
mapS_fallback = upward_fft(map_fill(map_trim(get_map(),
                xx_lim=xx_lim,yy_lim=yy_lim)),mapS.alt)

@testset "map_combine tests" begin
    @test typeof(map_combine(mapS,mapS_fallback)) <: MapS
    @test typeof(map_combine([mapS,upward_fft(mapS,mapS.alt+5)],mapS_fallback)) <: MapS3D
end
