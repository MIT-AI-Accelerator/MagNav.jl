using MagNav, Test, MAT, DataFrames, Geodesy, Plots

test_file = "test_data/test_data_grid.mat"
grid_data = matopen(test_file,"r") do file
    read(file,"grid_data")
end

map_file  = "test_data/test_data_map.mat"
mapS      = get_map(map_file)
itp_mapS  = map_interpolate(mapS,:linear) # linear to match MATLAB

traj_file = "test_data/test_data_traj.mat"
traj      = get_traj(traj_file,:traj,silent=true)

gxf_file = "test_data/HighAlt_Mag.gxf"
(map_map,map_xx,map_yy) = map_get_gxf(gxf_file)
mapP = get_map(string(MagNav.ottawa_area_maps(),"/HighAlt_5181.h5"))

mapV = get_map(MagNav.emm720)
mapV = map_trim(mapV,traj)

mapSd = MagNav.MapSd(mapS.map,mapS.xx,mapS.yy,mapS.alt*one.(mapS.map))
mapVd = MagNav.MapVd(mapV.mapX,mapV.mapY,mapV.mapZ,mapV.xx,mapV.yy,mapV.alt*one.(mapV.mapX))

@testset "map_interpolate tests" begin
    @test itp_mapS(traj.lon[1],traj.lat[1]) ≈ grid_data["itp_map"]
    @test_nowarn map_interpolate(mapS,:quad)
    @test_nowarn map_interpolate(mapS,:cubic)
    @test_throws ErrorException map_interpolate(mapS,:test)
    @test_nowarn map_interpolate(mapV,:X)
    @test_nowarn map_interpolate(mapV,:Y)
    @test_nowarn map_interpolate(mapV,:Z)
    @test_throws ErrorException map_interpolate(mapV,:test)
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
    @test map_trim(mapS ).map  ≈ mapS.map
    @test map_trim(mapSd).map  ≈ mapSd.map
    @test map_trim(mapV ).mapX ≈ mapV.mapX
    @test map_trim(mapVd).mapX ≈ mapVd.mapX
end

@testset "map_correct_igrf! tests" begin
    @test_logs (:info,) map_correct_igrf!(deepcopy(mapS );add_igrf_date=2013+293/365)
    @test_logs (:info,) map_correct_igrf!(deepcopy(mapSd);add_igrf_date=2013+293/365)
end

@testset "map_chessboard tests" begin
    mapSd.alt[1,1] = mapS.alt+200
    mapSd.alt[2,2] = mapS.alt-601
    @test typeof(map_chessboard(mapSd,mapS.alt;dz=200)) <: MagNav.MapS
    @test typeof(map_chessboard(mapSd,mapS.alt;down_cont=false,dz=200)) <: MagNav.MapS
end

lla2utm  = UTMZfromLLA(WGS84)
utm_temp = lla2utm.(LLA.(rad2deg.(mapS.yy),rad2deg.(mapS.xx),mapS.alt))
mapUTM   = MagNav.MapS( mapS.map,[utm_temp[i].x for i = 1:length(mapS.xx)],
                        [utm_temp[i].y for i = 1:length(mapS.yy)],mapS.alt)
mapUTMd  = MagNav.MapSd(mapS.map,[utm_temp[i].x for i = 1:length(mapS.xx)],
                        [utm_temp[i].y for i = 1:length(mapS.yy)],mapSd.alt)

@testset "map_utm2lla! tests" begin
    @test typeof(map_utm2lla!(mapUTM )) <: MagNav.MapS
    @test typeof(map_utm2lla!(mapUTMd)) <: MagNav.MapSd
end

@testset "map_gxf2h5 tests" begin
    @test typeof(map_gxf2h5(gxf_file,5181;get_lla=true ,save_h5=false)) <: MagNav.MapS
    @test typeof(map_gxf2h5(gxf_file,5181;get_lla=false,save_h5=false)) <: MagNav.MapS
    @test typeof(map_gxf2h5(gxf_file,gxf_file,5181;
                            up_cont=false,get_lla=true ,save_h5=false)) <: MagNav.MapSd
    @test typeof(map_gxf2h5(gxf_file,gxf_file,5181;
                            up_cont=false,get_lla=false,save_h5=false)) <: MagNav.MapSd
    @test typeof(map_gxf2h5(gxf_file,gxf_file,120;
                            up_cont=true ,get_lla=true ,save_h5=false)) <: MagNav.MapS
    @test typeof(map_gxf2h5(gxf_file,gxf_file,120;
                            up_cont=true ,get_lla=false,save_h5=false)) <: MagNav.MapS
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
    @test typeof(plot_path(traj;fewer_pts=true,show_plot=false,Nmax=50)) <: Plots.Plot
end

p1 = plot_basic(traj.tt,traj.lat);
df_event = DataFrame(flight=:test,t=49.5,event="test")

@testset "plot_events! tests" begin
    @test_nowarn plot_events!(p1,df_event.t[1],df_event.event[1];t_units=:min)
    @test_nowarn plot_events!(p1,df_event.flight[1],df_event)
end

@testset "map_check tests" begin
    @test map_check(mapS,traj) ≈ true
    @test map_check(mapV,traj) ≈ true
    @test all(map_check([mapS,mapV],traj)) ≈ true
end
