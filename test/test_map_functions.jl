using MagNav, Test, MAT
using DataFrames, LinearAlgebra, Plots, Statistics
using MagNav: MapS, MapSd, MapS3D, MapV
using Geodesy: LLA, LLAfromUTM, UTM, UTMfromLLA, WGS84, utm_zone

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

mapSd  = MapSd(mapS.info,mapS.map,mapS.xx,mapS.yy,mapS.alt*one.(mapS.map),mapS.mask)
mapS3D = upward_fft(mapS,[mapS.alt,mapS.alt+5])
mapS_  = MapS("Map",[0.0 1; 1 1],map_xx[1:2],map_yy[1:2],mapS.alt,trues(2,2))

@testset "map_interpolate tests" begin
    @test itp_mapS(traj.lat[1],traj.lon[1]) ≈ grid_data["itp_map"]
    @test map_interpolate(mapS  ,:linear) isa Function
    @test map_interpolate(mapSd ,:quad  ) isa Function
    @test map_interpolate(mapS3D,:cubic ) isa Function
    @test_throws ErrorException map_interpolate(mapS,:test)
    @test map_interpolate(mapV,:X,:linear) isa Function
    @test map_interpolate(mapV,:Y,:quad  ) isa Function
    @test map_interpolate(mapV,:Z,:cubic ) isa Function
    @test_throws ErrorException map_interpolate(mapV,:test)
    @test_throws ErrorException map_interpolate(mapV,:X,:test)
    mapS3D(mapS3D.alt[1]  -1) isa MapS
    mapS3D(mapS3D.alt[end]+1) isa MapS
    @test mapS3D(mapS3D.alt[1]).map ≈ mapS.map
end

@testset "map_get_gxf tests" begin
    @test map_get_gxf(gxf_file) isa Tuple{Matrix,Vector,Vector}
end

@testset "map_params tests" begin
    @test MagNav.map_params(mapS) isa Tuple{BitMatrix,BitMatrix,Int,Int}
    @test MagNav.map_params(mapV) isa Tuple{BitMatrix,BitMatrix,Int,Int}
end

@testset "map_lla_lim tests" begin
    @test MagNav.map_lla_lim(map_xx,map_yy) isa NTuple{2,Vector}
end

@testset "map_trim tests" begin
    @test map_trim(map_map,map_xx,map_yy) == (68:91,52:75)
    @test_throws ErrorException map_trim(map_map,map_xx,map_yy;map_units=:test)
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
    @test map_correct_igrf(mapS  ;add_igrf_date=add_igrf_date) isa MapS
    @test map_correct_igrf(mapSd ;add_igrf_date=add_igrf_date) isa MapSd
    @test map_correct_igrf(mapS3D;add_igrf_date=add_igrf_date) isa MapS3D
end

@testset "map_fill tests" begin
    @test map_fill(mapS.map,mapS.xx,mapS.yy) isa Matrix
    @test map_fill(mapS  ) isa MapS
    @test map_fill(mapSd ) isa MapSd
    @test map_fill(mapS3D) isa MapS3D
end

@testset "map_chessboard tests" begin
    mapSd.alt[1,1] = mapS.alt+200
    mapSd.alt[2,2] = mapS.alt-601
    @test map_chessboard(mapSd,mapS.alt;dz=200) isa MapS
    @test map_chessboard(mapSd,mapS.alt;down_cont=false,dz=200) isa MapS
end

(zone_utm,is_north) = utm_zone(mean(rad2deg.(mapS.yy)),mean(rad2deg.(mapS.xx)))
lla2utm   = UTMfromLLA(zone_utm,is_north,WGS84)
utm_temp  = lla2utm.(LLA.(rad2deg.(mapS.yy),rad2deg.(mapS.xx)))
mapUTM_xx = [utm_temp[i].x for i in eachindex(mapS.xx)]
mapUTM_yy = [utm_temp[i].y for i in eachindex(mapS.yy)]
mapUTM    = MapS( mapS.info,mapS.map,mapUTM_xx,mapUTM_yy, mapS.alt,mapS.mask)
mapUTMd   = MapSd(mapS.info,mapS.map,mapUTM_xx,mapUTM_yy,mapSd.alt,mapS.mask)
mapUTM3D  = MapS3D(mapUTM.info,mapUTM.map[:,:,[1,1]],mapUTM_xx,mapUTM_yy,
                   [mapUTM.alt,mapUTM.alt+5],mapUTM.mask[:,:,[1,1]])

@testset "map_utm2lla tests" begin
    @test map_utm2lla(mapUTM.map,mapUTM.xx,mapUTM.yy,mapUTM.alt,mapUTM.mask)[1] isa Matrix
    @test map_utm2lla(mapUTM  ) isa MapS
    @test map_utm2lla(mapUTMd ) isa MapSd
    @test map_utm2lla(mapUTM3D) isa MapS3D
    @test map_utm2lla(mapS_.map,mapUTM.xx[1:2],mapUTM.yy[1:2],mapS_.alt,mapS_.mask)[1] isa Matrix
end

map_h5 = joinpath(@__DIR__,"test_map_functions.h5")

@testset "map_gxf2h5 tests" begin
    @test map_gxf2h5(gxf_file,5181;get_lla=true ,save_h5=false) isa MapS
    @test map_gxf2h5(gxf_file,5181;get_lla=false,save_h5=true,map_h5=map_h5) isa MapS
    @test map_gxf2h5(gxf_file,gxf_file,5181;
                     up_cont=false,get_lla=true ,save_h5=false) isa MapSd
    @test map_gxf2h5(gxf_file,gxf_file,5181;
                     up_cont=false,get_lla=false,save_h5=false) isa MapSd
    @test map_gxf2h5(gxf_file,gxf_file,120;
                     up_cont=true ,get_lla=true ,save_h5=false) isa MapS
    @test map_gxf2h5(gxf_file,gxf_file,120;
                     up_cont=true ,get_lla=false,save_h5=false) isa MapS
    @test map_gxf2h5(gxf_file,gxf_file,-1;
                     up_cont=true ,get_lla=false,save_h5=true,map_h5=map_h5) isa MapS
end

rm(map_h5)

p1 = plot()

@testset "plot_map tests" begin
    @test plot_map!(p1,mapS) isa Nothing
    @test plot_map(mapS;plot_units=:deg) isa Plots.Plot
    @test plot_map(mapS;plot_units=:rad) isa Plots.Plot
    @test plot_map(mapS;plot_units=:m  ) isa Plots.Plot
    @test plot_map(mapS.map,rad2deg.(mapS.xx),rad2deg.(mapS.yy);
                   map_units=:deg,plot_units=:deg) isa Plots.Plot
    @test plot_map(mapS.map,rad2deg.(mapS.xx),rad2deg.(mapS.yy);
                   map_units=:deg,plot_units=:rad) isa Plots.Plot
    @test plot_map(mapS.map,rad2deg.(mapS.xx),rad2deg.(mapS.yy);
                   map_units=:deg,plot_units=:m  ) isa Plots.Plot
    @test plot_map(mapV) isa Tuple{Plots.Plot,Plots.Plot,Plots.Plot}
    @test_throws ErrorException plot_map(mapS;map_units=:test)
    @test_throws ErrorException plot_map(mapS;plot_units=:test)
end

@testset "map_cs tests" begin
    for map_color in [:usgs,:gray,:gray1,:gray2,:plasma,:magma]
        @test MagNav.map_cs(map_color) isa Plots.ColorGradient
    end
end

show_plot = false
p1 = plot_path(traj;show_plot)

@testset "plot_path tests" begin
    @test plot_path!(p1,traj;show_plot,path_color=:black) isa Nothing
    @test plot_path(traj;Nmax=50,show_plot) isa Plots.Plot
end

p1 = plot_basic(traj.tt,traj.lat;show_plot)
df_event = DataFrame(flight=:test,tt=49.5,event="test")

@testset "plot_events! tests" begin
    @test plot_events!(p1,df_event.tt[1]/60,df_event.event[1]) isa Nothing
    @test plot_events!(p1,df_event.flight[1],df_event;t_units=:min) isa Nothing
    @test plot_events( p1,df_event.flight[1],df_event;t_units=:min) isa Plots.Plot
end

@testset "map_check tests" begin
    @test all(map_check([mapS,mapSd,mapS3D,mapV],traj)) == true
end

@testset "get_map_val tests" begin
    @test [get_map_val(mapS  ,traj.lat[1],traj.lon[1],traj.alt[1]),
           get_map_val(mapSd ,traj.lat[1],traj.lon[1],traj.alt[1]),
           get_map_val(mapS3D,traj.lat[1],traj.lon[1],traj.alt[1])] ==
           get_map_val([mapS,mapSd,mapS3D],traj,1)
    @test get_map_val(mapS3D,traj.lat[1],traj.lon[1],mapS3D.alt[1]-1) isa Real
    @test get_map_val(mapS3D,traj.lat[1],traj.lon[1],mapS3D.alt[end]+1) isa Real
end

# create map_cache (2 maps at 305 & 915 m, trimmed NAMAD for speed)
map_305   = mapS
map_915   = upward_fft(map_305,3*map_305.alt)
xx_lim    = MagNav.get_lim(map_305.xx,0.2)
yy_lim    = MagNav.get_lim(map_305.yy,0.2)
namad     = map_trim(get_map();xx_lim=xx_lim,yy_lim=yy_lim)
map_cache = Map_Cache(maps=[map_305,map_915],fallback=namad)

# test lat & lon point
lat = mean(map_305.yy)
lon = mean(map_305.xx)

# test alt points at 315, 610, & 965 m
alt_315 =   map_305.alt + 10
alt_610 = 2*map_305.alt
alt_965 =   map_915.alt + 50

# determine alt buckets for test points
alt_bucket_300_305 = max(floor(alt_315/100)*100, map_305.alt)
alt_bucket_600_305 = max(floor(alt_610/100)*100, map_305.alt)
alt_bucket_900_915 = max(floor(alt_965/100)*100, map_915.alt)

# upward continue to alt buckets & interpolate
mapS_300_305     = upward_fft(map_fill(map_trim(map_305)), alt_bucket_300_305)
mapS_600_305     = upward_fft(map_fill(map_trim(map_305)), alt_bucket_600_305)
mapS_900_915     = upward_fft(map_fill(map_trim(map_915)), alt_bucket_900_915)
itp_mapS_300_305 = map_interpolate(mapS_300_305)
itp_mapS_600_305 = map_interpolate(mapS_600_305)
itp_mapS_900_915 = map_interpolate(mapS_900_915)

# test alt, lat, & lon point outside of maps for NAMAD fallback testing
alt_out = map_305.alt - 5
lat_out = MagNav.get_lim(map_305.yy,0.1)[1]
lon_out = MagNav.get_lim(map_305.xx,0.1)[1]

# NAMAD maps for NAMAD fallback testing
alt_bucket_namad_1 = max(floor(alt_out/100)*100, namad.alt)
alt_bucket_namad_2 = max(floor(alt_315/100)*100, namad.alt)
mapS_namad_1       = upward_fft(map_fill(map_trim(namad)), alt_bucket_namad_1)
mapS_namad_2       = upward_fft(map_fill(map_trim(namad)), alt_bucket_namad_2)
itp_mapS_namad_1   = map_interpolate(mapS_namad_1)
itp_mapS_namad_2   = map_interpolate(mapS_namad_2)

# test that same alt bucket maps are used as within map_cache
@testset "Map_Cache tests" begin
    @test map_cache(lat,lon,alt_315) ≈ itp_mapS_300_305(lat,lon) # using map at 305 m (> 300 m)
    @test map_cache(lat,lon,alt_610) ≈ itp_mapS_600_305(lat,lon) # using map at 600 m (> 305 m)
    @test map_cache(lat,lon,alt_965) ≈ itp_mapS_900_915(lat,lon) # using map at 915 m (> 900 m)
    @test map_cache(lat,lon,alt_out) ≈ itp_mapS_namad_1(lat,lon) # NAMAD fallback, alt < all maps alts
    @test map_cache(lat_out,lon_out,alt_315) ≈ itp_mapS_namad_2(lat_out,lon_out) # NAMAD fallback, failed map_check()
end

@testset "map_border tests" begin
    @test map_border(map_map,map_xx,map_yy;inner=true,sort_border=true) isa NTuple{2,Vector}
    @test map_border(mapS  ;inner=true ,sort_border=true ) isa NTuple{2,Vector}
    @test map_border(mapSd ;inner=true ,sort_border=false) isa NTuple{2,Vector}
    @test map_border(mapS3D;inner=false,sort_border=false) isa NTuple{2,Vector}
    @test map_border(mapS_ ;inner=false,sort_border=false) isa NTuple{2,Vector}
    @test MagNav.map_border_clean(trues(3,3)) == trues(3,3)
    @test MagNav.map_border_sort([1:3;],[1,0,1],1,1) == ([1],[1])
end

ind = [1,100]

@testset "map_resample tests" begin
    @test map_resample(mapS,mapS.xx[ind],mapS.yy[ind]).map ≈ mapS.map[ind,ind]
    @test map_resample(mapS,mapS) isa MapS
end

xx_lim = extrema(mapS.xx) .+ (-0.01,0.01)
yy_lim = extrema(mapS.yy) .+ (-0.01,0.01)
namad  = upward_fft(map_trim(get_map(),xx_lim=xx_lim,yy_lim=yy_lim),mapS.alt)

@testset "map_combine tests" begin
    @test map_combine(mapS,namad) isa MapS
    @test map_combine([mapS,upward_fft(mapS,mapS.alt+5)],namad) isa MapS3D
    @test map_combine([mapS,upward_fft(mapS,mapS.alt+5)];use_fallback=false) isa MapS3D
end
