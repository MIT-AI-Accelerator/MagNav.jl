using MagNav, Test, MAT

test_file = joinpath(@__DIR__,"test_data/test_data_map.mat")
map_data  = matopen(test_file,"r") do file
    read(file,"map_data")
end

test_file = joinpath(@__DIR__,"test_data/test_data_traj.mat")
traj_data = matopen(test_file,"r") do file
    read(file,"traj")
end

map_map = map_data["map"]
map_xx  = deg2rad.(vec(map_data["xx"]))
map_yy  = deg2rad.(vec(map_data["yy"]))
map_alt = map_data["alt"]

tt  = vec(traj_data["tt"])
lat = deg2rad.(vec(traj_data["lat"]))
lon = deg2rad.(vec(traj_data["lon"]))
alt = vec(traj_data["alt"])
vn  = vec(traj_data["vn"])
ve  = vec(traj_data["ve"])
vd  = vec(traj_data["vd"])
fn  = vec(traj_data["fn"])
fe  = vec(traj_data["fe"])
fd  = vec(traj_data["fd"])
Cnb = traj_data["Cnb"]
N   = length(lat)
dt  = tt[2] - tt[1]

traj = MagNav.Traj(N,dt,tt,lat,lon,alt,vn,ve,vd,fn,fe,fd,Cnb)
mapS = MagNav.MapS(map_map,map_xx,map_yy,map_alt)

ind = trues(N)
ind[51:end] .= false

map_map = map_map[ind,ind]
map_xx  = map_xx[ind]
map_yy  = map_yy[ind]

map_kmz  = joinpath(@__DIR__,"test")
path_kml = joinpath(@__DIR__,"test")

@testset "map2kmz tests" begin
    @test_nowarn map2kmz(map_map,map_xx,map_yy,map_kmz)
    @test_nowarn map2kmz(map_map,map_xx,map_yy,map_kmz;map_units=:deg)
    @test_throws ErrorException map2kmz(map_map,map_xx,map_yy,map_kmz;map_units=:test)
    @test_nowarn map2kmz(mapS,map_kmz;plot_alt=mapS.alt)
end

@testset "path2kml tests" begin
    @test_nowarn path2kml(lat,lon,alt,path_kml)
    @test_throws ErrorException path2kml(lat,lon,alt,path_kml;path_units=:test)
    @test_nowarn path2kml(traj(ind),path_kml;points=true)
end

map_kmz  = MagNav.add_extension(map_kmz ,".kmz")
path_kml = MagNav.add_extension(path_kml,".kml")

rm(map_kmz)
rm(path_kml)
