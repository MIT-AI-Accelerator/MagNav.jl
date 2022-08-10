using MagNav, Test, MAT

test_file = "test_data/test_data_map.mat"
map_data  = matopen(test_file,"r") do file
    read(file,"map_data")
end

test_file = "test_data/test_data_traj.mat"
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

ind = trues(length(tt))
ind[51:end] .= false

map_name  = "test"
path_name = "test"

@testset "map2kmz tests" begin
    @test_nowarn map2kmz(map_map,map_xx,map_yy,map_name)
    @test_nowarn map2kmz(mapS,map_name)
end

@testset "path2kml tests" begin
    @test_nowarn path2kml(lat,lon,alt;path_name=path_name,points=false)
    @test_nowarn path2kml(lat,lon,alt;path_name=path_name,points=true)
    @test_nowarn path2kml(traj;path_name=path_name,points=false)
    @test_nowarn path2kml(traj,ind;path_name=path_name,points=true)
end

rm(map_name*".kmz")
rm(path_name*".kml")
