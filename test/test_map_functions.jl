using MagNav, Test, MAT

test_file = "test_data/test_data_grid.mat"
grid_data = matopen(test_file,"r") do file
    read(file,"grid_data")
end

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

lat = deg2rad(traj_data["lat"][1])
lon = deg2rad(traj_data["lon"][1])

itp_mapS = map_interpolate(map_map,map_xx,map_yy,:linear) # linear to match MATLAB

@testset "itp_mapS tests" begin
    @test itp_mapS(lon,lat) ≈ grid_data["itp_map"]
end
