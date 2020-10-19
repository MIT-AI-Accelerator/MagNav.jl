using MagNav, Test, MAT

test_file = "test_data_traj.mat"
traj = matopen(test_file,"r") do file
    read(file,"traj")
end

test_file = "test_data_map.mat"
map_data = matopen(test_file,"r") do file
    read(file,"map_data")
end

test_file = "test_data_grid.mat"
grid_data = matopen(test_file,"r") do file
    read(file,"grid_data")
end

lat = traj["lat"][1]
lon = traj["lon"][1]

map_map = map_data["map"]
map_xx  = vec(map_data["xx"])
map_yy  = vec(map_data["yy"])

interp_mapS = gen_interp_map(map_map,map_xx,map_yy)

@testset "Interp Map Tests" begin
    @test interp_mapS(lon,lat) ≈ grid_data["interp_map"]
end
