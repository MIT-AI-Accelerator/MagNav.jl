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
xn  = 0.001*ones(2)
xl  = 0.001*ones(16)
x   = 0.001*ones(18)
nx  = 18

map_map = map_data["map"]
map_xx  = vec(map_data["xx"])
map_yy  = vec(map_data["yy"])

interp_mapS = gen_interp_map(map_map,map_xx,map_yy)

@testset "Interp Map Tests" begin
    @test interp_mapS(lon,lat) ≈ grid_data["interp_map"]
end

@testset "Map Grad Tests" begin
    @test map_grad(interp_mapS,lon,lat) ≈ vec(grid_data["map_grad"])
end

@testset "hRBPF Tests" begin
    @test hRBPF(interp_mapS,xn,xl,deg2rad(lat),deg2rad(lon))[1] ≈
          grid_data["hRBPF"]
end

@testset "H Tests" begin
    @test get_H(interp_mapS,x,deg2rad(lat),deg2rad(lon),nx) ≈ grid_data["H"]
end

@testset "h Tests" begin
    @test get_h(interp_mapS,x,deg2rad(lat),deg2rad(lon)) ≈ grid_data["h"]
end
