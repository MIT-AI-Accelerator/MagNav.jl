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
    read(file,"traj_data")
end

xn  = vec(grid_data["xn"])
xl  = vec(grid_data["xl"])
x   = vec(grid_data["x"])

map_map = map_data["map"]
map_xx  = deg2rad.(vec(map_data["xx"]))
map_yy  = deg2rad.(vec(map_data["yy"]))

lat = deg2rad(traj_data["lat"][1])
lon = deg2rad(traj_data["lon"][1])
alt = traj_data["alt"][1]

# using linear map interpolation to match MATLAB as well as possible
itp_mapS = map_interpolate(map_map,map_xx,map_yy,:linear)

@testset "Interp Map Tests" begin
    @test itp_mapS(lon,lat) ≈ grid_data["itp_map"]
end

@testset "Map Grad Tests" begin
    @test map_grad(itp_mapS,lat,lon) ≈ reverse(vec(grid_data["grad"]))
end

@testset "h Tests" begin
    @test get_h(itp_mapS,x,lat,lon,alt;core=false)[1] ≈ grid_data["h"]
    @test get_h(itp_mapS,[xn;0;xl[2:end]],lat,lon,alt;
                core=false)[1] ≈ grid_data["hRBPF"]
end

@testset "H Tests" begin
    @test get_H(itp_mapS,x,lat,lon,alt;core=false) ≈ grid_data["H"]
end
