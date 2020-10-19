using MagNav, Test, MAT

test_file = "test_data_TL.mat"
TL_data = matopen(test_file,"r") do file
    read(file,"TL_data")
end

test_file = "test_data_traj.mat"
traj = matopen(test_file,"r") do file
    read(file,"traj")
end

test_file = "test_data_map.mat"
map_data = matopen(test_file,"r") do file
    read(file,"map_data")
end

mag_cor   = TL_data["mag_cor"]
mag_cor_d = TL_data["mag_cor_d"]

lat = traj["lat"][1]
lon = traj["lon"][1]

map_map = map_data["map"]
map_xx  = vec(map_data["xx"])
map_yy  = vec(map_data["yy"])

interp_mapS = gen_interp_map(map_map,map_xx,map_yy)

@testset "Detrend Tests" begin
    @test detrend(mag_cor) ≈ mag_cor_d
    @test detrend([3,6,8]) ≈ [-1,2,-1] / 6
end

@testset "Map Grad Tests" begin
    @test map_grad(interp_mapS,lon,lat) ≈ [646.2163302216359,13720.07531468666]
end
