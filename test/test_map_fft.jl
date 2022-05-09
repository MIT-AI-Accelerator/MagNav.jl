using MagNav, Test, MAT

test_file = "test_data/test_data_map.mat"
map_data  = matopen(test_file,"r") do file
    read(file,"map_data")
end

map_map = map_data["map"]
nx = size(map_map,2)
ny = size(map_map,1)
dx = map_data["dx"]
dy = map_data["dy"]
dz = map_data["dz"]

(k,kx,ky) = create_k(dx,dy,nx,ny)

@testset "Upward FFT Tests" begin
    @test upward_fft(map_map,dx,dy,dz;expand=false) ≈ map_data["map_out"]
end

@testset "Create K Tests" begin
    @test k  ≈ map_data["k"]
    @test kx ≈ map_data["kx"]
    @test ky ≈ map_data["ky"]
end
