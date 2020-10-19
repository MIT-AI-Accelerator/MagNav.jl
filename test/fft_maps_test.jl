using MagNav, Test, MAT

test_file = "test_data_map.mat"
map_data = matopen(test_file,"r") do file
    read(file,"map_data")
end

map     = map_data["map"]
d_east  = map_data["east_spacing"]
d_north = map_data["north_spacing"]
d_alt   = map_data["alt_uc"]
map_uc  = map_data["map_uc"]

k  = map_data["k"]
kx = map_data["kx"]
ky = map_data["ky"]
Nx = size(map,2)
Ny = size(map,1)

@testset "Upward FFT Tests" begin
    @test upward_fft(map,d_east,d_north,d_alt) ≈ map_uc
end

(k_out,kx_out,ky_out) = create_K(d_east,d_north,Nx,Ny)

@testset "Create K Tests" begin
    @test k_out  ≈ k
    @test kx_out ≈ kx
    @test ky_out ≈ ky
end
