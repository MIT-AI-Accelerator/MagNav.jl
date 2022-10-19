using MagNav, Test, MAT

test_file = joinpath(@__DIR__,"test_data/test_data_map.mat")
map_data  = matopen(test_file,"r") do file
    read(file,"map_data")
end

traj_file = joinpath(@__DIR__,"test_data/test_data_traj.mat")
traj = get_traj(traj_file,:traj,silent=true)

map_map = map_data["map"]
nx = size(map_map,2)
ny = size(map_map,1)
dx = map_data["dx"]
dy = map_data["dy"]
dz = map_data["dz"]

(k,kx,ky) = create_k(dx,dy,nx,ny)

mapS = get_map(MagNav.namad)
mapS = map_trim(mapS,traj)

mapV = get_map(MagNav.emm720)
mapV = map_trim(mapV,traj)

@testset "upward_fft tests" begin
    @test upward_fft(map_map,dx,dy,dz;expand=false) ≈ map_data["map_out"]
    @test_nowarn upward_fft(map_map,dx,dy,dz;expand=true)
    @test_nowarn upward_fft(mapS,mapS.alt+dz;expand=false)
    @test_nowarn upward_fft(mapS,mapS.alt+dz;expand=true)
    @test_nowarn upward_fft(mapV,mapV.alt+dz;expand=false)
    @test_nowarn upward_fft(mapV,mapV.alt+dz;expand=true)
    @test upward_fft(mapS,mapS.alt-dz).map ≈ mapS.map
end

@testset "vector_fft tests" begin
    @test_nowarn vector_fft(map_map,dx,dy,0.25*one.(map_map),zero(map_map))
end

@testset "create_k tests" begin
    @test k  ≈ map_data["k"]
    @test kx ≈ map_data["kx"]
    @test ky ≈ map_data["ky"]
end

@testset "downward_L tests" begin
    @test_nowarn downward_L(mapS,mapS.alt-dz,[1,10,100];expand=false)
    @test_nowarn downward_L(mapS,mapS.alt-dz,[1,10,100];expand=true)
end

@testset "psd tests" begin
    @test_nowarn psd(map_map,dx,dy)
end
