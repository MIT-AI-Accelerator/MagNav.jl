using MagNav, Test, MAT
using MagNav: MapS, MapSd, MapS3D, MapV

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

(k,kx,ky) = MagNav.create_k(dx,dy,nx,ny)

mapS   = get_map(MagNav.emag2)
mapS   = map_trim(mapS,traj)
mapS3D = upward_fft(mapS,[mapS.alt,mapS.alt+dz])

mapV = get_map(MagNav.emm720)
mapV = map_trim(mapV,traj)

@testset "upward_fft tests" begin
    @test upward_fft(map_map,dx,dy,dz;expand=false) ≈ map_data["map_out"]
    @test upward_fft(map_map,dx,dy,dz;expand=true) isa Matrix
    @test upward_fft(mapS,mapS.alt+dz;expand=false) isa MapS
    @test upward_fft(mapS,mapS.alt+dz;expand=true) isa MapS
    @test upward_fft(mapS,[mapS.alt,mapS.alt+dz]) isa MapS3D
    @test upward_fft(mapS,[mapS.alt,mapS.alt-dz];α=200) isa MapS3D
    @test upward_fft(mapS,[mapS.alt-1,mapS.alt,mapS.alt+1];α=200) isa MapS3D
    @test upward_fft(mapS3D,mapS.alt+2*dz) isa MapS3D
    @test upward_fft(mapV,mapV.alt+dz;expand=false) isa MapV
    @test upward_fft(mapV,mapV.alt+dz;expand=true) isa MapV
    @test upward_fft(mapS,mapS.alt-dz).map ≈ mapS.map
end

@testset "vector_fft tests" begin
    @test vector_fft(map_map,dx,dy,0.25*one.(map_map),zero(map_map)) isa NTuple{3,Matrix}
end

@testset "create_k tests" begin
    @test k  ≈ map_data["k"]
    @test kx ≈ map_data["kx"]
    @test ky ≈ map_data["ky"]
end

@testset "downward_L tests" begin
    @test MagNav.downward_L(mapS,mapS.alt-dz,[1,10,100];expand=false) isa Vector
    @test MagNav.downward_L(mapS,mapS.alt-dz,[1,10,100];expand=true ) isa Vector
end

@testset "psd tests" begin
    @test MagNav.psd(map_map,dx,dy) isa NTuple{3,Matrix}
    @test MagNav.psd(mapS) isa NTuple{3,Matrix}
end
