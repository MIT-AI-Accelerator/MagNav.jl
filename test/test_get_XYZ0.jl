using MagNav, Test, MAT

traj_file = "test_data/test_data_traj.mat"
ins_file  = "test_data/test_data_ins.mat"

xyz    = get_XYZ0(traj_file,:traj,:none;silent=true)
traj   = xyz.traj
ins    = xyz.ins
flux_a = xyz.flux_a

ind = trues(length(xyz.traj.tt))
ind[51:end] .= false

@testset "get_XYZ0 tests" begin
    @test_nowarn get_XYZ0(traj_file,:traj,:none,
                          flight = 1,
                          line   = 1,
                          dt     = 1,
                          silent = true)
end

@testset "get_traj tests" begin
    @test_nowarn get_traj(traj_file,:traj;silent=true)
    @test_nowarn get_traj(xyz,ind)
    @test_nowarn traj(ind)
end

@testset "get_ins tests" begin
    @test_nowarn get_ins(ins_file,:ins_data;silent=true)
    @test_nowarn get_ins(xyz,ind)
    @test_nowarn ins(ind)
    @test_nowarn MagNav.zero_ins_ll(ins.lat,ins.lon,1,
                                    traj.lat[1:1],traj.lon[1:1])
end

@testset "get_flux tests" begin
    @test_nowarn flux_a(ind)
end
