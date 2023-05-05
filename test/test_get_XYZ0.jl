using MagNav, Test, MAT
using MagNav: delete_field, write_field

traj_file = joinpath(@__DIR__,"test_data/test_data_traj.mat")
ins_file  = joinpath(@__DIR__,"test_data/test_data_ins.mat")

xyz    = get_XYZ0(traj_file,:traj,:none;silent=true)
traj   = xyz.traj
ins    = xyz.ins
flux_a = xyz.flux_a

ind = trues(traj.N)
ind[51:end] .= false

xyz_h5 = joinpath(@__DIR__,"test_get_XYZ0.h5")
write_field(xyz_h5,:tt,traj.tt)
write_field(xyz_h5,:lat,rad2deg.(traj.lat))
write_field(xyz_h5,:lon,rad2deg.(traj.lon))
write_field(xyz_h5,:alt,traj.alt)
write_field(xyz_h5,:ins_tt,ins.tt)
write_field(xyz_h5,:ins_lat,rad2deg.(ins.lat))
write_field(xyz_h5,:ins_lon,rad2deg.(ins.lon))
write_field(xyz_h5,:ins_alt,ins.alt)
write_field(xyz_h5,:mag_1_uc,xyz.mag_1_uc)

@testset "get_XYZ0 tests" begin
    @test_nowarn get_XYZ0(traj_file,:traj,:none,
                          flight = 1,
                          line   = 1,
                          dt     = 1,
                          silent = true)
    @test typeof(get_XYZ0(xyz_h5;silent=true)) <: MagNav.XYZ0
    delete_field(xyz_h5,:tt)
    delete_field(xyz_h5,:ins_tt)
    delete_field(xyz_h5,:mag_1_uc)
    write_field(xyz_h5,:mag_1_c,xyz.mag_1_c)
    write_field(xyz_h5,:flight,xyz.flight)
    write_field(xyz_h5,:line,xyz.line)
    write_field(xyz_h5,:dt,traj.dt)
    write_field(xyz_h5,:roll,zero(traj.lat))
    write_field(xyz_h5,:pitch,zero(traj.lat))
    write_field(xyz_h5,:yaw,zero(traj.lat))
    write_field(xyz_h5,:ins_dt,ins.dt)
    write_field(xyz_h5,:ins_roll,zero(ins.lat))
    write_field(xyz_h5,:ins_pitch,zero(ins.lat))
    write_field(xyz_h5,:ins_yaw,zero(ins.lat))
    @test typeof(get_XYZ0(xyz_h5;silent=true)) <: MagNav.XYZ0
    MagNav.overwrite_field(xyz_h5,:ins_alt,ins.alt*NaN)
    @test typeof(get_XYZ0(xyz_h5;silent=true)) <: MagNav.XYZ0
    @test_throws AssertionError get_XYZ0("test")
end

rm(xyz_h5)

@testset "get_traj tests" begin
    @test_nowarn get_traj(traj_file,:traj;silent=true)
    @test get_traj(xyz,ind).Cnb == traj(ind).Cnb
    @test_throws AssertionError get_traj("test")
end

@testset "get_ins tests" begin
    @test_nowarn get_ins(ins_file,:ins_data;silent=true)
    @test get_ins(xyz,ind).P == ins(ind).P
    @test get_ins(xyz,ind;t_zero_ll=10).lat[1:10] == traj.lat[ind][1:10]
    @test_nowarn MagNav.zero_ins_ll(ins.lat,ins.lon,1,
                                    traj.lat[1:1],traj.lon[1:1])
    @test_throws AssertionError get_ins("test")
end

@testset "get_flux tests" begin
    @test_nowarn flux_a(ind)
end
