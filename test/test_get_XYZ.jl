using MagNav, Test, MAT, DataFrames
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

@testset "get_xyz tests" begin
    @test_nowarn get_XYZ0(traj_file,:traj,:none,
                          flight = 1,
                          line   = 1,
                          dt     = 1,
                          silent = true)
    @test_nowarn get_XYZ1(traj_file,:traj,:none,
                          flight = 1,
                          line   = 1,
                          dt     = 1,
                          silent = true)
    @test typeof(get_XYZ0(xyz_h5;silent=true)) <: MagNav.XYZ0
    @test typeof(get_XYZ1(xyz_h5;silent=true)) <: MagNav.XYZ1
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
    @test_throws ErrorException get_XYZ0(xyz_h5;silent=true)
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
    @test_nowarn get_flux(traj_file,:flux_a,:traj)
end

xyz_dir   = MagNav.sgl_2020_train()
flights   = [:Flt1002,:Flt1003,:Flt1004,:Flt1005,:Flt1006,:Flt1007]
xyz_types = [:XYZ20,:XYZ20,:XYZ20,:XYZ20,:XYZ20,:XYZ20]
xyz_sets  = [1,1,1,1,1,1]
xyz_files = [xyz_dir*"/$(f)_train.h5" for f in flights]
df_flight = DataFrame(flight   = flights,
                      xyz_type = xyz_types,
                      xyz_set  = xyz_sets,
                      xyz_h5   = xyz_files)

@testset "get_XYZ20 tests" begin
    for xyz_h5 in xyz_files
        xyz = get_XYZ20(xyz_h5;tt_sort=true,silent=true)
        @test xyz.traj.N ≈ length(xyz.traj.lat) ≈ length(xyz.traj.lon)
    end
    for xyz_h5 in xyz_files #* not actually 160 Hz, should still pass
        xyz = get_XYZ20(xyz_h5,xyz_h5;silent=true)
        @test xyz.traj.N ≈ length(xyz.traj.lat) ≈ length(xyz.traj.lon)
    end
    for flight in flights
        xyz = get_XYZ(flight,df_flight;tt_sort=true,reorient_vec=true,silent=true)
        @test xyz.traj.N ≈ length(xyz.traj.lat) ≈ length(xyz.traj.lon)
    end
end

df_flight[:,:xyz_type] .= :XYZ21

#* not actually XYZ21, should still pass, keeping this way until public release
@testset "get_XYZ21 tests" begin
    @test_nowarn get_XYZ21(xyz_files[1];tt_sort=true,silent=true)
    @test_nowarn get_XYZ(flights[1],df_flight;tt_sort=true,silent=true)
end
