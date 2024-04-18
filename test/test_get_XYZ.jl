using MagNav, Test, MAT
using DataFrames, CSV
using MagNav: delete_field, write_field

traj_mat = joinpath(@__DIR__,"test_data/test_data_traj.mat")
ins_mat  = joinpath(@__DIR__,"test_data/test_data_ins.mat")

traj_field = :traj
ins_field  = :ins_data

xyz    = get_XYZ0(traj_mat,traj_field,:none;silent=true)
traj   = xyz.traj
ins    = xyz.ins
flux_a = xyz.flux_a

ind = trues(traj.N)
ind[51:end] .= false

df_traj = DataFrame(lat      = traj.lat,
                    lon      = traj.lon,
                    alt      = traj.alt,
                    mag_1_c  = xyz.mag_1_c,
                    flux_a_x = flux_a.x,
                    flux_a_y = flux_a.y,
                    flux_a_z = flux_a.z,
                    flux_a_t = flux_a.t)

df_ins  = DataFrame(ins_lat  = ins.lat,
                    ins_lon  = ins.lon,
                    ins_alt  = ins.alt)

traj_csv = joinpath(@__DIR__,"test_traj.csv")
xyz_csv  = joinpath(@__DIR__,"test_xyz.csv")
CSV.write(traj_csv,df_traj)
CSV.write(xyz_csv,hcat(df_traj,df_ins))

traj_data = matopen(traj_mat,"r") do file
    read(file,"$traj_field")
end

ins_data = matopen(ins_mat,"r") do file
    read(file,"$ins_field")
end

xyz_mat = joinpath(@__DIR__,"test_xyz.mat")
matopen(xyz_mat,"w") do file
    write(file,"$traj_field",traj_data)
    write(file,"$ins_field" ,ins_data)
end

xyz_h5 = joinpath(@__DIR__,"test_xyz.h5")
write_field(xyz_h5,:tt,traj.tt)
write_field(xyz_h5,:lat,rad2deg.(traj.lat))
write_field(xyz_h5,:lon,rad2deg.(traj.lon))
write_field(xyz_h5,:alt,traj.alt)
write_field(xyz_h5,:ins_tt,ins.tt)
write_field(xyz_h5,:ins_lat,rad2deg.(ins.lat))
write_field(xyz_h5,:ins_lon,rad2deg.(ins.lon))
write_field(xyz_h5,:ins_alt,ins.alt)
write_field(xyz_h5,:mag_1_uc,xyz.mag_1_uc)

flights   = [:Flt1002,:Flt1003,:Flt1004]
xyz_types = [:XYZ0,:XYZ1,:test]
xyz_sets  = [0,0,0]
xyz_files = [xyz_h5,xyz_h5,xyz_h5]
df_flight = DataFrame(flight   = flights,
                      xyz_type = xyz_types,
                      xyz_set  = xyz_sets,
                      xyz_file = xyz_files)

@testset "get_XYZ0 & get_XYZ1 tests" begin
    @test get_XYZ0(xyz_csv ;silent=true) isa MagNav.XYZ0
    @test get_XYZ1(traj_csv;silent=true) isa MagNav.XYZ1
    @test get_XYZ0(traj_mat,traj_field,:none,
                   flight = 1,
                   line   = 1,
                   dt     = 1,
                   silent = true) isa MagNav.XYZ0
    @test get_XYZ1(xyz_mat,traj_field,ins_field,
                   flight = 1,
                   line   = 1,
                   dt     = 1,
                   silent = true) isa MagNav.XYZ1
    @test get_XYZ0(xyz_h5;silent=true) isa MagNav.XYZ0
    @test get_XYZ1(xyz_h5;silent=true) isa MagNav.XYZ1
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
    @test get_XYZ0(xyz_h5;silent=true) isa MagNav.XYZ0
    @test get_XYZ(flights[1],df_flight)  isa MagNav.XYZ0
    @test get_XYZ(flights[2],df_flight)  isa MagNav.XYZ1
    @test_throws ErrorException get_XYZ(flights[3],df_flight)
    MagNav.overwrite_field(xyz_h5,:ins_alt,ins.alt*NaN)
    @test_throws ErrorException get_XYZ0(xyz_h5;silent=true)
    @test_throws AssertionError get_XYZ0("test")
end

@testset "get_traj tests" begin
    @test get_traj(traj_csv           ;silent=true) isa MagNav.Traj
    @test get_traj(traj_mat,traj_field;silent=true) isa MagNav.Traj
    @test get_traj(xyz,ind).Cnb == traj(ind).Cnb
    @test_throws AssertionError get_traj("test")
end

@testset "get_ins tests" begin
    @test get_ins(xyz_csv          ;silent=true) isa MagNav.INS
    @test get_ins(xyz_mat,ins_field;silent=true) isa MagNav.INS
    @test get_ins(xyz,ind).P == ins(ind).P
    @test get_ins(xyz,ind;N_zero_ll=5).lat[1:5] == traj.lat[ind][1:5]
    @test get_ins(xyz,ind;t_zero_ll=4).lat[1:5] == traj.lat[ind][1:5]
    @test MagNav.zero_ins_ll(ins.lat,ins.lon,1,
                             traj.lat[1:1],traj.lon[1:1]) isa NTuple{2,Vector}
    @test_throws AssertionError get_ins("test")
end

@testset "get_flux tests" begin
    @test get_flux(traj_csv,:flux_a           ) isa MagNav.MagV
    @test get_flux(traj_mat,:flux_a,traj_field) isa MagNav.MagV
    @test flux_a(ind) isa MagNav.MagV
end

rm(traj_csv)
rm(xyz_csv)
rm(xyz_mat)
rm(xyz_h5)

xyz_dir   = MagNav.sgl_2020_train()
flights   = [:Flt1002,:Flt1003,:Flt1004,:Flt1005,:Flt1006,:Flt1007]
xyz_types = repeat([:XYZ20],length(flights))
xyz_sets  = repeat([1],length(flights))
xyz_files = [xyz_dir*"/$(f)_train.h5" for f in flights]
df_flight = DataFrame(flight   = flights,
                      xyz_type = xyz_types,
                      xyz_set  = xyz_sets,
                      xyz_file = xyz_files)

@testset "get_XYZ20 tests" begin
    for xyz_file in xyz_files
        xyz = get_XYZ20(xyz_file;tt_sort=true,silent=true)
        @test xyz isa MagNav.XYZ20
        @test xyz.traj.N ≈ length(xyz.traj.lat) ≈ length(xyz.traj.lon)
    end
    for xyz_file in xyz_files #* note: not actually 160 Hz, should still pass
        xyz = get_XYZ20(xyz_file,xyz_file;silent=true)
        @test xyz isa MagNav.XYZ20
        @test xyz.traj.N ≈ length(xyz.traj.lat) ≈ length(xyz.traj.lon)
    end
    for flight in flights
        xyz = get_XYZ(flight,df_flight;tt_sort=true,reorient_vec=true,silent=true)
        @test xyz isa MagNav.XYZ20
        @test xyz.traj.N ≈ length(xyz.traj.lat) ≈ length(xyz.traj.lon)
    end
end

xyz_dir   = MagNav.sgl_2021_train()
flights   = [:Flt2001,:Flt2002,:Flt2004,:Flt2005,:Flt2006,
             :Flt2007,:Flt2008,:Flt2015,:Flt2016,:Flt2017]
xyz_types = repeat([:XYZ21],length(flights))
xyz_sets  = repeat([1],length(flights))
xyz_files = [xyz_dir*"/$(f)_train.h5" for f in flights]
df_flight = DataFrame(flight   = flights,
                      xyz_type = xyz_types,
                      xyz_set  = xyz_sets,
                      xyz_file = xyz_files)

@testset "get_XYZ21 tests" begin
    for xyz_file in xyz_files
        xyz = get_XYZ21(xyz_file;tt_sort=true,silent=true)
        @test xyz isa MagNav.XYZ21
        @test xyz.traj.N ≈ length(xyz.traj.lat) ≈ length(xyz.traj.lon)
    end
    for flight in flights
        xyz = get_XYZ(flight,df_flight;tt_sort=true,reorient_vec=true,silent=true)
        @test xyz isa MagNav.XYZ21
        @test xyz.traj.N ≈ length(xyz.traj.lat) ≈ length(xyz.traj.lon)
    end
end
