using MagNav, Test, DataFrames

xyz_dir   = MagNav.sgl_2020_train()
flights   = [:Flt1002,:Flt1003,:Flt1004,:Flt1005,:Flt1006,:Flt1007]
xyz_types = [:XYZ20,:XYZ20,:XYZ20,:XYZ20,:XYZ20,:XYZ20]
xyz_sets  = [1,1,1,1,1,1]
xyz_files = [string(xyz_dir,"/$(f)_train.h5") for f in flights]
df_flight = DataFrame(flight   = flights,
                      xyz_type = xyz_types,
                      xyz_set  = xyz_sets,
                      xyz_h5   = xyz_files)

@testset "get_XYZ20 tests" begin
    for xyz_h5 in xyz_files
        xyz = get_XYZ20(xyz_h5;tt_sort=true,silent=true)
        @test xyz.traj.N ≈ length(xyz.traj.lat) ≈ length(xyz.traj.lon)
    end
    for flight in flights
        xyz = get_XYZ(flight,df_flight;tt_sort=true,silent=true)
        @test xyz.traj.N ≈ length(xyz.traj.lat) ≈ length(xyz.traj.lon)
    end
end

@testset "get_XYZ21 tests" begin #* not actually XYZ21, should still pass
    @test_nowarn get_XYZ21(xyz_files[1];tt_sort=true,silent=true)
end
