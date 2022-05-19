using MagNav, Test

data_dir  = MagNav.sgl_2020_train()
data_2_h5 = string(data_dir,"/Flt1002_train.h5")
data_3_h5 = string(data_dir,"/Flt1003_train.h5")
data_4_h5 = string(data_dir,"/Flt1004_train.h5")
data_5_h5 = string(data_dir,"/Flt1005_train.h5")
data_6_h5 = string(data_dir,"/Flt1006_train.h5")
data_7_h5 = string(data_dir,"/Flt1007_train.h5")
data_2    = get_XYZ21(data_2_h5;tt_sort=true,silent=true);
data_3    = get_XYZ21(data_3_h5;tt_sort=true,silent=true);
data_4    = get_XYZ21(data_4_h5;tt_sort=false,silent=true);
data_5    = get_XYZ21(data_5_h5;tt_sort=false,silent=true);
data_6    = get_XYZ21(data_6_h5;tt_sort=false,silent=true);
data_7    = get_XYZ21(data_7_h5;tt_sort=false,silent=true);

@testset "Artifact N Tests" begin
    @test data_2.traj.N == length(data_2.traj.lat) == length(data_2.traj.lon)
    @test data_3.traj.N == length(data_3.traj.lat) == length(data_3.traj.lon)
    @test data_4.traj.N == length(data_4.traj.lat) == length(data_4.traj.lon)
    @test data_5.traj.N == length(data_5.traj.lat) == length(data_5.traj.lon)
    @test data_6.traj.N == length(data_6.traj.lat) == length(data_6.traj.lon)
    @test data_7.traj.N == length(data_7.traj.lat) == length(data_7.traj.lon)
end

@testset "Artifact dt Tests" begin
    @test data_2.traj.dt == data_3.traj.dt == data_4.traj.dt == 0.1
    @test data_5.traj.dt == data_6.traj.dt == data_7.traj.dt == 0.1
end
