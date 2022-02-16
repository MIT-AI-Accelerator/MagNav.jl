using MagNav, Test

data_dir  = MagNav.data_dir()
data_2_h5 = string(data_dir,"/Flt1002_train.h5")
data_3_h5 = string(data_dir,"/Flt1003_train.h5")
data_4_h5 = string(data_dir,"/Flt1004_train.h5")
data_5_h5 = string(data_dir,"/Flt1005_train.h5")
data_2    = get_flight_data(data_2_h5);
data_3    = get_flight_data(data_3_h5);
data_4    = get_flight_data(data_4_h5;tt_sort=false);
data_5    = get_flight_data(data_5_h5;tt_sort=false);

@testset "Artifact N Tests" begin
    @test data_2.N == length(data_2.LAT) == length(data_2.LONG)
    @test data_3.N == length(data_3.LAT) == length(data_3.LONG)
    @test data_4.N == length(data_4.LAT) == length(data_4.LONG)
    @test data_5.N == length(data_5.LAT) == length(data_5.LONG)
end

@testset "Artifact DT Tests" begin
    @test data_2.DT == data_3.DT == data_4.DT == data_5.DT == 0.1
end
