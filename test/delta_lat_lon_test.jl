using MagNav, Test

lat = 39.160667350241980

d_north_1 = 1.0
d_north_2 = [0.5,10.0,200.0]

d_east_1  = 1.0
d_east_2  = [0.5,10.0,200.0]

d_lat_1   = 1.565761736512648e-07
d_lat_2   = [7.828808682563242e-08,1.565761736512649e-06,3.131523473025297e-05]

d_lon_1   = 2.019352321699552e-07
d_lon_2   = [1.009676160849776e-07,2.019352321699552e-06,4.038704643399104e-05]

@testset "Delta Lat Lon Tests" begin
    @test delta_lat(d_north_1,deg2rad(lat)) ≈ d_lat_1
    @test delta_lon(d_east_1 ,deg2rad(lat)) ≈ d_lon_1
    @test delta_lat(d_north_2,deg2rad(lat)) ≈ d_lat_2
    @test delta_lon(d_east_2 ,deg2rad(lat)) ≈ d_lon_2
end

@testset "Delta North East Tests" begin
    @test delta_north(d_lat_1,deg2rad(lat)) ≈ d_north_1
    @test delta_east( d_lon_1,deg2rad(lat)) ≈ d_east_1
    @test delta_north(d_lat_2,deg2rad(lat)) ≈ d_north_2
    @test delta_east( d_lon_2,deg2rad(lat)) ≈ d_east_2
end
