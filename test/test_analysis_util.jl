using MagNav, Test, MAT

test_file = "test_data/test_data_params.mat"
params    = matopen(test_file,"r") do file
    read(file,"params")
end

test_file = "test_data/test_data_TL.mat"
TL_data   = matopen(test_file,"r") do file
    read(file,"TL_data")
end

λ = params["TL"]["lambda"]

A_a_f_t      = TL_data["A_a_f_t"]
mag_1_uc_f_t = TL_data["mag_1_uc_f_t"]

TL_a_1       = linreg(mag_1_uc_f_t,A_a_f_t;λ=λ)
mag_1_comp_d = detrend(TL_data["mag_1_comp"])

lat    = deg2rad(39.160667350241980)
dn_1   = 1.0
dn_2   = [0.5,10.0,200.0]
de_1   = 1.0
de_2   = [0.5,10.0,200.0]
dlat_1 = 1.565761736512648e-07
dlat_2 = [7.828808682563242e-08,1.565761736512649e-06,3.131523473025297e-05]
dlon_1 = 2.019352321699552e-07
dlon_2 = [1.009676160849776e-07,2.019352321699552e-06,4.038704643399104e-05]

@testset "Delta Lat Lon Tests" begin
    @test dn2dlat(dn_1,lat) ≈ dlat_1
    @test de2dlon(de_1,lat) ≈ dlon_1
    @test dn2dlat(dn_2,lat) ≈ dlat_2
    @test de2dlon(de_2,lat) ≈ dlon_2
end

@testset "Delta North East Tests" begin
    @test dlat2dn(dlat_1,lat) ≈ dn_1
    @test dlon2de(dlon_1,lat) ≈ de_1
    @test dlat2dn(dlat_2,lat) ≈ dn_2
    @test dlon2de(dlon_2,lat) ≈ de_2
end

@testset "Linear Regression Tests" begin
    @test TL_a_1           ≈ TL_data["TL_a_1"]
    @test linreg([3,6,9])  ≈ [0,3]
end

@testset "Detrend Tests" begin
    @test mag_1_comp_d     ≈ TL_data["mag_1_comp_d"]
    @test detrend([3,6,8]) ≈ [-1,2,-1] / 6
    @test detrend([1,1],[1 0.1; 0.1 1]) ≈ zeros(2)
end
