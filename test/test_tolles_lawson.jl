using MagNav, Test, MAT, Statistics

test_file = joinpath(@__DIR__,"test_data/test_data_params.mat")
params    = matopen(test_file,"r") do file
    read(file,"params")
end

test_file = joinpath(@__DIR__,"test_data/test_data_TL.mat")
TL_data   = matopen(test_file,"r") do file
    read(file,"TL_data")
end

pass1     = params["TL"]["pass1"]
pass2     = params["TL"]["pass2"]
trim      = round(Int,params["TL"]["trim"])
λ         = params["TL"]["lambda"]
Bt_scale  = params["TL"]["Bt_scale"]

fs        = TL_data["fs"]
flux_a_x  = vec(TL_data["flux_a_x"])
flux_a_y  = vec(TL_data["flux_a_y"])
flux_a_z  = vec(TL_data["flux_a_z"])
mag_1_uc  = vec(TL_data["mag_1_uc"])

A_a       = create_TL_A(flux_a_x,flux_a_y,flux_a_z;Bt_scale=Bt_scale)
TL_a_1    = create_TL_coef(flux_a_x,flux_a_y,flux_a_z,mag_1_uc;
                           λ=λ,pass1=pass1,pass2=pass2,fs=fs,
                           trim=trim,Bt_scale=Bt_scale)
mag_1_c   = mag_1_uc - A_a*TL_a_1
mag_1_c_d = mag_1_uc - detrend(A_a*TL_a_1)

@testset "create_TL_A tests" begin
    @test A_a ≈ TL_data["A_a"]
end

@testset "create_TL_coef tests" begin
    @test std(mag_1_c-TL_data["mag_1_c"]) < 0.1
    @test std(mag_1_c_d-TL_data["mag_1_c_d"]) < 0.1
end

@testset "create_TL_A & create_TL_coef arguments tests" begin
    terms_set = [[:permanent,:induced,:eddy,:bias],
                 [:permanent,:induced,:eddy],
                 [:permanent,:induced],
                 [:permanent],
                 [:induced],
                 [:i5,:e8,:fdm],
                 [:i3,:e3,:f3]]

    for terms in terms_set
        @test_nowarn create_TL_A(flux_a_x,flux_a_y,flux_a_z;terms=terms)
        @test_nowarn create_TL_coef(flux_a_x,flux_a_y,flux_a_z,mag_1_uc;
                                    terms=terms)
    end

    @test_nowarn create_TL_coef(flux_a_x,flux_a_y,flux_a_z,mag_1_uc;
                                fs=fs,pass1=0)
    @test_nowarn create_TL_coef(flux_a_x,flux_a_y,flux_a_z,mag_1_uc;
                                fs=fs,pass2=fs)
    @test std(create_TL_coef(flux_a_x,flux_a_y,flux_a_z,mag_1_uc;
                              fs=fs,pass1=0,pass2=fs)) >= 0
end

@testset "fdm tests" begin
    @test_nowarn fdm(mag_1_c_d;central=true ,fourth=false)
    @test_nowarn fdm(mag_1_c_d;central=false,fourth=false)
    @test_nowarn fdm(mag_1_c_d;central=true ,fourth=true)
    @test_nowarn fdm(mag_1_c_d;central=false,fourth=true)
end
