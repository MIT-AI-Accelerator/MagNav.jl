using MagNav, Test, MAT

test_file = "test_data_TL.mat"
TL_data = matopen(test_file,"r") do file
    read(file,"TL_data")
end

mag_uc_f_t  = TL_data["mag_uc_f_t"]
mag_c_f_t_o = TL_data["mag_c_f_t_o"]
A_f_t       = TL_data["A_f_t"]
TL_coef     = TL_data["TL_coef"]
mag_cor     = TL_data["mag_cor"]
mag_cor_d   = TL_data["mag_cor_d"]

@testset "Linear Regression Tests" begin
    @test linreg(mag_uc_f_t,A_f_t)  ≈ TL_coef
    @test linreg([3,6,9])           ≈ [0,3]

end

@testset "Detrend Tests" begin
    @test detrend(mag_uc_f_t,A_f_t) ≈ mag_c_f_t_o
    @test detrend(mag_cor)          ≈ mag_cor_d
    @test detrend([3,6,8])          ≈ [-1,2,-1] / 6
end
