using MagNav, Test, MAT

test_file = "test_data_TL.mat"
TL_data = matopen(test_file,"r") do file
    read(file,"TL_data")
end

mag_cor   = TL_data["mag_cor"]
mag_cor_d = TL_data["mag_cor_d"]

@testset "Detrend Tests" begin
    @test detrend(mag_cor) ≈ mag_cor_d
    @test detrend([3,6,8]) ≈ [-1,2,-1] / 6
end
