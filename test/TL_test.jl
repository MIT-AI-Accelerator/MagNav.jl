using MagNav, Test, DSP, MAT, Statistics

test_file = "test_data_TL.mat"
TL_data = matopen(test_file,"r") do file
    read(file,"TL_data")
end

Bx         = TL_data["Bx"]
By         = TL_data["By"]
Bz         = TL_data["Bz"]
meas_uc    = TL_data["mag_uc"]
meas_uc_t  = TL_data["mag_uc_t"]

pass1 = TL_data["Fpass1"]
pass2 = TL_data["Fpass2"]
fs    = TL_data["Fs"]

tr = round(Int64,TL_data["trim"])
A  = create_TL_A(Bx,By,Bz)[tr+1:end-tr,:]

TL_coef  = create_TL_coef(Bx,By,Bz,meas_uc; pass1=pass1,pass2=pass2,fs=fs)
meas_c_t = meas_uc_t - A*TL_coef .+ mean(A*TL_coef)

@testset "Create TL A Tests" begin
    @test A ≈ TL_data["A_t"]
end

@testset "Create TL Coefficients Tests" begin
    @test mean(abs.(meas_c_t - TL_data["mag_c_t"])) < 0.1
    @test rms(meas_c_t - TL_data["mag_c_t"]) < 0.1
end

@testset "TL input options" begin
    # test that the code runs with a variety of different terms selected
    terms = [ ["permanent","induced","eddy"],
              ["permanent","induced"],
              ["permanent"],
              ["induced"] ]

    for t in terms
        local A, TL_coef, meas_c_t

        @test_nowarn A  = create_TL_A(Bx,By,Bz; terms = t)[tr+1:end-tr,:]

        @test_nowarn TL_coef  = create_TL_coef(Bx,By,Bz,meas_uc; pass1=pass1,pass2=pass2,fs=fs,terms=t)
        # meas_c_t = meas_uc_t - A*TL_coef .+ mean(A*TL_coef)
    end

    @test_nowarn A  = create_TL_coef(Bx,By,Bz,meas_uc; pass1=0.0)[tr+1:end-tr,:]
    @test_nowarn A  = create_TL_coef(Bx,By,Bz,meas_uc; pass2=fs*10)[tr+1:end-tr,:]
    @test_nowarn A  = create_TL_coef(Bx,By,Bz,meas_uc; pass1=0.0,pass2=fs*10)[tr+1:end-tr,:]
    

end
