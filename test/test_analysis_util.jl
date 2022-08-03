using MagNav, Test, MAT, DataFrames

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

@testset "dn2dlat & de2dlon tests" begin
    @test dn2dlat(dn_1,lat) ≈ dlat_1
    @test de2dlon(de_1,lat) ≈ dlon_1
    @test dn2dlat(dn_2,lat) ≈ dlat_2
    @test de2dlon(de_2,lat) ≈ dlon_2
end

@testset "dlat2dn & dlon2de tests" begin
    @test dlat2dn(dlat_1,lat) ≈ dn_1
    @test dlon2de(dlon_1,lat) ≈ de_1
    @test dlat2dn(dlat_2,lat) ≈ dn_2
    @test dlon2de(dlon_2,lat) ≈ de_2
end

@testset "linreg tests" begin
    @test TL_a_1           ≈ TL_data["TL_a_1"]
    @test linreg([3,6,9])  ≈ [0,3]
end

@testset "detrend tests" begin
    @test mag_1_comp_d     ≈ TL_data["mag_1_comp_d"]
    @test detrend([3,6,8]) ≈ [-1,2,-1] / 6
    @test detrend([3,6,8];mean_only=true) ≈ [-8,1,7] / 3
    @test detrend([1,1],[1 0.1; 0.1 1])   ≈ zeros(2)
    @test detrend([1,1],[1 0.1; 0.1 1];mean_only=true) ≈ zeros(2)
end

@testset "detrend tests" begin
    @test_nowarn bpf_data(A_a_f_t)
    @test_nowarn bpf_data(mag_1_uc_f_t)
    @test_nowarn bpf_data!(A_a_f_t)
    @test_nowarn bpf_data!(mag_1_uc_f_t)
end

flight = :Flt1003
xyz_h5 = string(MagNav.sgl_2020_train(),"/$(flight)_train.h5")
xyz    = get_XYZ20(xyz_h5;tt_sort=true,silent=true)
line   = xyz.line[1]
ind    = xyz.line .== line
ind[51:end] .= false

df_line = DataFrame(flight  = flight,
                    line    = line,
                    t_start = xyz.traj.tt[ind][1],
                    t_end   = xyz.traj.tt[ind][end],
                    test    = false)

df_flight = DataFrame(flight   = flight,
                      xyz_type = :XYZ20,
                      xyz_set  = 1,
                      xyz_h5   = xyz_h5)

@testset "get_x tests" begin
    @test_nowarn get_x(xyz,ind)
    @test_nowarn get_x([xyz,xyz],[ind,ind])
    @test_nowarn get_x(line,df_line,df_flight)
end

@testset "get_y tests" begin
    @test_nowarn get_y(xyz,ind;)
    @test_nowarn get_y(xyz,ind;y_type=:a)
    @test_nowarn get_y(line,df_line,df_flight,DataFrame();y_type=:e)
end

@testset "get_Axy tests" begin
    @test_nowarn get_Axy(line,df_line,df_flight,DataFrame())
end
