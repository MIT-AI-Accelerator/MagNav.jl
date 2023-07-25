using MagNav, Test, MAT
using DataFrames, Flux, LinearAlgebra, SatelliteToolbox, Statistics
using MagNav: project_vec_to_2d, unpack_data_norms

test_file = joinpath(@__DIR__,"test_data/test_data_params.mat")
params    = matopen(test_file,"r") do file
    read(file,"params")
end

test_file = joinpath(@__DIR__,"test_data/test_data_TL.mat")
TL_data   = matopen(test_file,"r") do file
    read(file,"TL_data")
end

λ = params["TL"]["lambda"]

A_a_f_t      = TL_data["A_a_f_t"]
mag_1_uc_f_t = vec(TL_data["mag_1_uc_f_t"])

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
    @test TL_a_1           ≈ vec(TL_data["TL_a_1"])
    @test linreg([3,6,9])  ≈ [0,3]
end

@testset "detrend tests" begin
    @test mag_1_comp_d     ≈ TL_data["mag_1_comp_d"]
    @test detrend([3,6,8]) ≈ [-1,2,-1] / 6
    @test detrend([3,6,8];mean_only=true) ≈ [-8,1,7] / 3
    @test detrend([1,1],[1 0.1; 0.1 1])   ≈ zeros(2)
    @test detrend([1,1],[1 0.1; 0.1 1];mean_only=true) ≈ zeros(2)
end

@testset "get_bpf tests" begin
    @test_nowarn get_bpf(;pass1=0.1,pass2=0.9)
    @test_nowarn get_bpf(;pass1= -1,pass2=0.9)
    @test_nowarn get_bpf(;pass1=Inf,pass2=0.9)
    @test_nowarn get_bpf(;pass1=0.1,pass2= -1)
    @test_nowarn get_bpf(;pass1=0.1,pass2=Inf)
    @test_throws ErrorException get_bpf(;pass1=-1,pass2=-1)
    @test_throws ErrorException get_bpf(;pass1=Inf,pass2=Inf)
end

@testset "bpf_data tests" begin
    @test_nowarn bpf_data(A_a_f_t)
    @test_nowarn bpf_data(mag_1_uc_f_t)
    @test_nowarn bpf_data!(A_a_f_t)
    @test_nowarn bpf_data!(mag_1_uc_f_t)
end

flight   = :Flt1003
xyz_type = :XYZ20
map_name = :Eastern_395
xyz_h5   = string(MagNav.sgl_2020_train(),"/$(flight)_train.h5")
map_h5   = string(MagNav.ottawa_area_maps(),"/$(map_name).h5")
xyz      = get_XYZ20(xyz_h5;tt_sort=true,silent=true)

tt    = xyz.traj.tt
line  = xyz.line
line2 = unique(line)[2]
line3 = unique(line)[3]
lines = [-1,line2]
ind   = line .== line2
ind[findall(ind)[51:end]] .= false

df_line = DataFrame(flight   = [flight,flight],
                    line     = [line2,line3],
                    t_start  = [tt[ind][1]  ,tt[get_ind(xyz;lines=line3)][1]],
                    t_end    = [tt[ind][end],tt[get_ind(xyz;lines=line3)][5]],
                    map_name = [map_name,map_name])

df_flight = DataFrame(flight   = flight,
                      xyz_type = xyz_type,
                      xyz_set  = 1,
                      xyz_h5   = xyz_h5)

df_map = DataFrame(map_name = map_name,
                   map_h5   = map_h5)

@testset "get_x tests" begin
    @test_nowarn get_x(xyz,ind)
    @test_nowarn get_x(xyz,ind,[:flight])
    @test_throws ErrorException get_x(xyz,ind,[:test])
    @test_throws ErrorException get_x(xyz,ind,[:mag_6_uc])
    @test_nowarn get_x([xyz,xyz],[ind,ind])
    @test_nowarn get_x(line2,df_line,df_flight)
    @test typeof(get_x(lines,df_line,df_flight)[1]) <: Matrix
    @test typeof(get_x([line2,line3],df_line,df_flight)[1]) <: Matrix
    @test_throws AssertionError get_x([line2,line2],df_line,df_flight)
end

map_val = randn(sum(ind))

@testset "get_y tests" begin
    @test_nowarn get_y(xyz,ind;y_type=:a)
    @test_nowarn get_y(xyz,ind,map_val;y_type=:b)
    @test_nowarn get_y(xyz,ind,map_val;y_type=:c)
    @test_nowarn get_y(xyz,ind;y_type=:d)
    @test_nowarn get_y(xyz,ind;y_type=:e,use_mag=:flux_a)
    @test_throws ErrorException get_y(xyz,ind;y_type=:test)
    @test typeof(get_y(line2,df_line,df_flight,df_map;y_type=:a)) <: Vector
    @test typeof(get_y(lines,df_line,df_flight,df_map;y_type=:c)) <: Vector
    @test typeof(get_y([line2,line3],df_line,df_flight,df_map;y_type=:d)) <: Vector
    @test_throws AssertionError get_y([line2,line2],df_line,df_flight,df_map)
end

@testset "get_Axy tests" begin
    @test_nowarn get_Axy(line2,df_line,df_flight,df_map;y_type=:a,mod_TL=true,return_B=true)
    @test typeof(get_Axy(lines,df_line,df_flight,df_map;y_type=:b,map_TL=true)[1]) <: Matrix
    @test typeof(get_Axy([line2,line3],df_line,df_flight,df_map)[1]) <: Matrix
    @test_throws AssertionError get_Axy([line2,line2],df_line,df_flight,df_map)
end

@testset "get_nn_m tests" begin
    @test_nowarn get_nn_m(1,1;hidden=[])
    @test_nowarn get_nn_m(1,1;hidden=[1])
    @test_nowarn get_nn_m(1,1;hidden=[1,1])
    @test_nowarn get_nn_m(1,1;hidden=[1,1,1])
    @test_nowarn get_nn_m(1,1;hidden=[1],final_bias=false)
    @test_nowarn get_nn_m(1,1;hidden=[1],skip_con=true)
    @test_throws ErrorException get_nn_m(1,1;hidden=[1,1,1,1])
    @test_throws ErrorException get_nn_m(1,1;hidden=[1,1],skip_con=true)
end

m = get_nn_m(3,1;hidden=[1])
weights = Flux.params(m)
α = 0.5

@testset "sparse_group_lasso tests" begin
    @test sparse_group_lasso(m)   ≈ sparse_group_lasso(weights)
    @test sparse_group_lasso(m,α) ≈ sparse_group_lasso(weights,α)
end

A = randn(Float32,5,9)
x = randn(Float32,5,3)
y = randn(Float32,5)
(A_bias,A_scale,A_norm) = norm_sets(A)
(x_bias,x_scale,x_norm) = norm_sets(x)
(y_bias,y_scale,y_norm) = norm_sets(y)

@testset "norm_sets tests" begin
    for norm_type in [:standardize,:normalize,:scale,:none]
        @test_nowarn norm_sets(x;norm_type=norm_type,no_norm=2)
        @test_nowarn norm_sets(x,x;norm_type=norm_type,no_norm=2)
        @test_nowarn norm_sets(x,x,x;norm_type=norm_type,no_norm=2)
        @test_nowarn norm_sets(y;norm_type=norm_type)
        @test_nowarn norm_sets(y,y;norm_type=norm_type)
        @test_nowarn norm_sets(y,y,y;norm_type=norm_type)
    end
    @test_throws ErrorException norm_sets(x;norm_type=:test)
    @test_throws ErrorException norm_sets(x,x;norm_type=:test)
    @test_throws ErrorException norm_sets(x,x,x;norm_type=:test)
    @test_throws ErrorException norm_sets(y;norm_type=:test)
    @test_throws ErrorException norm_sets(y,y;norm_type=:test)
    @test_throws ErrorException norm_sets(y,y,y;norm_type=:test)
end

@testset "denorm_sets tests" begin
    @test denorm_sets(x_bias,x_scale,x_norm) ≈ x
    @test denorm_sets(x_bias,x_scale,x_norm,x_norm)[1] ≈ x
    @test denorm_sets(x_bias,x_scale,x_norm,x_norm)[2] ≈ x
    @test denorm_sets(x_bias,x_scale,x_norm,x_norm,x_norm)[1] ≈ x
    @test denorm_sets(x_bias,x_scale,x_norm,x_norm,x_norm)[2] ≈ x
    @test denorm_sets(x_bias,x_scale,x_norm,x_norm,x_norm)[3] ≈ x
    @test denorm_sets(y_bias,y_scale,y_norm) ≈ y
    @test denorm_sets(y_bias,y_scale,y_norm,y_norm)[1] ≈ y
    @test denorm_sets(y_bias,y_scale,y_norm,y_norm)[2] ≈ y
    @test denorm_sets(y_bias,y_scale,y_norm,y_norm,y_norm)[1] ≈ y
    @test denorm_sets(y_bias,y_scale,y_norm,y_norm,y_norm)[2] ≈ y
    @test denorm_sets(y_bias,y_scale,y_norm,y_norm,y_norm)[3] ≈ y
end

v_scale      = I(size(x,2))
data_norms_7 = (A_bias,A_scale,v_scale,x_bias,x_scale,y_bias,y_scale)
data_norms_6 = (A_bias,A_scale,x_bias,x_scale,y_bias,y_scale)
data_norms_5 = (v_scale,x_bias,x_scale,y_bias,y_scale)
data_norms_4 = (x_bias,x_scale,y_bias,y_scale)
data_norms_A = (0,1,v_scale,x_bias,x_scale,y_bias,y_scale)

@testset "unpack_data_norms tests" begin
    @test unpack_data_norms(data_norms_7) == data_norms_7
    @test unpack_data_norms(data_norms_6) == data_norms_7
    @test unpack_data_norms(data_norms_5) == data_norms_A
    @test unpack_data_norms(data_norms_4) == data_norms_A
    @test_throws ErrorException unpack_data_norms((0,0,0,0,0,0,0,0))
    @test_throws ErrorException unpack_data_norms((0,0,0))
end

@testset "get_ind tests" begin
    @test sum(get_ind(tt,line;ind=ind)) ≈ 50
    @test sum(get_ind(tt[ind],line[ind];ind=1:50)) ≈ 50
    @test sum(get_ind(tt[ind],line[ind];tt_lim=(tt[ind][5]))) ≈ 5
    @test sum(get_ind(tt[ind],line[ind];tt_lim=(tt[ind][5],tt[ind][9]))) ≈ 5
    @test sum.(get_ind(tt,line;ind=ind,splits=(0.5,0.5))) == (25,25)
    @test sum.(get_ind(tt,line;ind=ind,splits=(0.7,0.2,0.1))) == (35,10,5)
    @test_throws AssertionError get_ind(tt[ind],line[ind];tt_lim=(1,1,1))
    @test_throws AssertionError get_ind(tt[ind],line[ind];splits=(1,1,1))
    @test_throws AssertionError get_ind(tt[ind],line[ind];splits=(1,0,0,0))
    @test sum(get_ind(xyz;lines=lines,tt_lim=(tt[ind][5]))) ≈ 5
    @test sum(get_ind(xyz,lines[2],df_line;l_seq=15)) ≈ 45
    @test sum.(get_ind(xyz,lines[2],df_line;splits=(0.5,0.5),l_seq=15)) == (15,15)
    @test sum(get_ind(xyz,lines,df_line)) ≈ 50
    @test sum.(get_ind(xyz,lines,df_line;splits=(0.5,0.5))) == (25,25)
    @test sum.(get_ind(xyz,lines,df_line;splits=(0.7,0.2,0.1))) == (35,10,5)
    @test_throws AssertionError get_ind(xyz,lines,df_line;splits=(1,1,1))
    @test_throws AssertionError get_ind(xyz,lines,df_line;splits=(1,0,0,0))
end

@testset "chunk_data tests" begin
    @test chunk_data(ones(3,2),ones(3),3) == ([[[1,1],[1,1],[1,1]]],[[1,1,1]])
end

m_rnn = Chain(GRU(3,1),Dense(1,1))

@testset "predict_rnn tests" begin
    @test_nowarn predict_rnn_full(m_rnn,x)
    @test_nowarn predict_rnn_windowed(m_rnn,x,3)
end

@testset "krr tests" begin
    @test typeof(krr(x,y,x,y)) <: Tuple{Vector,Vector}
end

features = [:f1,:f2,:f3]
(df_shap,baseline_shap) = eval_shapley(m,x,features)

@testset "shapley & gsa tests" begin
    @test_nowarn MagNav.predict_shapley(m,DataFrame(x,features))
    @test_nowarn eval_shapley(m,x,features)
    @test_nowarn plot_shapley(df_shap,baseline_shap)
    @test typeof(eval_gsa(m,x)) <: Vector
end

@testset "get_igrf tests" begin
    @test_nowarn get_igrf(xyz,ind)
    @test_nowarn get_igrf(xyz,ind;frame=:nav,norm_igrf=true)
end

vec_in   = randn(3)
vec_body = randn(3)
igrf_in  = vec_in ./ norm(vec_in)
dcm      = xyz.ins.Cnb[:,:,1]

@testset "project_vec_to_2d tests" begin
    @test_throws AssertionError project_vec_to_2d(vec_in,[1,0,0],[1,0,0])
    @test_throws AssertionError project_vec_to_2d(vec_in,[1,1,0],[0,1,0])
    @test_throws AssertionError project_vec_to_2d(vec_in,[1,0,0],[1,1,0])
    @test_nowarn project_vec_to_2d(vec_in,[1,0,0],[0,1,0])
end

@testset "project_body_field_to_2d_igrf tests" begin
    @test_nowarn project_body_field_to_2d_igrf(vec_body,igrf_in,dcm)
    # Create a "north" vector in the body frame
    north_vec_nav = vec([1.0, 0.0, 0.0]);
    dcms = xyz.ins.Cnb[:,:,ind] #
    # Check that North in 3-D stays as North in 2-D (repeat below for cardinal dirs)
    north_vec_body = [dcms[:,:,i]'*north_vec_nav for i=1:size(dcms)[3]];
    n2ds = [project_body_field_to_2d_igrf(north_vec_body[i], north_vec_nav, dcms[:,:,i])
             for i=1:size(dcms)[3]];
    @test all(map(v -> v ≈ [1.0, 0.0], n2ds))

    east_vec_nav = vec([0.0, 1.0, 0.0])
    east_vec_body = [dcms[:,:,i]'*east_vec_nav for i=1:size(dcms)[3]]
    e2ds = [project_body_field_to_2d_igrf(east_vec_body[i], north_vec_nav, dcms[:,:,i])
            for i=1:size(dcms)[3]]
    @test all(map(v -> v ≈ [0.0, 1.0], e2ds))

    west_vec_nav = vec([0.0, -1.0, 0.0])
    west_vec_body = [dcms[:,:,i]'*west_vec_nav for i=1:size(dcms)[3]]
    w2ds = [project_body_field_to_2d_igrf(west_vec_body[i], north_vec_nav, dcms[:,:,i])
            for i=1:size(dcms)[3]]
    @test all(map(v -> v ≈ [0.0, -1.0], w2ds))

    south_vec_nav = vec([-1.0, 0.0, 0.0])
    south_vec_body = [dcms[:,:,i]'*south_vec_nav for i=1:size(dcms)[3]]
    s2ds = [project_body_field_to_2d_igrf(south_vec_body[i], north_vec_nav, dcms[:,:,i])
            for i=1:size(dcms)[3]]
    @test all(map(v -> v ≈ [-1.0, 0.0], s2ds))
end

@testset "get_optimal_rotation_matrix tests" begin
    @test_throws AssertionError get_optimal_rotation_matrix([1 0],vec_body')
    @test_throws AssertionError get_optimal_rotation_matrix([1 0],[1 0 0])
    @test_throws AssertionError get_optimal_rotation_matrix([1 0 0],[1 0])
    @test_nowarn get_optimal_rotation_matrix(vec_in',vec_body')
end
