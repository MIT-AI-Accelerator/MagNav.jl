using MagNav, Test, MAT
using DataFrames, DSP, Flux, LinearAlgebra, Plots, Statistics
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
    @test TL_a_1          ≈ vec(TL_data["TL_a_1"])
    @test linreg([3,6,9]) ≈ [0,3]
end

@testset "detrend tests" begin
    @test mag_1_comp_d     ≈ TL_data["mag_1_comp_d"]
    @test detrend([3,6,8]) ≈ [-1,2,-1] / 6
    @test detrend([3,6,8];mean_only=true) ≈ [-8,1,7] / 3
    @test detrend([1,1],[1 0.1; 0.1 1])   ≈ zeros(2)
    @test detrend([1,1],[1 0.1; 0.1 1];mean_only=true) ≈ zeros(2)
end

@testset "get_bpf tests" begin
    @test get_bpf(;pass1=0.1,pass2=0.9) isa DSP.Filters.ZeroPoleGain
    @test get_bpf(;pass1= -1,pass2=0.9) isa DSP.Filters.ZeroPoleGain
    @test get_bpf(;pass1=Inf,pass2=0.9) isa DSP.Filters.ZeroPoleGain
    @test get_bpf(;pass1=0.1,pass2= -1) isa DSP.Filters.ZeroPoleGain
    @test get_bpf(;pass1=0.1,pass2=Inf) isa DSP.Filters.ZeroPoleGain
    @test_throws ErrorException get_bpf(;pass1=-1,pass2=-1)
    @test_throws ErrorException get_bpf(;pass1=Inf,pass2=Inf)
end

@testset "bpf_data tests" begin
    @test bpf_data( A_a_f_t     ) isa Matrix
    @test bpf_data( mag_1_uc_f_t) isa Vector
    @test bpf_data!(A_a_f_t     ) isa Nothing
    @test bpf_data!(mag_1_uc_f_t) isa Nothing
end

@testset "downsample tests" begin
    @test size(MagNav.downsample(A_a_f_t,100 )) == (96,18)
    @test size(MagNav.downsample(mag_1_uc_f_t)) == (960,)
end

flight   = :Flt1003
xyz_type = :XYZ20
map_name = :Eastern_395
xyz_h5   = MagNav.sgl_2020_train()*"/$(flight)_train.h5"
map_h5   = MagNav.ottawa_area_maps()*"/$(map_name).h5"
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
                      xyz_file = xyz_h5)

df_map = DataFrame(map_name = map_name,
                   map_file = map_h5)

@testset "get_x tests" begin
    @test get_x(xyz,ind) isa Tuple{Matrix,Vector,Vector,Vector}
    @test get_x(xyz,ind,[:flight]) isa Tuple{Matrix,Vector,Vector,Vector}
    @test_throws ErrorException get_x(xyz,ind,[:test])
    @test_throws ErrorException get_x(xyz,ind,[:mag_6_uc])
    @test get_x([xyz,xyz],[ind,ind]) isa Tuple{Matrix,Vector,Vector,Vector}
    @test get_x(line2,df_line,df_flight) isa Tuple{Matrix,Vector,Vector,Vector}
    @test get_x(lines,df_line,df_flight) isa Tuple{Matrix,Vector,Vector,Vector}
    @test get_x([line2,line3],df_line,df_flight) isa Tuple{Matrix,Vector,Vector,Vector}
    @test_throws AssertionError get_x([line2,line2],df_line,df_flight)
end

map_val = randn(sum(ind))

@testset "get_y tests" begin
    @test get_y(xyz,ind;y_type=:a) isa Vector
    @test get_y(xyz,ind,map_val;y_type=:b) isa Vector
    @test get_y(xyz,ind,map_val;y_type=:c) isa Vector
    @test get_y(xyz,ind;y_type=:d) isa Vector
    @test get_y(xyz,ind;y_type=:e,use_mag=:flux_a) isa Vector
    @test_throws ErrorException get_y(xyz,ind;y_type=:test)
    @test get_y(line2,df_line,df_flight,df_map;y_type=:a) isa Vector
    @test get_y(lines,df_line,df_flight,df_map;y_type=:c) isa Vector
    @test get_y([line2,line3],df_line,df_flight,df_map;y_type=:d) isa Vector
    @test_throws AssertionError get_y([line2,line2],df_line,df_flight,df_map)
end

@testset "get_Axy tests" begin
    @test get_Axy(line2,df_line,df_flight,df_map;y_type=:a,mod_TL=true,return_B=true)[1] isa Matrix
    @test get_Axy(lines,df_line,df_flight,df_map;y_type=:b,map_TL=true)[2] isa Matrix
    @test get_Axy([line2,line3],df_line,df_flight,df_map)[3] isa Vector
    @test_throws AssertionError get_Axy([line2,line2],df_line,df_flight,df_map)
end

@testset "get_nn_m tests" begin
    @test get_nn_m(1,1;hidden=[     ]) isa Chain
    @test get_nn_m(1,1;hidden=[1    ]) isa Chain
    @test get_nn_m(1,1;hidden=[1,1  ]) isa Chain
    @test get_nn_m(1,1;hidden=[1,1,1]) isa Chain
    @test get_nn_m(1,1;hidden=[1],final_bias=false) isa Chain
    @test get_nn_m(1,1;hidden=[1],skip_con  =true ) isa Chain
    @test get_nn_m(1,1;hidden=[1]  ,model_type=:m3w,dropout_prob=0  ) isa Chain
    @test get_nn_m(1,1;hidden=[1]  ,model_type=:m3w,dropout_prob=0.5) isa Chain
    @test get_nn_m(1,1;hidden=[1,1],model_type=:m3w,dropout_prob=0  ) isa Chain
    @test get_nn_m(1,1;hidden=[1,1],model_type=:m3w,dropout_prob=0.5) isa Chain
    @test get_nn_m(1,1;hidden=[1]  ,model_type=:m3tf,tf_layer_type=:prelayer ,tf_norm_type=:layer,N_tf_head=1) isa Chain
    @test get_nn_m(1,1;hidden=[1]  ,model_type=:m3tf,tf_layer_type=:postlayer,tf_norm_type=:layer,N_tf_head=1) isa Chain
    @test get_nn_m(1,1;hidden=[1,1],model_type=:m3tf,tf_layer_type=:prelayer ,tf_norm_type=:batch,N_tf_head=1) isa Chain
    @test get_nn_m(1,1;hidden=[1,1],model_type=:m3tf,tf_layer_type=:postlayer,tf_norm_type=:none ,N_tf_head=1) isa Chain
    @test_throws ErrorException get_nn_m(1,1;hidden=[1,1,1,1])
    @test_throws ErrorException get_nn_m(1,1;hidden=[1,1],skip_con=true)
    @test_throws ErrorException get_nn_m(1,1;hidden=[     ],model_type=:m3w)
    @test_throws ErrorException get_nn_m(1,1;hidden=[1,1,1],model_type=:m3w)
    @test_throws AssertionError get_nn_m(1,1;hidden=[     ],model_type=:m3tf,N_tf_head=1)
    @test_throws ErrorException get_nn_m(1,1;hidden=[1,1,1],model_type=:m3tf,N_tf_head=1)
    @test_throws ErrorException get_nn_m(1,1;hidden=[1],model_type=:m3tf,tf_layer_type=:test,N_tf_head=1)
    @test_throws ErrorException get_nn_m(1,1;hidden=[1],model_type=:m3tf,tf_norm_type =:test,N_tf_head=1)
end

m = get_nn_m(3,1;hidden=[1])
weights = Flux.Params(Flux.trainables(m))
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
        @test norm_sets(x    ;norm_type=norm_type,no_norm=2) isa NTuple{3,Matrix}
        @test norm_sets(x,x  ;norm_type=norm_type,no_norm=2) isa NTuple{4,Matrix}
        @test norm_sets(x,x,x;norm_type=norm_type,no_norm=2) isa NTuple{5,Matrix}
        @test norm_sets(y    ;norm_type=norm_type) isa NTuple{3,Vector}
        @test norm_sets(y,y  ;norm_type=norm_type) isa NTuple{4,Vector}
        @test norm_sets(y,y,y;norm_type=norm_type) isa NTuple{5,Vector}
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
    @test sum(get_ind(xyz,lines[2],df_line;l_window=15)) ≈ 45
    @test sum.(get_ind(xyz,lines[2],df_line;splits=(0.5,0.5),l_window=15)) == (15,15)
    @test sum(get_ind(xyz,lines,df_line)) ≈ 50
    @test sum.(get_ind(xyz,lines,df_line;splits=(0.5,0.5))) == (25,25)
    @test sum.(get_ind(xyz,lines,df_line;splits=(0.7,0.2,0.1))) == (35,10,5)
    @test_throws AssertionError get_ind(xyz,lines,df_line;splits=(1,1,1))
    @test_throws AssertionError get_ind(xyz,lines,df_line;splits=(1,0,0,0))
end

@testset "chunk_data tests" begin
    @test chunk_data(x,y,size(x,1)) == ([x'],[y])
    @test chunk_data([1:4 1:4],1:4,2) == ([[1 2; 1 2],[3 4; 3 4]],[[1,2],[3,4]])
end

m_rnn = Chain(GRU(3 => 1), Dense(1 => 1))

@testset "predict_rnn tests" begin
    @test predict_rnn_full(m_rnn,x) isa Vector
    @test predict_rnn_windowed(m_rnn,x,3) isa Vector
    @test predict_rnn_full(m_rnn,x) == predict_rnn_windowed(m_rnn,x,size(x,1))
end

# (model,data_norms,_,_) = krr_fit(x,y)

# @testset "krr tests" begin
#     @test krr_fit(x,y)[2:4] == krr_fit( x,y;data_norms      )[2:4]
#     @test krr_fit(x,y)[3:4] == krr_test(x,y,data_norms,model)[1:2]
# end

features = [:f1,:f2,:f3]
(df_shap,baseline_shap) = eval_shapley(m,x,features)

@testset "shapley & gsa tests" begin
    @test MagNav.predict_shapley(m,DataFrame(x,features)) isa DataFrame
    @test eval_shapley(m,x,features) isa Tuple{DataFrame,Real}
    @test plot_shapley(df_shap,baseline_shap) isa Plots.Plot
    @test eval_gsa(m,x) isa Vector
end

@testset "get_igrf tests" begin
    @test get_igrf(xyz,ind) isa Vector
    @test get_igrf(xyz,ind;frame=:nav,norm_igrf=true) isa Vector
end

vec_nav  = randn(3)
vec_body = randn(3)
igrf_nav = vec_nav ./ norm(vec_nav)
Cnb      = xyz.ins.Cnb[:,:,ind]

@testset "project_vec_to_2d tests" begin
    @test_throws AssertionError project_vec_to_2d(vec_nav,[1,0,0],[1,0,0])
    @test_throws AssertionError project_vec_to_2d(vec_nav,[1,1,0],[0,1,0])
    @test_throws AssertionError project_vec_to_2d(vec_nav,[1,0,0],[1,1,0])
    @test project_vec_to_2d(vec_nav,[1,0,0],[0,1,0]) isa Vector
end

@testset "project_body_field_to_2d_igrf tests" begin
    @test project_body_field_to_2d_igrf(vec_body,igrf_nav,Cnb[:,:,1]) isa Vector

    # create "north" vector in navigation frame
    north_vec_nav = vec([1.0, 0.0, 0.0])

    # check that north in 3D stays north in 2D (repeated for cardinal directions)
    north_vec_body = [Cnb[:,:,i]'*north_vec_nav for i=1:size(Cnb)[3]];
    n2ds = [project_body_field_to_2d_igrf(north_vec_body[i], north_vec_nav, Cnb[:,:,i])
            for i=1:size(Cnb)[3]]
    @test all(map(v -> v ≈ [1.0, 0.0], n2ds))

    east_vec_nav  = vec([0.0, 1.0, 0.0])
    east_vec_body = [Cnb[:,:,i]'*east_vec_nav for i=1:size(Cnb)[3]]
    e2ds = [project_body_field_to_2d_igrf(east_vec_body[i], north_vec_nav, Cnb[:,:,i])
            for i=1:size(Cnb)[3]]
    @test all(map(v -> v ≈ [0.0, 1.0], e2ds))

    west_vec_nav  = vec([0.0, -1.0, 0.0])
    west_vec_body = [Cnb[:,:,i]'*west_vec_nav for i=1:size(Cnb)[3]]
    w2ds = [project_body_field_to_2d_igrf(west_vec_body[i], north_vec_nav, Cnb[:,:,i])
            for i=1:size(Cnb)[3]]
    @test all(map(v -> v ≈ [0.0, -1.0], w2ds))

    south_vec_nav  = vec([-1.0, 0.0, 0.0])
    south_vec_body = [Cnb[:,:,i]'*south_vec_nav for i=1:size(Cnb)[3]]
    s2ds = [project_body_field_to_2d_igrf(south_vec_body[i], north_vec_nav, Cnb[:,:,i])
            for i=1:size(Cnb)[3]]
    @test all(map(v -> v ≈ [-1.0, 0.0], s2ds))
end

@testset "get_optimal_rotation_matrix tests" begin
    @test_throws AssertionError get_optimal_rotation_matrix([1 0],vec_body')
    @test_throws AssertionError get_optimal_rotation_matrix([1 0],[1 0 0])
    @test_throws AssertionError get_optimal_rotation_matrix([1 0 0],[1 0])
    @test size(get_optimal_rotation_matrix(vec_nav',vec_body')) == (3,3)
end

x = [-1,0,1]
xlim = (-2.0,2)

@testset "expand_range tests" begin
    @test MagNav.expand_range(x,xlim,true)[1] == -3:3
    @test MagNav.expand_range(x,xlim,false)[2] == 2:4
end

(B_unit,Bt,B_vec_dot) = create_TL_A(xyz.flux_a,ind;terms=[:p],return_B=true)
B_vec = B_unit .* Bt
(TL_coef_p,TL_coef_i,TL_coef_e) = MagNav.TL_vec2mat(TL_a_1,[:p,:i,:e])
(TL_aircraft,TL_perm,TL_induced,TL_eddy) =
     MagNav.get_TL_aircraft_vec(B_vec',B_vec_dot',TL_coef_p,TL_coef_i,TL_coef_e;
                                return_parts=true)
y_nn = zeros(3,50)
y = y_hat = zeros(50)

mag_gif = joinpath(@__DIR__,"comp_xai")

@testset "gif_animation_m3 tests" begin
    ENV["GKSwstype"] = "100"
    @test gif_animation_m3(TL_perm, TL_induced, TL_eddy,
                           TL_aircraft, B_unit', y_nn, y, y_hat, xyz,
                           xyz.ins.lat[ind], xyz.ins.lon[ind];
                           ind=ind, save_plot=true, mag_gif=mag_gif) isa Plots.AnimatedGif
end

mag_gif = MagNav.add_extension(mag_gif,".gif")
rm(mag_gif)
