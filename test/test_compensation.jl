using MagNav, Test, MAT, DataFrames, Flux, Statistics, Zygote
using DelimitedFiles: readdlm, writedlm

generate = false       # to generate comp_csv
silent   = true        # to suppress print outs
df       = DataFrame() # empty dataframe

flight   = :Flt1007
line     = 1007.06
xyz_type = :XYZ20
map_name = :Renfrew_395
xyz_h5   = MagNav.sgl_2020_train()*"/$(flight)_train.h5"
map_h5   = MagNav.ottawa_area_maps()*"/$(map_name).h5"

xyz = get_XYZ20(xyz_h5;tt_sort=true,silent)
ind = xyz.line .== line
ind[findall(ind)[26:end]] .= false

mapS   = map_trim(get_map(map_h5),xyz.traj(ind))
map_h5 = joinpath(@__DIR__,"test_compensation.h5")
save_map(mapS,map_h5)

t_start = xyz.traj.tt[ind][1]
t_end   = xyz.traj.tt[ind][end]

df_line = DataFrame(flight   = [flight],
                    line     = [line],
                    t_start  = [t_start],
                    t_end    = [t_end],
                    map_name = [map_name])

df_flight = DataFrame(flight   = flight,
                      xyz_type = xyz_type,
                      xyz_set  = 1,
                      xyz_h5   = xyz_h5)

df_map = DataFrame(map_name = map_name,
                   map_h5   = map_h5)

terms_p     = [:p]
terms_pi    = [:p,:i]
terms_pie   = [:p,:i,:e]
terms_pieb  = [:p,:i,:e,:b]
TL_coef_pie = zeros(18)
batchsize   = 5
epoch_adam  = 11

comp_params_1   = NNCompParams(model_type=:m1  ,terms=terms_p,
                               terms_A=terms_pie,TL_coef=TL_coef_pie,
                               epoch_adam=epoch_adam,batchsize=batchsize)
comp_params_2a  = NNCompParams(model_type=:m2a ,terms=terms_p,
                               terms_A=terms_pie,TL_coef=TL_coef_pie,
                               epoch_adam=epoch_adam,batchsize=batchsize)
comp_params_2b  = NNCompParams(model_type=:m2b ,terms=terms_p,
                               terms_A=terms_pie,TL_coef=TL_coef_pie,
                               epoch_adam=epoch_adam,batchsize=batchsize)
comp_params_2c  = NNCompParams(model_type=:m2c ,terms=terms_p,
                               terms_A=terms_pie,TL_coef=TL_coef_pie,
                               epoch_adam=epoch_adam,batchsize=batchsize)
comp_params_2d  = NNCompParams(model_type=:m2d ,terms=terms_p,
                               terms_A=terms_pie,TL_coef=TL_coef_pie,
                               epoch_adam=epoch_adam,batchsize=batchsize)
comp_params_3tl = NNCompParams(model_type=:m3tl,terms=terms_pi,
                               terms_A=terms_pieb,TL_coef=[TL_coef_pie;0],
                               epoch_adam=epoch_adam,batchsize=batchsize)
comp_params_3s  = NNCompParams(model_type=:m3s ,terms=terms_pi,
                               terms_A=terms_pieb,TL_coef=[TL_coef_pie;0],
                               epoch_adam=epoch_adam,batchsize=batchsize)
comp_params_3v  = NNCompParams(model_type=:m3v ,terms=terms_pi,
                               terms_A=terms_pieb,TL_coef=[TL_coef_pie;0],
                               epoch_adam=epoch_adam,batchsize=batchsize)
comp_params_3sc = NNCompParams(model_type=:m3sc,terms=terms_pi,
                               terms_A=terms_pie,TL_coef=TL_coef_pie,
                               epoch_adam=epoch_adam,batchsize=batchsize)
comp_params_3vc = NNCompParams(model_type=:m3vc,terms=terms_pi,
                               terms_A=terms_pie,TL_coef=TL_coef_pie,
                               epoch_adam=epoch_adam,batchsize=batchsize)
comp_params_3w =  NNCompParams(model_type=:m3w,terms=terms_pi,
                               terms_A=terms_pie,TL_coef=TL_coef_pie,
                               epoch_adam=epoch_adam,batchsize=batchsize,
                               frac_train=0.6)
comp_params_3tf = NNCompParams(model_type=:m3tf,terms=terms_pi,
                               terms_A=terms_pie,TL_coef=TL_coef_pie,
                               epoch_adam=epoch_adam,batchsize=batchsize,
                               frac_train=0.6)

comp_params_TL         = LinCompParams(model_type=:TL,y_type=:a)
comp_params_mod_TL     = LinCompParams(model_type=:mod_TL,y_type=:a)
comp_params_map_TL     = LinCompParams(model_type=:map_TL,y_type=:a,sub_igrf=true)
comp_params_elasticnet = LinCompParams(model_type=:elasticnet,y_type=:a)
comp_params_plsr       = LinCompParams(model_type=:plsr,y_type=:a,k_plsr=1)

drop_fi_bson = joinpath(@__DIR__,"drop_fi")
drop_fi_csv  = joinpath(@__DIR__,"drop_fi")
perm_fi_csv  = joinpath(@__DIR__,"perm_fi")

comp_params_1_drop  = NNCompParams(comp_params_1,drop_fi=true,
                      drop_fi_bson=drop_fi_bson,drop_fi_csv=drop_fi_csv)
comp_params_1_perm  = NNCompParams(comp_params_1,perm_fi=true,
                      perm_fi_csv=perm_fi_csv)
comp_params_2c_drop = NNCompParams(comp_params_2c,drop_fi=true,
                      drop_fi_bson=drop_fi_bson,drop_fi_csv=drop_fi_csv)
comp_params_2c_perm = NNCompParams(comp_params_2c,perm_fi=true,
                      perm_fi_csv=perm_fi_csv)
comp_params_3s_drop = NNCompParams(comp_params_3s,drop_fi=true,
                      drop_fi_bson=drop_fi_bson,drop_fi_csv=drop_fi_csv)
comp_params_3s_perm = NNCompParams(comp_params_3s,perm_fi=true,
                      perm_fi_csv=perm_fi_csv)

comp_params_nn_bad      = NNCompParams(model_type=:test)
comp_params_m3_bad      = NNCompParams(comp_params_3s,y_type=:e)
comp_params_lin_bad     = LinCompParams(model_type=:test)
comp_params_nn_bad_drop = NNCompParams(model_type=:test,drop_fi=true,
                                       drop_fi_bson=drop_fi_bson,
                                       drop_fi_csv=drop_fi_csv)

comp_params_list = [comp_params_1,
                    comp_params_2a,
                    comp_params_2b,
                    comp_params_2c,
                    comp_params_2d,
                    comp_params_3tl,
                    comp_params_3s,
                    comp_params_3v,
                    comp_params_3sc,
                    comp_params_3vc,
                    comp_params_3w,
                    comp_params_3tf,
                    comp_params_TL,
                    comp_params_mod_TL,
                    comp_params_map_TL,
                    comp_params_elasticnet,
                    comp_params_plsr,
                    comp_params_1_drop,
                    comp_params_1_perm,
                    comp_params_2c_drop,
                    comp_params_2c_perm,
                    comp_params_3s_drop,
                    comp_params_3s_perm]

major    = VERSION.major
minor    = VERSION.minor
comp_csv = joinpath(@__DIR__,"test_data/comp_err_$major.$minor.csv")

if generate
    comp_err = zeros(Float32,length(comp_params_list),2)
else
    comp_err = readdlm(comp_csv,',')
end

@testset "comp_train_test tests" begin
    atol = minor == 6 ? 5f-6 : 5f-7
    for (i,comp_params) in enumerate(comp_params_list)
        comp_params_ = deepcopy(comp_params)
        (err_train_1,err_test_1) = comp_train_test(comp_params,xyz,xyz,ind,ind,
                                                   mapS,mapS;silent)[[4,7]]
        (err_train_2,err_test_2) = comp_train_test(comp_params,line,line,
                                                   df_line,df_flight,df_map;
                                                   silent)[[4,7]]
        @test std(err_train_1)   ≈ std(err_train_2)
        @test std(err_test_1 )   ≈ std(err_test_2 )
        if generate # store compensation reproducibility results
            comp_err[i,1] = std(err_train_1)
            comp_err[i,2] = std(err_test_1 )
        else # test compensation reproducibility
            @test isapprox(comp_err[i,1],std(err_train_1),atol=atol)
            @test isapprox(comp_err[i,2],std(err_test_1 ),atol=atol)
        end
        @test MagNav.compare_fields(comp_params_,comp_params;silent) == 0 # no mutating
    end
end

generate && (writedlm(comp_csv,comp_err,','))

comp_params_1   = comp_train(comp_params_1  ,xyz,ind;silent)[1]
comp_params_2a  = comp_train(comp_params_2a ,xyz,ind;silent)[1]
comp_params_2b  = comp_train(comp_params_2b ,xyz,ind;silent)[1]
comp_params_2c  = comp_train(comp_params_2c ,xyz,ind;silent)[1]
comp_params_2d  = comp_train(comp_params_2d ,xyz,ind;silent)[1]
comp_params_3tl = comp_train(comp_params_3tl,xyz,ind;silent)[1]
comp_params_3s  = comp_train(comp_params_3s ,xyz,ind;silent)[1]
comp_params_3v  = comp_train(comp_params_3v ,xyz,ind;silent)[1]
comp_params_3sc = comp_train(comp_params_3sc,xyz,ind;silent)[1]
comp_params_3vc = comp_train(comp_params_3vc,xyz,ind;silent)[1]

@testset "comp_train (re-train) tests" begin
    @test std(comp_train(comp_params_1 ,xyz,ind;silent)[end-1]) < 1
    @test std(comp_train(comp_params_2a,xyz,ind;silent)[end-1]) < 1
    @test std(comp_train(comp_params_2b,xyz,ind;silent)[end-1]) < 1
    @test std(comp_train(comp_params_2c,xyz,ind;silent)[end-1]) < 1
    @test std(comp_train(comp_params_2d,xyz,ind;silent)[end-1]) < 1
    @test std(comp_train(comp_params_1  ,line,df_line,df_flight,df;
                         silent)[end-1]) < 1
    @test std(comp_train(comp_params_2a ,line,df_line,df_flight,df;
                         silent)[end-1]) < 1
    @test std(comp_train(comp_params_2b ,line,df_line,df_flight,df;
                         silent)[end-1]) < 1
    @test std(comp_train(comp_params_2c ,line,df_line,df_flight,df;
                         silent)[end-1]) < 1
    @test std(comp_train(comp_params_2d ,line,df_line,df_flight,df;
                         silent)[end-1]) < 1
    @test std(comp_train(comp_params_3tl,line,df_line,df_flight,df;
                         silent)[end-1]) < 1
    @test std(comp_train(comp_params_3s ,line,df_line,df_flight,df;
                         silent)[end-1]) < 1
    @test std(comp_train(comp_params_3v ,line,df_line,df_flight,df;
                         silent)[end-1]) < 1
    @test std(comp_train(comp_params_3sc,line,df_line,df_flight,df;
                         silent)[end-1]) < 50
    @test std(comp_train(comp_params_3vc,line,df_line,df_flight,df;
                         silent)[end-1]) < 50
end

@testset "comp_m2bc_test tests" begin
    @test std(comp_m2bc_test(comp_params_2b,line,df_line,df_flight,df;
                             silent)[end-1]) < 1
    @test std(comp_m2bc_test(comp_params_2c,line,df_line,df_flight,df;
                             silent)[end-1]) < 1
end

@testset "comp_m3_test tests" begin
    @test_throws AssertionError comp_m3_test(comp_params_m3_bad,line,
                                             df_line,df_flight,df;silent)[end-1]
    @test_throws AssertionError comp_m3_test(comp_params_3tl,line,
                                             df_line,df_flight,df;silent)[end-1]
    @test std(comp_m3_test(comp_params_3s ,line,df_line,df_flight,df;
                           silent)[end-1]) < 1
    @test std(comp_m3_test(comp_params_3v ,line,df_line,df_flight,df;
                           silent)[end-1]) < 1
    @test std(comp_m3_test(comp_params_3sc,line,df_line,df_flight,df;
                           silent)[end-1]) < 50
    @test std(comp_m3_test(comp_params_3vc,line,df_line,df_flight,df;
                           silent)[end-1]) < 50
end

epoch_lbfgs = 1
k_pca       = 5
frac_train  = 1

comp_params_1   = NNCompParams(model_type  = :m1,
                               y_type      = :a,
                               terms       = terms_p,
                               terms_A     = terms_pie,
                               sub_igrf    = true,
                               TL_coef     = TL_coef_pie,
                               epoch_lbfgs = epoch_lbfgs,
                               batchsize   = batchsize,
                               frac_train  = frac_train,
                               k_pca       = k_pca)
comp_params_2a  = NNCompParams(model_type  = :m2a,
                               y_type      = :a,
                               terms       = terms_p,
                               terms_A     = terms_pie,
                               sub_igrf    = true,
                               TL_coef     = TL_coef_pie,
                               epoch_lbfgs = epoch_lbfgs,
                               batchsize   = batchsize,
                               frac_train  = frac_train,
                               k_pca       = k_pca)
comp_params_2b  = NNCompParams(model_type  = :m2b,
                               y_type      = :a,
                               terms       = terms_p,
                               terms_A     = terms_pie,
                               sub_igrf    = true,
                               TL_coef     = TL_coef_pie,
                               epoch_lbfgs = epoch_lbfgs,
                               batchsize   = batchsize,
                               frac_train  = frac_train,
                               k_pca       = k_pca)
comp_params_2c  = NNCompParams(model_type  = :m2c,
                               y_type      = :e,
                               terms       = terms_p,
                               terms_A     = terms_pie,
                               TL_coef     = TL_coef_pie,
                               epoch_lbfgs = epoch_lbfgs,
                               batchsize   = batchsize,
                               frac_train  = frac_train,
                               k_pca       = k_pca)
comp_params_2d  = NNCompParams(model_type  = :m2d,
                               y_type      = :e,
                               terms       = terms_p,
                               terms_A     = terms_pie,
                               TL_coef     = TL_coef_pie,
                               epoch_lbfgs = epoch_lbfgs,
                               batchsize   = batchsize,
                               frac_train  = frac_train,
                               k_pca       = k_pca)

k_pca_big = 100

comp_params_3tl = NNCompParams(comp_params_3tl,
                               data_norms  = NNCompParams().data_norms,
                               epoch_lbfgs = epoch_lbfgs,
                               frac_train  = frac_train,
                               k_pca       = k_pca_big)

comp_params_3s  = NNCompParams(comp_params_3s,
                               epoch_lbfgs = epoch_lbfgs,
                               frac_train  = frac_train,
                               k_pca       = k_pca_big)

comp_params_3v  = NNCompParams(comp_params_3v,
                               epoch_lbfgs = epoch_lbfgs,
                               frac_train  = frac_train,
                               k_pca       = k_pca_big)

comp_params_3sc = NNCompParams(comp_params_3sc,
                               y_type      = :a,
                               terms_A     = terms_pieb,
                               TL_coef     = [comp_params_3sc.TL_coef;0],
                               epoch_lbfgs = epoch_lbfgs,
                               frac_train  = frac_train,
                               k_pca       = k_pca_big)

comp_params_3vc = NNCompParams(comp_params_3vc,
                               y_type      = :a,
                               terms_A     = terms_pieb,
                               TL_coef     = [comp_params_3vc.TL_coef;0],
                               epoch_lbfgs = epoch_lbfgs,
                               frac_train  = frac_train,
                               k_pca       = k_pca_big)

x = [1:5;][:,:]
y = [1:5;]

@testset "comp_train tests" begin
    @test std(MagNav.elasticnet_fit(x,y;λ=0.01,silent)[end]) < 1
    @test isone(MagNav.plsr_fit(x,y,size(x,2)+1;return_set=true,silent)[:,:,1])
    @test std(comp_train(comp_params_1  ,xyz,ind;
                         xyz_test=xyz,ind_test=ind,silent)[end-1]) < 1
    @test std(comp_train(comp_params_1  ,[xyz,xyz],[ind,ind];
                         xyz_test=xyz,ind_test=ind,silent)[end-1]) < 1
    @test std(comp_train(comp_params_2a ,[xyz,xyz],[ind,ind];
                         xyz_test=xyz,ind_test=ind,silent)[end-1]) < 1
    @test std(comp_train(comp_params_2b ,[xyz,xyz],[ind,ind];
                         xyz_test=xyz,ind_test=ind,silent)[end-1]) < 1
    @test std(comp_train(comp_params_2c ,[xyz,xyz],[ind,ind];
                         xyz_test=xyz,ind_test=ind,silent)[end-1]) < 1
    @test std(comp_train(comp_params_2d ,[xyz,xyz],[ind,ind];
                         xyz_test=xyz,ind_test=ind,silent)[end-1]) < 1
    @test std(comp_train(comp_params_3tl,[xyz,xyz],[ind,ind];
                         xyz_test=xyz,ind_test=ind,silent)[end-1]) < 1
    @test std(comp_train(comp_params_3s ,[xyz,xyz],[ind,ind];
                         xyz_test=xyz,ind_test=ind,silent)[end-1]) < 1
    @test std(comp_train(comp_params_3v ,[xyz,xyz],[ind,ind];
                         xyz_test=xyz,ind_test=ind,silent)[end-1]) < 1
    @test std(comp_train(comp_params_3sc,[xyz,xyz],[ind,ind];
                         xyz_test=xyz,ind_test=ind,silent)[end-1]) < 50
    @test std(comp_train(comp_params_3vc,[xyz,xyz],[ind,ind];
                         xyz_test=xyz,ind_test=ind,silent)[end-1]) < 50
    @test std(comp_train(comp_params_TL,[xyz,xyz],[ind,ind];
                         xyz_test=xyz,ind_test=ind,silent)[end-1]) < 1
    @test std(comp_train(comp_params_mod_TL,[xyz,xyz],[ind,ind];
                         xyz_test=xyz,ind_test=ind,silent)[end-1]) < 1
    @test std(comp_train(comp_params_map_TL,[xyz,xyz],[ind,ind],mapS;
                         xyz_test=xyz,ind_test=ind,silent)[end-1]) < 1
    @test std(comp_train(comp_params_elasticnet,[xyz,xyz],[ind,ind];
                         xyz_test=xyz,ind_test=ind,silent)[end-1]) < 1
    @test std(comp_train(comp_params_plsr,[xyz,xyz],[ind,ind];
                         xyz_test=xyz,ind_test=ind,silent)[end-1]) < 1
    @test std(comp_train(comp_params_1_drop,[xyz,xyz],[ind,ind];
                         xyz_test=xyz,ind_test=ind,silent=false)[end-1]) < 1
    @test std(comp_train(comp_params_2c_drop,[xyz,xyz],[ind,ind];
                         xyz_test=xyz,ind_test=ind,silent=false)[end-1]) < 1
    @test std(comp_train(comp_params_3s_drop,[xyz,xyz],[ind,ind];
                         xyz_test=xyz,ind_test=ind,silent=false)[end-1]) < 1
    @test_throws ErrorException comp_train(comp_params_nn_bad,
                                           [xyz,xyz],[ind,ind];silent)
    @test_throws AssertionError comp_train(comp_params_m3_bad,
                                           [xyz,xyz],[ind,ind];silent)
    @test_throws ErrorException comp_train(comp_params_lin_bad,
                                           [xyz,xyz],[ind,ind];silent)
    @test_throws ErrorException comp_train(comp_params_nn_bad_drop,
                                           [xyz,xyz],[ind,ind];silent)
    @test_throws ErrorException comp_train(comp_params_nn_bad,
                                           xyz,ind;silent)
    @test_throws AssertionError comp_train(comp_params_m3_bad,
                                           xyz,ind;silent)
    @test_throws ErrorException comp_train(comp_params_lin_bad,
                                           xyz,ind;silent)
    @test_throws ErrorException comp_train(comp_params_nn_bad_drop,
                                           xyz,ind;silent)
    @test_throws ErrorException comp_train(comp_params_nn_bad,line,
                                           df_line,df_flight,df;silent)
    @test_throws AssertionError comp_train(comp_params_m3_bad,line,
                                           df_line,df_flight,df;silent)
    @test_throws ErrorException comp_train(comp_params_lin_bad,line,
                                           df_line,df_flight,df;silent)
    @test_throws ErrorException comp_train(comp_params_nn_bad_drop,line,
                                           df_line,df_flight,df;silent)
end

@testset "comp_test tests" begin
    @test std(comp_test(comp_params_3sc,xyz,ind;                  silent)[end-1]) < 50
    @test std(comp_test(comp_params_3vc,line,df_line,df_flight,df;silent)[end-1]) < 50
    @test_throws ErrorException comp_test(comp_params_nn_bad,xyz,ind;silent)
    @test_throws ErrorException comp_test(comp_params_lin_bad,xyz,ind;silent)
    @test_throws ErrorException comp_test(comp_params_nn_bad_drop,xyz,ind;silent)
    @test_throws ErrorException comp_test(comp_params_nn_bad,line,
                                          df_line,df_flight,df;silent)
    @test_throws ErrorException comp_test(comp_params_lin_bad,line,
                                          df_line,df_flight,df;silent)
    @test_throws ErrorException comp_test(comp_params_nn_bad_drop,line,
                                          df_line,df_flight,df;silent)
end

terms_pi5e8 = [:p,:i5,:e8]
terms_pi3e3 = [:p,:i3,:e3]

@testset "TL_coef extraction tests" begin
    for terms in [terms_pi,terms_pie,terms_pi5e8,terms_pi3e3]
        TL_coef_1 = 30000*rand(size(create_TL_A(xyz.flux_a,1:5;terms=terms),2))
        (TL_coef_p_1,TL_coef_i_1,TL_coef_e_1) = MagNav.TL_vec2mat(TL_coef_1,terms)
        TL_coef_2 = MagNav.TL_mat2vec(TL_coef_p_1,TL_coef_i_1,TL_coef_e_1,terms)
        (TL_coef_p_2,TL_coef_i_2,TL_coef_e_2) = MagNav.TL_vec2mat(TL_coef_2,terms)
        @test TL_coef_1   ≈ TL_coef_2
        @test TL_coef_p_1 ≈ TL_coef_p_2
        @test TL_coef_i_1 ≈ TL_coef_i_2
        if any([:eddy,:e,:eddy9,:e9,:eddy8,:e8,:eddy3,:e3] .∈ (terms,))
            @test TL_coef_e_1 ≈ TL_coef_e_2
        end
        B_vec = B_vec_dot = ones(3,1)
        TL_aircraft = MagNav.get_TL_aircraft_vec(B_vec,B_vec_dot,TL_coef_p_1,
                                                 TL_coef_i_1,TL_coef_e_1)
        @test TL_aircraft isa Matrix
    end
end

@testset "get_split tests" begin
    @test MagNav.get_split(2,0.5,:none)[1][1] in [1,2]
    @test MagNav.get_split(2,1  ,:none)[1] == 1:2
    @test MagNav.get_split(2,0.5,:sliding,l_window=1)[1] == 1:1
    @test MagNav.get_split(2,1  ,:sliding,l_window=1)[1] == 1:2
    @test MagNav.get_split(2,0.5,:contiguous,l_window=1)[1][1] in [1,2]
    @test MagNav.get_split(2,1  ,:contiguous,l_window=1)[1] in [[1,2],[2,1]]
    @test_throws ErrorException MagNav.get_split(1,1,:test)
end

x_norm = ones(3,3) ./ 3
y      = ones(3)

@testset "linear_test tests" begin
    @test MagNav.linear_test(x_norm,y,[0],[1],(y,[0]);silent)[1] == y
end

@testset "print_time tests" begin
    @test MagNav.print_time(30) isa Nothing
    @test MagNav.print_time(90) isa Nothing
end

drop_fi_bson = MagNav.remove_extension(drop_fi_bson,".bson")
drop_fi_csv  = MagNav.add_extension(drop_fi_csv,".csv")
perm_fi_csv  = MagNav.add_extension(perm_fi_csv,".csv")

[rm(drop_fi_bson*"_$i.bson") for i = 1:10]
rm(drop_fi_csv)
rm(perm_fi_csv)
rm(map_h5)
