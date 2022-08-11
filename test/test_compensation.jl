using MagNav, Test, MAT, DataFrames, Flux, Statistics, Zygote

flight = :Flt1003
xyz_h5 = string(MagNav.sgl_2020_train(),"/$(flight)_train.h5")
xyz    = get_XYZ20(xyz_h5;tt_sort=true,silent=true)
line_train = unique(xyz.line)[2]
line_test  = unique(xyz.line)[3]
ind_train  = xyz.line .== line_train
ind_test   = xyz.line .== line_test
ind_train[findfirst(ind_train)+50:end] .= false
ind_test[ findfirst(ind_test )+50:end] .= false

line    = [line_train,line_test]
t_start = [xyz.traj.tt[ind_train][1],xyz.traj.tt[ind_test][1]]
t_end   = [xyz.traj.tt[ind_train][end],xyz.traj.tt[ind_test][end]]
test    = [false,true]

df_line = DataFrame(flight  = [flight,flight],
                    line    = line,
                    t_start = t_start,
                    t_end   = t_end,
                    test    = test)

df_flight = DataFrame(flight   = flight,
                      xyz_type = :XYZ20,
                      xyz_set  = 1,
                      xyz_h5   = xyz_h5)

terms     = [:p]
terms_A   = [:p,:i,:e]
batchsize = 5

comp_params_1  = MagNav.NNCompParams(model_type=:m1 ,terms=terms,
                                     terms_A=terms_A,batchsize=batchsize)
comp_params_2a = MagNav.NNCompParams(model_type=:m2a,terms=terms,
                                     terms_A=terms_A,batchsize=batchsize)
comp_params_2b = MagNav.NNCompParams(model_type=:m2b,terms=terms,
                                     terms_A=terms_A,batchsize=batchsize)
comp_params_2c = MagNav.NNCompParams(model_type=:m2c,terms=terms,
                                     terms_A=terms_A,batchsize=batchsize)
comp_params_2d = MagNav.NNCompParams(model_type=:m2d,terms=terms,
                                     terms_A=terms_A,batchsize=batchsize)

comp_params_nn_bad      = MagNav.NNCompParams( model_type=:test)
comp_params_lin_bad     = MagNav.LinCompParams(model_type=:test)
comp_params_nn_bad_drop = MagNav.NNCompParams( model_type=:test,drop_fi=true)

x = [1:5;;]
y = [1:5;]

@testset "comp_train tests" begin
    @test std(comp_train(xyz,ind_train;comp_params=comp_params_1 )[end-1]) < 1
    @test std(comp_train(xyz,ind_train;comp_params=comp_params_2a)[end-1]) < 1
    @test std(comp_train(xyz,ind_train;comp_params=comp_params_2b)[end-1]) < 1
    @test std(comp_train(xyz,ind_train;comp_params=comp_params_2c)[end-1]) < 1
    @test std(comp_train(xyz,ind_train;comp_params=comp_params_2d)[end-1]) < 1
    @test std(comp_train(line_train,df_line,df_flight,DataFrame(),
                         comp_params_1 )[end-1]) < 1
    @test std(comp_train(line_train,df_line,df_flight,DataFrame(),
                         comp_params_2a)[end-1]) < 1
    @test std(comp_train(line_train,df_line,df_flight,DataFrame(),
                         comp_params_2b)[end-1]) < 1
    @test std(comp_train(line_train,df_line,df_flight,DataFrame(),
                         comp_params_2c)[end-1]) < 1
    @test std(comp_train(line_train,df_line,df_flight,DataFrame(),
                         comp_params_2d)[end-1]) < 1
    @test isone(plsr_fit(x,y;return_set=true)[:,:,1])
    @test std(elasticnet_fit(x,y;λ=0.01)[end]) < 1
    @test_throws ErrorException comp_train(xyz,ind_train;
                                           comp_params=comp_params_nn_bad)
    @test_throws ErrorException comp_train(xyz,ind_train;
                                           comp_params=comp_params_lin_bad)
    @test_throws ErrorException comp_train(xyz,ind_train;
                                           comp_params=comp_params_nn_bad_drop)
    @test_throws ErrorException comp_train(line_train,df_line,df_flight,DataFrame(),
                                           comp_params_nn_bad)
    @test_throws ErrorException comp_train(line_train,df_line,df_flight,DataFrame(),
                                           comp_params_lin_bad)
    @test_throws ErrorException comp_train(line_train,df_line,df_flight,DataFrame(),
                                           comp_params_nn_bad_drop)
end

comp_params_1  = comp_train(xyz,ind_train;comp_params=comp_params_1 )[1]
comp_params_2a = comp_train(xyz,ind_train;comp_params=comp_params_2a)[1]
comp_params_2b = comp_train(xyz,ind_train;comp_params=comp_params_2b)[1]
comp_params_2c = comp_train(xyz,ind_train;comp_params=comp_params_2c)[1]
comp_params_2d = comp_train(xyz,ind_train;comp_params=comp_params_2d)[1]

comp_train(xyz,ind_train;comp_params=MagNav.NNCompParams(comp_params_1,drop_fi=true))

@testset "comp_train (re-train) tests" begin
    @test std(comp_train(xyz,ind_train;comp_params=comp_params_1 )[end-1]) < 1
    @test std(comp_train(xyz,ind_train;comp_params=comp_params_2a)[end-1]) < 1
    @test std(comp_train(xyz,ind_train;comp_params=comp_params_2b)[end-1]) < 1
    @test std(comp_train(xyz,ind_train;comp_params=comp_params_2c)[end-1]) < 1
    @test std(comp_train(xyz,ind_train;comp_params=comp_params_2d)[end-1]) < 1
    @test std(comp_train(line_train,df_line,df_flight,DataFrame(),
                         comp_params_1 )[end-1]) < 1
    @test std(comp_train(line_train,df_line,df_flight,DataFrame(),
                         comp_params_2a)[end-1]) < 1
    @test std(comp_train(line_train,df_line,df_flight,DataFrame(),
                         comp_params_2b)[end-1]) < 1
    @test std(comp_train(line_train,df_line,df_flight,DataFrame(),
                         comp_params_2c)[end-1]) < 1
    @test std(comp_train(line_train,df_line,df_flight,DataFrame(),
                         comp_params_2d)[end-1]) < 1
end

@testset "comp_test tests" begin
    @test std(comp_test(xyz,ind_test;comp_params=comp_params_1 ,
                        silent=true)[end-1]) < 1
    @test std(comp_test(xyz,ind_test;comp_params=comp_params_2a,
                        silent=true)[end-1]) < 1
    @test std(comp_test(xyz,ind_test;comp_params=comp_params_2b,
                        silent=true)[end-1]) < 1
    @test std(comp_test(xyz,ind_test;comp_params=comp_params_2c,
                        silent=true)[end-1]) < 1
    @test std(comp_test(xyz,ind_test;comp_params=comp_params_2d,
                        silent=true)[end-1]) < 1
    @test std(comp_test(line_test,df_line,df_flight,DataFrame(),
                        comp_params_1 ,silent=true)[end-1]) < 1
    @test std(comp_test(line_test,df_line,df_flight,DataFrame(),
                        comp_params_2a,silent=true)[end-1]) < 1
    @test std(comp_test(line_test,df_line,df_flight,DataFrame(),
                        comp_params_2b,silent=true)[end-1]) < 1
    @test std(comp_test(line_test,df_line,df_flight,DataFrame(),
                        comp_params_2c,silent=true)[end-1]) < 1
    @test std(comp_test(line_test,df_line,df_flight,DataFrame(),
                        comp_params_2d,silent=true)[end-1]) < 1
    @test_throws ErrorException comp_test(xyz,ind_test;
                                          comp_params=comp_params_nn_bad)
    @test_throws ErrorException comp_test(xyz,ind_test;
                                          comp_params=comp_params_lin_bad)
    @test_throws ErrorException comp_test(xyz,ind_test;
                                          comp_params=comp_params_nn_bad_drop)
    @test_throws ErrorException comp_test(line_test,df_line,df_flight,DataFrame(),
                                          comp_params_nn_bad)
    @test_throws ErrorException comp_test(line_test,df_line,df_flight,DataFrame(),
                                          comp_params_lin_bad)
    @test_throws ErrorException comp_test(line_test,df_line,df_flight,DataFrame(),
                                          comp_params_nn_bad_drop)
end

rm("drop_fi_1.bson")
rm("drop_fi_2.bson")
rm("drop_fi_3.bson")
rm("drop_fi_4.bson")

@testset "comp_m2bc_test tests" begin
    @test std(comp_m2bc_test(line_test,df_line,df_flight,DataFrame(),
                        comp_params_2b,silent=true)[end-1]) < 1
    @test std(comp_m2bc_test(line_test,df_line,df_flight,DataFrame(),
                        comp_params_2c,silent=true)[end-1]) < 1
end

terms       = [:p]
terms_A     = [:p,:i,:e]
batchsize   = 5
epoch_lbfgs = 1
k_pca       = 5
frac_train  = 1

comp_params_1  = MagNav.NNCompParams(model_type  = :m1,
                                     terms       = terms,
                                     terms_A     = terms_A,
                                     epoch_lbfgs = epoch_lbfgs,
                                     batchsize   = batchsize,
                                     k_pca       = k_pca,
                                     frac_train  = frac_train)
comp_params_2a = MagNav.NNCompParams(model_type=:m2a,
                                     terms       = terms,
                                     terms_A     = terms_A,
                                     epoch_lbfgs = epoch_lbfgs,
                                     batchsize   = batchsize,
                                     k_pca       = k_pca,
                                     frac_train  = frac_train)
comp_params_2b = MagNav.NNCompParams(model_type=:m2b,
                                     terms       = terms,
                                     terms_A     = terms_A,
                                     epoch_lbfgs = epoch_lbfgs,
                                     batchsize   = batchsize,
                                     k_pca       = k_pca,
                                     frac_train  = frac_train)
comp_params_2c = MagNav.NNCompParams(model_type=:m2c,
                                     terms       = terms,
                                     terms_A     = terms_A,
                                     epoch_lbfgs = epoch_lbfgs,
                                     batchsize   = batchsize,
                                     k_pca       = k_pca,
                                     frac_train  = frac_train)
comp_params_2d = MagNav.NNCompParams(model_type=:m2d,
                                     terms       = terms,
                                     terms_A     = terms_A,
                                     epoch_lbfgs = epoch_lbfgs,
                                     batchsize   = batchsize,
                                     k_pca       = k_pca,
                                     frac_train  = frac_train)

comp_params_TL         = MagNav.LinCompParams(model_type=:TL,y_type=:a)
comp_params_mod_TL     = MagNav.LinCompParams(model_type=:mod_TL,y_type=:a)
comp_params_elasticnet = MagNav.LinCompParams(model_type=:elasticnet,y_type=:a)
comp_params_plsr       = MagNav.LinCompParams(model_type=:plsr,y_type=:a,k_plsr=1)

comp_params_1_drop     = MagNav.NNCompParams(comp_params_1 ,drop_fi=true)
comp_params_1_perm     = MagNav.NNCompParams(comp_params_1 ,perm_fi=true)
comp_params_2c_drop    = MagNav.NNCompParams(comp_params_2c,drop_fi=true)
comp_params_2c_perm    = MagNav.NNCompParams(comp_params_2c,perm_fi=true)

@testset "comp_train_test tests" begin
    @test std(comp_train_test(xyz,xyz,ind_train,ind_test;
                              comp_params=comp_params_1 )[end-1]) < 10
    @test std(comp_train_test(xyz,xyz,ind_train,ind_test;
                              comp_params=comp_params_2a)[end-1]) < 10
    @test std(comp_train_test(xyz,xyz,ind_train,ind_test;
                              comp_params=comp_params_2b)[end-1]) < 10
    @test std(comp_train_test(xyz,xyz,ind_train,ind_test;
                              comp_params=comp_params_2c)[end-1]) < 10
    @test std(comp_train_test(xyz,xyz,ind_train,ind_test;
                              comp_params=comp_params_2d)[end-1]) < 10
    @test std(comp_train_test(xyz,xyz,ind_train,ind_test;
                              comp_params=comp_params_TL)[end-1]) < 10
    @test std(comp_train_test(xyz,xyz,ind_train,ind_test;
                              comp_params=comp_params_mod_TL)[end-1]) < 10
    @test std(comp_train_test(xyz,xyz,ind_train,ind_test;
                              comp_params=comp_params_elasticnet)[end-1]) < 10
    @test std(comp_train_test(xyz,xyz,ind_train,ind_test;
                              comp_params=comp_params_plsr)[end-1]) < 10
    @test std(comp_train_test(line_train,line_test,df_line,df_flight,
                              DataFrame(),comp_params_1 )[end-1]) < 10
    @test std(comp_train_test(line_train,line_test,df_line,df_flight,
                              DataFrame(),comp_params_2a)[end-1]) < 10
    @test std(comp_train_test(line_train,line_test,df_line,df_flight,
                              DataFrame(),comp_params_2b)[end-1]) < 10
    @test std(comp_train_test(line_train,line_test,df_line,df_flight,
                              DataFrame(),comp_params_2c)[end-1]) < 10
    @test std(comp_train_test(line_train,line_test,df_line,df_flight,
                              DataFrame(),comp_params_2d)[end-1]) < 10
    @test std(comp_train_test(line_train,line_test,df_line,df_flight,
                              DataFrame(),comp_params_TL)[end-1]) < 10
    @test std(comp_train_test(line_train,line_test,df_line,df_flight,
                              DataFrame(),comp_params_mod_TL)[end-1]) < 10
    @test std(comp_train_test(line_train,line_test,df_line,df_flight,
                              DataFrame(),comp_params_elasticnet)[end-1]) < 10
    @test std(comp_train_test(line_train,line_test,df_line,df_flight,
                              DataFrame(),comp_params_plsr)[end-1]) < 10

    @test std(comp_train_test(xyz,xyz,ind_train,ind_test;
                              comp_params=comp_params_1_drop )[end-1]) < 10
    @test std(comp_train_test(xyz,xyz,ind_train,ind_test;
                              comp_params=comp_params_2c_drop)[end-1]) < 10
    @test std(comp_train_test(xyz,xyz,ind_train,ind_test;
                              comp_params=comp_params_1_perm )[end-1]) < 10
    @test std(comp_train_test(xyz,xyz,ind_train,ind_test;
                              comp_params=comp_params_2c_perm)[end-1]) < 10
    @test std(comp_train_test(line_train,line_test,df_line,df_flight,
                              DataFrame(),comp_params_1_drop )[end-1]) < 10
    @test std(comp_train_test(line_train,line_test,df_line,df_flight,
                              DataFrame(),comp_params_2c_drop)[end-1]) < 10
    @test std(comp_train_test(line_train,line_test,df_line,df_flight,
                              DataFrame(),comp_params_1_perm )[end-1]) < 10
    @test std(comp_train_test(line_train,line_test,df_line,df_flight,
                              DataFrame(),comp_params_2c_perm)[end-1]) < 10
end

@testset "print_time tests" begin
    @test typeof(MagNav.print_time(30)) == Nothing
    @test typeof(MagNav.print_time(90)) == Nothing
end

rm("drop_fi_1.bson")
rm("drop_fi_2.bson")
rm("drop_fi_3.bson")
rm("drop_fi_4.bson")
rm("drop_fi.csv")
rm("perm_fi.csv")
