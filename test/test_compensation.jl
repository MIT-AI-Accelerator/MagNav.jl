using MagNav, Test, MAT, DataFrames, Statistics

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

terms_A   = [:p,:i,:e]
batchsize = 5
comp_params_1  = MagNav.NNCompParams(model_type=:m1 ,terms_A=terms_A,batchsize=batchsize)
comp_params_2a = MagNav.NNCompParams(model_type=:m2a,terms_A=terms_A,batchsize=batchsize)
comp_params_2b = MagNav.NNCompParams(model_type=:m2b,terms_A=terms_A,batchsize=batchsize)
comp_params_2c = MagNav.NNCompParams(model_type=:m2c,terms_A=terms_A,batchsize=batchsize)
comp_params_2d = MagNav.NNCompParams(model_type=:m2d,terms_A=terms_A,batchsize=batchsize)

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
end

comp_params_1  = comp_train(xyz,ind_train;comp_params=comp_params_1 )[1]
comp_params_2a = comp_train(xyz,ind_train;comp_params=comp_params_2a)[1]
comp_params_2b = comp_train(xyz,ind_train;comp_params=comp_params_2b)[1]
comp_params_2c = comp_train(xyz,ind_train;comp_params=comp_params_2c)[1]
comp_params_2d = comp_train(xyz,ind_train;comp_params=comp_params_2d)[1]

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
end

@testset "comp_m2bc_test tests" begin
    @test std(comp_m2bc_test(line_test,df_line,df_flight,DataFrame(),
                        comp_params_2b,silent=true)[end-1]) < 1
    @test std(comp_m2bc_test(line_test,df_line,df_flight,DataFrame(),
                        comp_params_2c,silent=true)[end-1]) < 1
end

@testset "comp_train_test tests" begin
    @test std(comp_train_test(xyz,xyz,ind_train,ind_test;
                              comp_params=comp_params_1 )[end-1]) < 1
    @test std(comp_train_test(xyz,xyz,ind_train,ind_test;
                              comp_params=comp_params_2a)[end-1]) < 1
    @test std(comp_train_test(xyz,xyz,ind_train,ind_test;
                              comp_params=comp_params_2b)[end-1]) < 1
    @test std(comp_train_test(xyz,xyz,ind_train,ind_test;
                              comp_params=comp_params_2c)[end-1]) < 1
    @test std(comp_train_test(xyz,xyz,ind_train,ind_test;
                              comp_params=comp_params_2d)[end-1]) < 1
    @test std(comp_train_test(line_train,line_test,df_line,df_flight,
                              DataFrame(),comp_params_1 )[end-1]) < 1
    @test std(comp_train_test(line_train,line_test,df_line,df_flight,
                              DataFrame(),comp_params_2a)[end-1]) < 1
    @test std(comp_train_test(line_train,line_test,df_line,df_flight,
                              DataFrame(),comp_params_2b)[end-1]) < 1
    @test std(comp_train_test(line_train,line_test,df_line,df_flight,
                              DataFrame(),comp_params_2c)[end-1]) < 1
    @test std(comp_train_test(line_train,line_test,df_line,df_flight,
                              DataFrame(),comp_params_2d)[end-1]) < 1
end
