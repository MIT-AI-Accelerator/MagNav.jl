using MagNav, Test, MAT

map_file  = "test_data/test_data_map.mat"
traj_file = "test_data/test_data_traj.mat"

mapS      = get_map(map_file)
itp_mapS  = map_interpolate(mapS)

xyz    = get_XYZ0(traj_file,:traj,:none;silent=true)
traj   = xyz.traj
ins    = xyz.ins
flux_a = xyz.flux_a

terms       = [:p]
batchsize   = 5
comp_params = MagNav.NNCompParams(model_type=:m1,terms=terms,batchsize=batchsize)
comp_params = comp_train(xyz,trues(traj.N);comp_params=comp_params)[1]

x = [xyz.mag_1_uc create_TL_A(flux_a;terms=terms)]
y = xyz.mag_1_c

(_,_,x_norm) = norm_sets(x)
(y_bias,y_scale,y_norm) = norm_sets(y)
y_norms = (y_bias,y_scale)

m = get_nn_m(comp_params.weights)
(P0_nn,nn_sigma) = ekf_online_nn_setup(x_norm,y_norm,m,y_norms;N_sigma=10)

(P0,Qd,R) = create_model(traj.dt,traj.lat[1];
                         vec_states = false,
                         TL_sigma   = nn_sigma,
                         P0_TL      = P0_nn)

@testset "ekf_online_nn tests" begin
    @test_nowarn ekf_online_nn_setup(x_norm,y_norm,m,y_norms;N_sigma=10)
    @test_nowarn ekf_online_nn(ins,xyz.mag_1_uc,itp_mapS,x_norm,m,y_norms,P0,Qd,R)
end
