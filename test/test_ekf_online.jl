using MagNav, Test, MAT

map_file  = "test_data/test_data_map.mat"
traj_file = "test_data/test_data_traj.mat"

mapS      = get_map(map_file)
itp_mapS  = map_interpolate(mapS)

xyz    = get_XYZ0(traj_file,:traj,:none;silent=true)
traj   = xyz.traj
ins    = xyz.ins
flux_a = xyz.flux_a

(x0_TL,P0_TL,TL_sigma) = ekf_online_setup(flux_a,xyz.mag_1_c;N_sigma=10)
(P0,Qd,R) = create_model(traj.dt,traj.lat[1];
                         vec_states = true,
                         TL_sigma   = TL_sigma,
                         P0_TL      = P0_TL)

@testset "ekf_online tests" begin
    @test_nowarn ekf_online_setup(flux_a,xyz.mag_1_c;N_sigma=10)
    @test_nowarn ekf_online(ins,xyz.mag_1_c,flux_a,itp_mapS,x0_TL,P0,Qd,R)
end
