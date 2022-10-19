using MagNav, Test, MAT

map_file  = joinpath(@__DIR__,"test_data/test_data_map.mat")
traj_file = joinpath(@__DIR__,"test_data/test_data_traj.mat")

mapS      = get_map(map_file)
itp_mapS  = map_interpolate(mapS)

xyz    = get_XYZ0(traj_file,:traj,:none;silent=true)
traj   = xyz.traj
ins    = xyz.ins
flux_a = xyz.flux_a

(x0_TL,P0_TL,TL_sigma) = ekf_online_setup(flux_a,xyz.mag_1_c;N_sigma=10)
(P0_1,Qd_1,R_1) = create_model(traj.dt,traj.lat[1];
                               vec_states = true,
                               TL_sigma   = TL_sigma,
                               P0_TL      = P0_TL)
(P0_2,Qd_2,R_2) = create_model(traj.dt,traj.lat[1];
                               vec_states = false,
                               TL_sigma   = TL_sigma,
                               P0_TL      = P0_TL)

@testset "ekf_online tests" begin
    @test typeof(ekf_online_setup(flux_a,xyz.mag_1_c;N_sigma=10)) <: Tuple{Vector,Matrix,Vector}
    @test_nowarn ekf_online(ins,xyz.mag_1_c,flux_a,itp_mapS,x0_TL,P0_1,Qd_1,R_1)
    @test_nowarn ekf_online(ins,xyz.mag_1_c,flux_a,itp_mapS,x0_TL,P0_2,Qd_2,R_2)
    @test typeof(run_filt(traj,ins,xyz.mag_1_c,itp_mapS,:ekf_online;
                 P0=P0_1,Qd=Qd_1,R=R_1,flux=flux_a,x0_TL=x0_TL)) <: Tuple{MagNav.CRLBout,MagNav.INSout,MagNav.FILTout}
end
