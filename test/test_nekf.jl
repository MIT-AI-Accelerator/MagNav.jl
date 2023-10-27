using MagNav, Test, MAT

map_file  = joinpath(@__DIR__,"test_data/test_data_map.mat")
traj_file = joinpath(@__DIR__,"test_data/test_data_traj.mat")

mapS      = get_map(map_file)
itp_mapS  = map_interpolate(mapS)

xyz    = get_XYZ0(traj_file,:traj,:none;silent=true)
traj   = xyz.traj
ins    = xyz.ins
flux_a = xyz.flux_a

ind = trues(traj.N)
ind[51:end] .= false

x = [xyz.mag_1_uc create_TL_A(flux_a;terms=[:p])][ind,:]
(m,data_norms) = nekf_train(xyz,ind,xyz.mag_1_c,itp_mapS,x;epoch_adam=1,l_window=10)
(v_scale,x_bias,x_scale) = data_norms
x_nn = ((x .- x_bias) ./ x_scale) * v_scale

@testset "nekf tests" begin
    @test_nowarn nekf_train(xyz,ind,xyz.mag_1_c,itp_mapS,x;epoch_adam=1,l_window=10)
    @test typeof(nekf(ins(ind),xyz.mag_1_c[ind],itp_mapS,x_nn,m)) <: MagNav.FILTres
    @test typeof(run_filt(traj(ind),ins(ind),xyz.mag_1_c[ind],itp_mapS,:nekf;
                 x_nn=x_nn,m=m)) <: Tuple{MagNav.CRLBout,MagNav.INSout,MagNav.FILTout}
    @test typeof(nekf_train(xyz,ind,xyz.mag_1_c,itp_mapS,x;epoch_adam=1,l_seq=10)) <: Tuple # deprecated
end
