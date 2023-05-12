using MagNav, Test, MAT, Random
Random.seed!(2)

test_file = joinpath(@__DIR__,"test_data/test_data_ekf.mat")
ekf_data  = matopen(test_file,"r") do file
    read(file,"ekf_data")
end

test_file = joinpath(@__DIR__,"test_data/test_data_grid.mat")
grid_data = matopen(test_file,"r") do file
    read(file,"grid_data")
end

test_file = joinpath(@__DIR__,"test_data/test_data_map.mat")
map_data  = matopen(test_file,"r") do file
    read(file,"map_data")
end

test_file = joinpath(@__DIR__,"test_data/test_data_params.mat")
params    = matopen(test_file,"r") do file
    read(file,"params")
end

test_file   = joinpath(@__DIR__,"test_data/test_data_pinson.mat")
pinson_data = matopen(test_file,"r") do file
    read(file,"pinson_data")
end

test_file = joinpath(@__DIR__,"test_data/test_data_traj.mat")
traj_data = matopen(test_file,"r") do file
    read(file,"traj")
end

xn = vec(grid_data["xn"])
xl = vec(grid_data["xl"])
x  = vec(grid_data["x"])

map_map = map_data["map"]
map_xx  = deg2rad.(vec(map_data["xx"]))
map_yy  = deg2rad.(vec(map_data["yy"]))
map_alt = map_data["alt"]

dt             = params["dt"]
init_pos_sigma = params["init_pos_sigma"]
init_alt_sigma = params["init_alt_sigma"]
init_vel_sigma = params["init_vel_sigma"]
init_att_sigma = params["init_att_sigma"]
meas_var       = params["meas_R"]
VRW_sigma      = sqrt(params["VRW_var"])
ARW_sigma      = sqrt(params["ARW_var"])
baro_sigma     = params["baro_std"]
ha_sigma       = params["ha_sigma"]
a_hat_sigma    = params["a_hat_sigma"]
acc_sigma      = params["acc_sigma"]
gyro_sigma     = params["gyro_sigma"]
fogm_sigma     = params["meas_sigma"]
baro_tau       = params["baro_tau"]
acc_tau        = params["acc_tau"]
gyro_tau       = params["gyro_tau"]
fogm_tau       = params["meas_tau"]

lat = deg2rad(traj_data["lat"][1])
lon = deg2rad(traj_data["lon"][1])
alt = traj_data["alt"][1]
vn  = traj_data["vn"][1]
ve  = traj_data["ve"][1]
vd  = traj_data["vd"][1]
fn  = traj_data["fn"][1]
fe  = traj_data["fe"][1]
fd  = traj_data["fd"][1]
Cnb = traj_data["Cnb"][:,:,1]

fogm_data_MT = [0.7396206598864331
                0.6954048478836197
                0.6591146045997668
                0.5585131871428970
                0.5185764218327329] # MersenneTwister

fogm_data_XO = [-0.00573724460026025
                 0.09446430193246710
                 0.03368954850407684
                 0.06685250250778804
                 0.04552990438603038] # Xoshiro

mapS = MagNav.MapS(map_map,map_xx,map_yy,map_alt)
(itp_mapS,der_mapS) = map_interpolate(mapS,:linear;return_vert_deriv=true) # linear to match MATLAB

(P0,Qd,R) = create_model(dt,lat[1];
                         init_pos_sigma = init_pos_sigma,
                         init_alt_sigma = init_alt_sigma,
                         init_vel_sigma = init_vel_sigma,
                         init_att_sigma = init_att_sigma,
                         meas_var       = meas_var,
                         VRW_sigma      = VRW_sigma,
                         ARW_sigma      = ARW_sigma,
                         baro_sigma     = baro_sigma,
                         ha_sigma       = ha_sigma,
                         a_hat_sigma    = a_hat_sigma,
                         acc_sigma      = acc_sigma,
                         gyro_sigma     = gyro_sigma,
                         fogm_sigma     = fogm_sigma,
                         baro_tau       = baro_tau,
                         acc_tau        = acc_tau,
                         gyro_tau       = gyro_tau,
                         fogm_tau       = fogm_tau)

@testset "create_model tests" begin
    @test P0 ≈ ekf_data["P0"]
    @test Qd ≈ ekf_data["Qd"]
    @test R  ≈ ekf_data["R"]
end

@testset "get_pinson tests" begin
    @test MagNav.get_pinson(17,lat[1],vn,ve,vd,fn,fe,fd,Cnb;
                            baro_tau   = baro_tau,
                            acc_tau    = acc_tau,
                            gyro_tau   = gyro_tau,
                            fogm_tau   = fogm_tau,
                            vec_states = false,
                            fogm_state = false) ≈ pinson_data["P17"]
    @test MagNav.get_pinson(17+1,lat[1],vn,ve,vd,fn,fe,fd,Cnb;
                            baro_tau   = baro_tau,
                            acc_tau    = acc_tau,
                            gyro_tau   = gyro_tau,
                            fogm_tau   = fogm_tau,
                            vec_states = false,
                            fogm_state = true) ≈ pinson_data["P18"]
    @test_nowarn MagNav.get_pinson(17+3,lat[1],vn,ve,vd,fn,fe,fd,Cnb;
                                   vec_states=true,fogm_state=false)
end

@testset "get_Phi tests" begin
    @test MagNav.get_Phi(18,lat[1],vn,ve,vd,fn,fe,fd,Cnb,baro_tau,acc_tau,
                         gyro_tau,fogm_tau,dt) ≈ pinson_data["Phi"]
end

@testset "get_H tests" begin
    @test MagNav.get_H(itp_mapS,x,lat,lon,alt;core=false) ≈ grid_data["H"]
end

@testset "get_h tests" begin
    @test MagNav.get_h(itp_mapS,x,lat,lon,alt;core=false)[1] ≈ grid_data["h"]
    @test MagNav.get_h(itp_mapS,[xn;0;xl[2:end]],lat,lon,alt;
                core=false)[1] ≈ grid_data["hRBPF"]
    @test_nowarn MagNav.get_h(itp_mapS,der_mapS,x,lat,lon,alt,map_alt;core=false)
    @test_nowarn MagNav.get_h(itp_mapS,der_mapS,x,lat,lon,alt,map_alt;core=true)
end

@testset "map_grad tests" begin
    @test MagNav.map_grad(itp_mapS,lat,lon) ≈ reverse(vec(grid_data["grad"]))
end

@testset "fogm tests" begin
    fogm_data = fogm(fogm_sigma,fogm_tau,dt,length(fogm_data_MT))
    @test (fogm_data ≈ fogm_data_MT) | (fogm_data ≈ fogm_data_XO)
end
