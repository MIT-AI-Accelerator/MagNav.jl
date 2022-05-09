using MagNav, Test, MAT, Random
Random.seed!(2)

test_file = "test_data/test_data_ekf.mat"
ekf_data  = matopen(test_file,"r") do file
    read(file,"ekf_data")
end

test_file = "test_data/test_data_params.mat"
params    = matopen(test_file,"r") do file
    read(file,"params")
end

test_file   = "test_data/test_data_pinson.mat"
pinson_data = matopen(test_file,"r") do file
    read(file,"pinson_data")
end

test_file = "test_data/test_data_traj.mat"
traj_data = matopen(test_file,"r") do file
    read(file,"traj_data")
end

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

lat1 = deg2rad(traj_data["lat"][1])
vn  = traj_data["vn"][1]
ve  = traj_data["ve"][1]
vd  = traj_data["vd"][1]
fn  = traj_data["fn"][1]
fe  = traj_data["fe"][1]
fd  = traj_data["fd"][1]
Cnb = traj_data["Cnb"][:,:,1]

fogm_data = [-0.00573724460026025
              0.09446430193246710
              0.03368954850407684
              0.06685250250778804
              0.04552990438603038]

(P0,Qd,R) = create_model(dt,lat1;
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

@testset "FOGM Tests" begin
    @test fogm(fogm_sigma,fogm_tau,dt,length(fogm_data)) ≈ fogm_data
end

@testset "Create Model Tests" begin
    @test P0 ≈ ekf_data["P0"]
    @test Qd ≈ ekf_data["Qd"]
    @test R  ≈ ekf_data["R"]
end

@testset "Pinson 17x17 Tests" begin
    @test get_pinson(17,lat1,vn,ve,vd,fn,fe,fd,Cnb;
                     baro_tau   = baro_tau,
                     acc_tau    = acc_tau,
                     gyro_tau   = gyro_tau,
                     fogm_tau   = fogm_tau,
                     fogm_state = false) ≈ pinson_data["P17"]
end

@testset "Pinson 18x18 Tests" begin
    @test get_pinson(18,lat1,vn,ve,vd,fn,fe,fd,Cnb;
                     baro_tau   = baro_tau,
                     acc_tau    = acc_tau,
                     gyro_tau   = gyro_tau,
                     fogm_tau   = fogm_tau,
                     fogm_state = true) ≈ pinson_data["P18"]
end

@testset "Get Phi Tests" begin
    @test get_Phi(18,lat1,vn,ve,vd,fn,fe,fd,Cnb,
                  baro_tau,acc_tau,gyro_tau,fogm_tau,dt) ≈ pinson_data["Phi"]
end
