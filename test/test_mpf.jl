using MagNav, Test, MAT, Random

test_file = "test_data/test_data_ekf.mat"
ekf_data  = matopen(test_file,"r") do file
    read(file,"ekf_data")
end

test_file = "test_data/test_data_ins.mat"
ins_data  = matopen(test_file,"r") do file
    read(file,"ins_data")
end

test_file = "test_data/test_data_map.mat"
map_data  = matopen(test_file,"r") do file
    read(file,"map_data")
end

test_file = "test_data/test_data_params.mat"
params    = matopen(test_file,"r") do file
    read(file,"params")
end

test_file = "test_data/test_data_traj.mat"
traj_data = matopen(test_file,"r") do file
    read(file,"traj")
end

P0 = ekf_data["P0"]
Qd = ekf_data["Qd"]
R  = ekf_data["R"]

ins_lat  = deg2rad.(vec(ins_data["lat"]))
ins_lon  = deg2rad.(vec(ins_data["lon"]))
ins_alt  = vec(ins_data["alt"])
ins_vn   = vec(ins_data["vn"])
ins_ve   = vec(ins_data["ve"])
ins_vd   = vec(ins_data["vd"])
ins_fn   = vec(ins_data["fn"])
ins_fe   = vec(ins_data["fe"])
ins_fd   = vec(ins_data["fd"])
ins_Cnb  = ins_data["Cnb"]
N        = length(ins_lat)

map_map  = map_data["map"]
map_xx   = deg2rad.(vec(map_data["xx"]))
map_yy   = deg2rad.(vec(map_data["yy"]))

dt       = params["dt"]
num_part = round(Int,params["num_particles"])
thresh   = params["resampleThresh"]
baro_tau = params["baro_tau"]
acc_tau  = params["acc_tau"]
gyro_tau = params["gyro_tau"]
fogm_tau = params["meas_tau"]
date     = 2020+185/366
core     = false

tt       = vec(traj_data["tt"])
mag_1_c  = vec(traj_data["mag_1_c"])

ins = MagNav.INS(N,dt,tt,ins_lat,ins_lon,ins_alt,ins_vn,ins_ve,ins_vd,
                 ins_fn,ins_fe,ins_fd,ins_Cnb,zeros(1,1,1))

itp_mapS = map_interpolate(map_map,map_xx,map_yy,:linear) # linear to match MATLAB

Random.seed!(2)
filt_res_1 = mpf(ins_lat,ins_lon,ins_alt,ins_vn,ins_ve,ins_vd,
                 ins_fn,ins_fe,ins_fd,ins_Cnb,mag_1_c,dt,itp_mapS;
                 P0       = P0,
                 Qd       = Qd,
                 R        = R,
                 num_part = num_part,
                 thresh   = thresh,
                 baro_tau = baro_tau,
                 acc_tau  = acc_tau,
                 gyro_tau = gyro_tau,
                 fogm_tau = fogm_tau,
                 date     = date,
                 core     = core)
Random.seed!(2)
filt_res_2 = mpf(ins,mag_1_c,itp_mapS;
                 P0       = P0,
                 Qd       = Qd,
                 R        = R,
                 num_part = num_part,
                 thresh   = thresh,
                 baro_tau = baro_tau,
                 acc_tau  = acc_tau,
                 gyro_tau = gyro_tau,
                 fogm_tau = fogm_tau,
                 date     = date,
                 core     = core)

@testset "mpf tests" begin
    @test filt_res_1.x[:,1]     ≈ filt_res_2.x[:,1]
    @test filt_res_1.x[:,end]   ≈ filt_res_2.x[:,end]
    @test filt_res_1.P[:,:,1]   ≈ filt_res_2.P[:,:,1]
    @test filt_res_1.P[:,:,end] ≈ filt_res_2.P[:,:,end]
    @test mpf(ins_lat,ins_lon,ins_alt,ins_vn,ins_ve,ins_vd,
              ins_fn,ins_fe,ins_fd,ins_Cnb,mag_1_c,dt,itp_mapS;
              num_part=100,thresh=0.5).c ≈ true
    @test mpf(ins,mag_1_c,itp_mapS;num_part=100,thresh=0.5).c ≈ true
    @test mpf(ins,zero(mag_1_c),itp_mapS;num_part=100,thresh=0.5).c ≈ false
end

@testset "mpf helper tests" begin
    @test MagNav.sys_resample([0,1,0]) ≈ [2,2,2]
    @test MagNav.sys_resample([0.3,0.3,0.3])[end] ≈ 3
    @test MagNav.part_cov([0,1,0],zeros(3,3),ones(3))           ≈ ones(3,3)
    @test MagNav.part_cov([0,1,0],zeros(3,3),ones(3),ones(3,3)) ≈ 2*ones(3,3)
    @test MagNav.filter_exit([0;;],[0;;],0,true)                ≈ zeros(2,2,1)
    @test MagNav.filter_exit([1;;],[1;;],1,false)[:, :, 1]      ≈ [1 0; 0 1]
end
