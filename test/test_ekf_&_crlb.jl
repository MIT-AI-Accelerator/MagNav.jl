using MagNav, Test, MAT
using LinearAlgebra
using MagNav: FILTres

test_file = joinpath(@__DIR__,"test_data","test_data_ekf.mat")
ekf_data  = matopen(test_file,"r") do file
    read(file,"ekf_data")
end

test_file = joinpath(@__DIR__,"test_data","test_data_ins.mat")
ins_data  = matopen(test_file,"r") do file
    read(file,"ins_data")
end

test_file = joinpath(@__DIR__,"test_data","test_data_map.mat")
map_data  = matopen(test_file,"r") do file
    read(file,"map_data")
end

test_file = joinpath(@__DIR__,"test_data","test_data_params.mat")
params    = matopen(test_file,"r") do file
    read(file,"params")
end

test_file = joinpath(@__DIR__,"test_data","test_data_traj.mat")
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

map_info = "Map"
map_map  = map_data["map"]
map_xx   = deg2rad.(vec(map_data["xx"]))
map_yy   = deg2rad.(vec(map_data["yy"]))
map_alt  = map_data["alt"]
map_mask = MagNav.map_params(map_map,map_xx,map_yy)[2]

dt       = params["dt"]
baro_tau = params["baro_tau"]
acc_tau  = params["acc_tau"]
gyro_tau = params["gyro_tau"]
fogm_tau = params["meas_tau"]

tt       = vec(traj_data["tt"])
lat      = deg2rad.(vec(traj_data["lat"]))
lon      = deg2rad.(vec(traj_data["lon"]))
alt      = vec(traj_data["alt"])
vn       = vec(traj_data["vn"])
ve       = vec(traj_data["ve"])
vd       = vec(traj_data["vd"])
fn       = vec(traj_data["fn"])
fe       = vec(traj_data["fe"])
fd       = vec(traj_data["fd"])
Cnb      = traj_data["Cnb"]
mag_1_c  = vec(traj_data["mag_1_c"])
N        = length(lat)

traj = MagNav.Traj(N,dt,tt,lat,lon,alt,vn,ve,vd,fn,fe,fd,Cnb)
ins  = MagNav.INS( N,dt,tt,ins_lat,ins_lon,ins_alt,ins_vn,ins_ve,ins_vd,
                   ins_fn,ins_fe,ins_fd,ins_Cnb,zeros(3,3,N))

mapS = MagNav.MapS(map_info,map_map,map_xx,map_yy,map_alt,map_mask)
map_cache = Map_Cache(maps=[mapS])
(itp_mapS,der_mapS) = map_interpolate(mapS,:linear;return_vert_deriv=true) # linear to match MATLAB

crlb_P = crlb(lat,lon,alt,vn,ve,vd,fn,fe,fd,Cnb,dt,itp_mapS;
              P0       = P0,
              Qd       = Qd,
              R        = R,
              baro_tau = baro_tau,
              acc_tau  = acc_tau,
              gyro_tau = gyro_tau,
              fogm_tau = fogm_tau,
              core     = false)

filt_res = ekf(ins_lat,ins_lon,ins_alt,ins_vn,ins_ve,ins_vd,
               ins_fn,ins_fe,ins_fd,ins_Cnb,mag_1_c,dt,itp_mapS;
               P0       = P0,
               Qd       = Qd,
               R        = R,
               baro_tau = baro_tau,
               acc_tau  = acc_tau,
               gyro_tau = gyro_tau,
               fogm_tau = fogm_tau,
               core     = false)

@testset "crlb tests" begin
    @test isapprox(crlb_P[:,:,1]  ,ekf_data["crlb_P"][:,:,1]  ,atol=1e-6)
    @test isapprox(crlb_P[:,:,end],ekf_data["crlb_P"][:,:,end],atol=1e-6)
    @test size(crlb(traj,itp_mapS )) == (18,18,100)
    @test size(crlb(traj,map_cache)) == (18,18,100)
end

@testset "ekf tests" begin
    @test isapprox(filt_res.x[:,1]    ,ekf_data["x_out"][:,1]    ,atol=1e-6)
    @test isapprox(filt_res.x[:,end]  ,ekf_data["x_out"][:,end]  ,atol=1e-3)
    @test isapprox(filt_res.P[:,:,1]  ,ekf_data["P_out"][:,:,1]  ,atol=1e-6)
    @test isapprox(filt_res.P[:,:,end],ekf_data["P_out"][:,:,end],atol=1e-3)
    @test ekf(ins,mag_1_c,itp_mapS) isa FILTres
    @test ekf(ins,mag_1_c,map_cache;core=true) isa FILTres
    @test ekf(ins,mag_1_c,itp_mapS;R=(1,10)) isa FILTres
    @test ekf(ins,mag_1_c,itp_mapS;der_mapS,map_alt) isa FILTres
end

# create EKF_RT struct with EKF initializations & parameters
ny = size(mag_1_c,2)
a  = EKF_RT(P        = P0,
            Qd       = Qd,
            R        = R,
            baro_tau = baro_tau,
            acc_tau  = acc_tau,
            gyro_tau = gyro_tau,
            fogm_tau = fogm_tau,
            core     = false,
            ny       = ny)

x = zeros(a.nx,N)
P = zeros(a.nx,a.nx,N)
r = zeros(a.ny,N)
for t = 1:N
    filt_res_ = a(ins_lat[t], ins_lon[t], ins_alt[t],
                  ins_vn[t] , ins_ve[t] , ins_vd[t] ,
                  ins_fn[t] , ins_fe[t] , ins_fd[t] ,
                  ins_Cnb[:,:,t], mag_1_c[t], tt[t], itp_mapS; dt=dt)
    x[:,t]   = filt_res_.x
    P[:,:,t] = filt_res_.P
    r[:,t]   = filt_res_.r
end

# test that x, P, & r for EKF_RT() exactly match ekf()
@testset "EKF_RT tests" begin
    @test x[:,1]     ≈ filt_res.x[:,1]
    @test x[:,end]   ≈ filt_res.x[:,end]
    @test P[:,:,1]   ≈ filt_res.P[:,:,1]
    @test P[:,:,end] ≈ filt_res.P[:,:,end]
    @test r[:,1]     ≈ filt_res.r[:,1]
    @test r[:,end]   ≈ filt_res.r[:,end]
    @test a(ins(N),mag_1_c[N],itp_mapS ) isa FILTres
    @test a(ins(N),mag_1_c[N],map_cache) isa FILTres
    @test a(ins(N),mag_1_c[N],itp_mapS;der_mapS,map_alt) isa FILTres
end
