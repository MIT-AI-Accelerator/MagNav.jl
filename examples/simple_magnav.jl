##*  This is a simple MagNav example to use with YOUR data from scratch
# 1) Install Julia (Juliaup is great): https://julialang.org/downloads/
# 2) Install Visual Studio Code + Julia extension (recommended IDE)
# 3) Start Julia & run this script (should show 7 m navigation errors)
# 4) Modify files in simple_magnav_data folder with your data
    # fields for traj (aka GPS) & ins structs:
        # tt       [s]    time stamps
        # lat      [deg]  latitude
        # lon      [deg]  longitude
        # alt      [m]    altitude
        # roll     [deg]  roll
        # pitch    [deg]  pitch
        # yaw      [deg]  yaw
        # mag_1_c  [nT]   scalar magnetic data (only in traj)
    # fields for map struct:
        # map  [nT]   magnetic anomaly map (shape of yy by xx)
        # xx   [deg]  longitude
        # yy   [deg]  latitude
        # alt  [m]    altitude (assumed single constant HAE)
# 5) Modify filter parameters in this script for your data
# 6) Re-run this script
# 7) Have FUN and/or frustration

##* Setup
cd(@__DIR__)
using Pkg
# Pkg.add("MagNav") # as needed
using MagNav
using Random: seed!
dir = "simple_magnav_data";

##* Option 1: load data from CSV files
seed!(33); # for reproducibility
xyz_csv = "$dir/xyz.csv";
map_csv = "$dir/map"; # folder with alt, map, xx, & yy CSV files
xyz1    = get_XYZ0(xyz_csv);
mapS1   = get_map(map_csv);

##* Option 2: load data from HDF5 files
seed!(33); # for reproducibility
xyz_h5  = "$dir/xyz.h5";
map_h5  = "$dir/map.h5";
xyz2    = get_XYZ0(xyz_h5);
mapS2   = get_map(map_h5);

##* Option 3: load data from MAT files
seed!(33); # for reproducibility
xyz_mat = "$dir/xyz.mat";
map_mat = "$dir/map.mat";
xyz3    = get_XYZ0(xyz_mat);
mapS3   = get_map(map_mat);

##* Choose data from load option 1, 2, or 3 above (HDF5 selected here)
xyz  = xyz2;
mapS = mapS2;

##* Create navigation model
(P0,Qd,R) = create_model(xyz.traj.dt, xyz.traj.lat[1];
                         init_pos_sigma = 3.0,           # initial position uncertainty [m]
                         init_alt_sigma = 0.001,         # initial altitude uncertainty [m]
                         init_vel_sigma = 0.01,          # initial velocity uncertainty [m/s]
                         init_att_sigma = deg2rad(0.01), # initial attitude uncertainty [rad]
                         meas_var       = 3.0^2,         # measurement (white) noise variance [nT^2]
                         VRW_sigma      = 0.000238,      # velocity random walk [m/s^2 /sqrt(Hz)]
                         ARW_sigma      = 0.000000581,   # angular  random walk [rad/s /sqrt(Hz)]
                         baro_sigma     = 1.0,           # barometer bias [m]
                         ha_sigma       = 0.001,         # barometer aiding altitude bias [m]
                         a_hat_sigma    = 0.01,          # barometer aiding vertical accel bias [m/s^2]
                         acc_sigma      = 0.000245,      # accelerometer bias [m/s^2]
                         gyro_sigma     = 0.00000000727, # gyroscope bias [rad/s]
                         fogm_sigma     = 3.0,           # FOGM catch-all bias [nT]
                         baro_tau       = 3600.0,        # barometer time constant [s]
                         acc_tau        = 3600.0,        # accelerometer time constant [s]
                         gyro_tau       = 3600.0,        # gyroscope time constant [s]
                         fogm_tau       = 600.0);        # FOGM catch-all time constant [s]

##* Run navigation filter
itp_mapS = map_interpolate(mapS); # map interpolation function
(crlb_out,ins_out,filt_out) = run_filt(xyz.traj, xyz.ins, xyz.mag_1_c, itp_mapS, :ekf;
                                       P0       = P0,     # initial covariance matrix
                                       Qd       = Qd,     # discrete time process/system noise matrix
                                       R        = R,      # measurement (white) noise variance
                                       baro_tau = 3600.0, # barometer time constant [s]
                                       acc_tau  = 3600.0, # accelerometer time constant [s]
                                       gyro_tau = 3600.0, # gyroscope time constant [s]
                                       fogm_tau = 600.0,  # FOGM catch-all time constant [s]
                                       date     = get_years(2025,1), # measurement date (decimal year), for core = true
                                       core     = false);            # if core field contained within xyz.mag_1_c field

##* Plot navigation results
p1 = plot_filt(xyz.traj,xyz.ins,filt_out;show_plot=false)[1]
