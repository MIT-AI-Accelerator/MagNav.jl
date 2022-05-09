## full example using the MagNav package with simulated data
cd(@__DIR__)
include("common_setup.jl"); # load common-use packages, DataFrames, and Dicts

##* flight, map, and INS data section ========================================
seed!(2) # for reproducibility
t    = 1800; # selected flight time [s]
mapS = get_map(MagNav.namad); # load built-in NAMAD map
xyz  = create_XYZ0(mapS;alt=mapS.alt,t=t,n_waves=2); # create flight data
traj = xyz.traj; # get trajectory (GPS) struct
ins  = xyz.ins;  # get INS struct
mapS = map_trim(mapS,traj;pad=10); # trim map for given trajectory (with padding)
itp_mapS = map_interpolate(mapS); # get map interpolation function

##* navigation filter section ================================================
(P0,Qd,R) = create_model(traj.dt,traj.lat[1]); # create filter model

# run filter (EKF or MPF), determine CRLB
mag_use = xyz.mag_1_c; # selected mag (using compensated mag)
(crlb_out,ins_out,filt_out) = run_filt(traj,ins,mag_use,itp_mapS,:ekf;P0,Qd,R);

#* plotting section ==========================================================
t0 = traj.tt[1];
tt = (traj.tt.-t0)/60; # [min]

## lat/lon for GPS, INS (after zeroing), and Filter --------------------------
p1 = plot_map(mapS;map_color=:gray); # map background
plot_filt!(p1,traj,ins,filt_out;show_plot=false) # overlay GPS, INS, & filter
plot!(p1,legend=:topleft) # move as needed
# png(p1,"flight_path") # to save figure

## northing/easting INS error (after zeroing) --------------------------------
p2 = plot(xlab="time [min]",ylab="error [m]",legend=:topright) # move as needed
plot!(p2,tt,ins_out.n_err,lab="northing")
plot!(p2,tt,ins_out.e_err,lab="easting")

## northing/easting filter residuals -----------------------------------------
(p3,p4) = plot_filt_err(traj,filt_out,crlb_out)

## generating Google Earth map -----------------------------------------------
# map2kmz(mapS,"example_sim_map") # likely need to run twice

## generating Google Earth flight paths --------------------------------------
# path2kml(traj;path_name="example_sim_gps")
# path2kml(ins;path_name="example_sim_ins")
# path2kml(filt_out;path_name="example_sim_filt")
