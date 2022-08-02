## full example using the MagNav package with real SGL data
cd(@__DIR__)
include("common_setup.jl"); # load common-use packages, DataFrames, and Dicts

#* flight and map data section ===============================================
## specify flight, load flight data ------------------------------------------
flight = :Flt1006 # specify flight #* full list of SGL flights in df_flight
xyz    = get_XYZ(flight,df_flight); # load flight data

## specify map, see line options ---------------------------------------------
map_name = :Eastern_395 # specify map #* full list of maps in df_map
#* full list of navigation-capable lines in df_nav
#* df_options is the portion of df_nav valid for the selected flight & map
df_options = df_nav[(df_nav.flight   .== flight  ) .& 
                    (df_nav.map_name .== map_name),:] # line options

## choose line, get flight data indices --------------------------------------
line = df_options.line[1] # select line (row) from df_options
ind  = get_ind(xyz,line,df_nav); # get indicies
# ind = get_ind(xyz;lines=[line]); # alternate way to get indicies

## choose line, get calibration box indices ----------------------------------
TL_i   = 6 # first calibration box of 1006.04
TL_ind = get_ind(xyz;tt_lim=[df_comp.t_start[TL_i],df_comp.t_end[TL_i]]);

#* baseline plots of scalar and vector (fluxgate) magnetometer data ==========
show_plot = true
save_plot = false
use_mags  = [:mag_1_uc,:mag_4_uc,:mag_5_uc]

## scalar magnetometer
b1 = plot_mag(xyz;ind,show_plot,save_plot,
              use_mags=use_mags,
              detrend_data=true,
              file_name="scalar_mags");

## vector magnetometer (fluxgate)
b2 = plot_mag(xyz;ind,show_plot,save_plot,
              use_mags=[:flux_b],
              detrend_data=true,
              file_name="vector_mag_b");

## mag_1 compensation (provided)
b3 = plot_mag(xyz;ind,show_plot,save_plot,
              use_mags=[:comp_mags],
              file_name="scalar_mags_comp");

## scalar magnetometer compensations (using Tolles-Lawson)
b4 = plot_mag_c(xyz,xyz;ind=ind,ind_comp=TL_ind,show_plot,save_plot,
                use_mags=[:mag_1_uc],
                use_vec=:flux_b,
                detrend_data=true,
                plot_diff=true,
                file_name="scalar_mags_comp");

#* plot frequency domain data ================================================
## Welch power spectral density (PSD)
b5 = plot_frequency(xyz;ind,show_plot,save_plot,
                    field=:mag_1_uc,
                    freq_type=:PSD,
                    file_name="mag_1_uc_PSD");

## spectrogram
b6 = plot_frequency(xyz;ind,show_plot,save_plot,
                    field=:mag_1_uc,
                    freq_type=:spec,
                    file_name="mag_1_uc_spec");

## create Tolles-Lawson coefficients -----------------------------------------
λ       = 0.025
use_vec = :flux_d
flux    = getfield(xyz,use_vec)
TL_b_1  = create_TL_coef(flux,xyz.mag_1_uc,TL_ind;λ=λ);
TL_b_2  = create_TL_coef(flux,xyz.mag_2_uc,TL_ind;λ=λ);
TL_b_3  = create_TL_coef(flux,xyz.mag_3_uc,TL_ind;λ=λ);
TL_b_4  = create_TL_coef(flux,xyz.mag_4_uc,TL_ind;λ=λ);
TL_b_5  = create_TL_coef(flux,xyz.mag_5_uc,TL_ind;λ=λ);

## get relevant scalar magnetometer data -------------------------------------
A = create_TL_A(flux,ind);
mag_1_sgl = xyz.mag_1_c[ind];
mag_1_uc  = xyz.mag_1_uc[ind];
mag_2_uc  = xyz.mag_2_uc[ind];
mag_3_uc  = xyz.mag_3_uc[ind];
mag_4_uc  = xyz.mag_4_uc[ind];
mag_5_uc  = xyz.mag_5_uc[ind];

## create Tolles-Lawson `A` matrix and perform compensation ------------------
mag_1_c = mag_1_uc - detrend(A*TL_b_1;mean_only=true);
mag_2_c = mag_2_uc - detrend(A*TL_b_2;mean_only=true);
mag_3_c = mag_3_uc - detrend(A*TL_b_3;mean_only=true);
mag_4_c = mag_4_uc - detrend(A*TL_b_4;mean_only=true);
mag_5_c = mag_5_uc - detrend(A*TL_b_5;mean_only=true);

## prepare data for filter, load map data, get interpolation -----------------
traj = get_traj(xyz,ind); # get trajectory (gps) struct
ins  = get_ins(xyz,ind;N_zero_ll=1); # get INS struct
mapS = get_map(map_name,df_map); # load map data
mapS.alt > 0 && (mapS = upward_fft(mapS,mean(traj.alt))); #* do not do with drape map
itp_mapS = map_interpolate(mapS); # get interpolation
map_val  = itp_mapS.(traj.lon,traj.lat) + (xyz.diurnal + xyz.igrf)[ind];

println("mag 1: ",round(std(map_val-mag_1_c),digits=2))
println("mag 2: ",round(std(map_val-mag_2_c),digits=2))
println("mag 3: ",round(std(map_val-mag_3_c),digits=2))
println("mag 4: ",round(std(map_val-mag_4_c),digits=2))
println("mag 5: ",round(std(map_val-mag_5_c),digits=2))

#* navigation filter section =================================================
## create filter model, only showing most relevant filter parameters (fp) here
(P0,Qd,R) = create_model(traj.dt,traj.lat[1];
                         init_pos_sigma = 0.1,
                         init_alt_sigma = 1.0,
                         init_vel_sigma = 1.0,
                         meas_var       = 5^2, #* increase if mag_use is bad
                         fogm_sigma     = 3,
                         fogm_tau       = 180);

## run filter (EKF or MPF), determine CRLB, & extract output -----------------
mag_use = mag_1_c; # selected magnetometer #* modify here to see what happens
(crlb_out,ins_out,filt_out) = run_filt(traj,ins,mag_use,itp_mapS,:ekf;
                                       P0,Qd,R,core=true);
#* type "?" then "crlb_out" or "ins_out" or "filt_out" in REPL to see fields

#* plotting section ==========================================================
t0 = traj.tt[1];
tt = (traj.tt.-t0)/60; # [min]

## comparison of magnetometers after compensation ----------------------------
p1 = plot(xlab="time [min]",ylab="magnetic field [nT]",legend=:topleft,dpi=300)
plot!(p1,tt,detrend(mag_1_uc ),lab="SGL raw Mag 1" ,color=:cyan,lw=2)
plot!(p1,tt,detrend(mag_1_sgl),lab="SGL comp Mag 1",color=:blue,lw=2)
plot!(p1,tt,detrend(mag_1_c  ),lab="MIT comp Mag 1",color=:red ,lw=2,ls=:dash)
# plot!(p1,tt,detrend(mag_2_c  ),lab="MIT comp Mag 2",color=:purple) #* bad
plot!(p1,tt,detrend(mag_3_c  ),lab="MIT comp Mag 3",color=:green)
plot!(p1,tt,detrend(mag_4_c  ),lab="MIT comp Mag 4",color=:black)
plot!(p1,tt,detrend(mag_5_c  ),lab="MIT comp Mag 5",color=:orange)
# png(p1,"comp_prof_1") # to save figure

## lat/lon for GPS, INS (after zeroing), and Filter --------------------------
p2 = plot_map(mapS;map_color=:gray,legend=false); # map background
plot_filt!(p2,traj,ins,filt_out;show_plot=false);
plot!(p2,legend=:topleft)
# png(p2,"flight_path") # to save figure

## northing/easting INS error (after zeroing) --------------------------------
p3 = plot(xlab="time [min]",ylab="error [m]",legend=:topleft,dpi=200)
plot!(p3,tt,ins_out.n_err,lab="northing")
plot!(p3,tt,ins_out.e_err,lab="easting")

## northing/easting filter residuals -----------------------------------------
(p4,p5) = plot_filt_err(traj,filt_out,crlb_out)

## map vs magnetometer measurement -------------------------------------------
p6 = plot_mag_map(traj,mag_use,itp_mapS)

## magnetometers with event(s) marked ----------------------------------------
p7 = plot(xlab="time [min]",ylab="magnetic field [nT]",dpi=200)
plot!(p7,tt,mag_1_uc,lab="mag_1_uc")
plot!(p7,tt,mag_3_uc,lab="mag_3_uc")
plot!(p7,tt,mag_5_uc,lab="mag_5_uc")
plot_events!(p7,flight,df_event;t0=t0)
display(p7)
