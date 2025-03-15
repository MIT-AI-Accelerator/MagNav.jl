### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ c18afc43-9df3-45a1-b6e5-283ca8b26872
begin
	cd(@__DIR__)
	# uncomment line below to use local MagNav.jl (downloaded folder)
	# using Pkg; Pkg.activate("../"); Pkg.instantiate()
	using MagNav
	using CSV, DataFrames
	using Plots: plot, plot!
	using Random: seed!
	using Statistics: mean, median, std
	seed!(33); # for reproducibility
	include("dataframes_setup.jl"); # setup DataFrames
end;

# ╔═╡ 13ac32f0-3d77-11ee-3009-e70e93e2e71b
md"# Using the MagNav Package with Real SGL Flight Data
This file is best viewed in a [Pluto](https://plutojl.org/) notebook. To run it this way, from the MagNav.jl directory, do:
```julia
julia> using Pluto
julia> Pluto.run() # select & open notebook
```

This is a reactive notebook, so feel free to change any parameters of interest.
"

# ╔═╡ b1d3b1b3-db8d-4bb0-a884-d57f217fef24
md"## Import packages and DataFrames

The DataFrames listed below provide useful information about the flight data collected by Sander Geophysics Ltd. (SGL) & magnetic anomaly maps.

Dataframe  | Description
:--------- | :----------
`df_map`   | map files relevant for SGL flights
`df_cal`   | SGL calibration flight lines
`df_flight`| SGL flight files
`df_all`   | all flight lines
`df_nav`   | all *navigation-capable* flight lines
`df_event` | pilot-recorded in-flight events
"

# ╔═╡ 2f1e71ff-8f7f-4704-8e7d-b4fd2846f7ed
md"## Load flight and map data

Select Flight 1006 (see [readme](https://github.com/MIT-AI-Accelerator/MagNav.jl/blob/master/readmes/Flt1006_readme.txt)) & load the flight data. The full list of SGL flights is in `df_flight`.
"

# ╔═╡ da75fd96-1e6a-46bc-bcea-cdcc8c5f2a86
begin
	flight = :Flt1006 # select flight, full list in df_flight
	xyz    = get_XYZ(flight,df_flight) # load flight data
end;

# ╔═╡ 6824d21a-acca-4657-9a1e-18571ce6bdce
df_flight

# ╔═╡ a2dafd09-e9f7-4a58-b269-32fee40d602d
md"The `xyz` flight data struct is of type `MagNav.XYZ20` (for the 2020 SGL flight data collection), which is a subtype of `MagNav.XYZ` (the abstract type for any flight data in MagNav.jl). There are 76 fields, which can be accessed using dot notation. These fields are described in the docs, which are easily accessed by searching `MagNav.XYZ20` in the Live Docs in the lower right within Pluto.
"

# ╔═╡ 57aeef2a-2b7e-44f7-b1b7-c40736c38063
typeof(xyz)

# ╔═╡ a750b1aa-04d9-4f7a-a380-219ac69e8a0d
fieldnames(MagNav.XYZ20)

# ╔═╡ 1a3aba8d-3aa8-454b-9fb0-708f0bd38c42
md"Select the map & view the flight line options (`df_options`) for the selected flight & map. The full list of SGL flights is in `df_flight`, the full list of maps is in `df_map`, & the full list of navigation-capable flight lines is in `df_nav`.
"

# ╔═╡ c192c83b-deb3-4b6e-b0d9-97357e0b554c
begin
	map_name = :Eastern_395 # select map, full list in df_map
	df_options = df_nav[(df_nav.flight   .== flight  ) .&
	                    (df_nav.map_name .== map_name),:]
end

# ╔═╡ e001a13c-2662-4c2f-a89a-de6658fa81db
md"Select a flight line (row of `df_options`) & get the flight data Boolean indices (mask).
"

# ╔═╡ 7390ee91-6a92-4070-af8b-e3c003c3b5fb
begin
	line = df_options.line[1] # select flight line (row) from df_options
	ind  = get_ind(xyz,line,df_nav) # get Boolean indices
	# ind = get_ind(xyz;lines=[line]) # alternative
end;

# ╔═╡ f665ea95-dac3-4823-94af-c7ba58cd4401
md"Select a flight line (row of `df_cal`) & get the flight data Boolean indices (mask) for Tolles-Lawson calibration. The full list of calibration flight line options is in `df_cal`.
"

# ╔═╡ 5346a989-cf32-436c-9181-d7aff6dd44a1
begin
	TL_i   = 6 # select first calibration box of 1006.04
	TL_ind = get_ind(xyz;tt_lim=[df_cal.t_start[TL_i],df_cal.t_end[TL_i]])
end;

# ╔═╡ db7dc866-b889-4fb7-81d4-97cd8435636e
md"## Baseline plots of scalar and vector (fluxgate) magnetometer data

Setup for the baseline plots.
"

# ╔═╡ 3db31fc5-9ec2-4ca2-a5e0-6b478a7b29f9
begin
	show_plot    = false
	save_plot    = false
	detrend_data = true
end;

# ╔═╡ 6796c1ee-74a1-4d21-b68d-bd08929a817f
md"Uncompensated (raw) scalar magnetometers.

The goal of compensation is to remove the magnetic field corruption caused by the platform (aircraft). Here, Magnetometer 1 is placed on a stinger (boom), & has low noise before compensation. Magnetometers 4 & 5 look like good candidates, while 3 is far more challenging.
"

# ╔═╡ d02f285c-389c-4b9d-8fef-243253fcc8bb
b1 = plot_mag(xyz;ind,show_plot,save_plot,detrend_data,
              use_mags = [:mag_1_uc,:mag_3_uc,:mag_4_uc,:mag_5_uc])

# ╔═╡ 3f69ee39-d3cb-4458-8381-345b289f5d3f
md"Vector (fluxgate) magnetometer `d`.
"

# ╔═╡ 18189264-9abf-464a-9101-3f7f4f312690
b2 = plot_mag(xyz;ind,show_plot,save_plot,detrend_data,
              use_mags = [:flux_d]) # try changing to :flux_a, :flux_b, :flux_c

# ╔═╡ 5401fd57-4ac0-4ee3-ab9c-63746ebfd854
md"Magnetometer 1 compensation as provided in dataset by SGL.
"

# ╔═╡ 15a1e17a-fd89-486d-9ad9-0ae73c0c2b79
b3 = plot_mag(xyz;ind,show_plot,save_plot,
              use_mags = [:comp_mags])

# ╔═╡ ee2119b5-b61f-4bad-87ed-abb4c88194bc
md"Magnetometer 1 compensation using Tolles-Lawson with vector (fluxgate) magnetometer `d`.
"

# ╔═╡ 05ebfbab-deaa-43f3-965a-aeb91fec9e67
b4 = plot_mag_c(xyz,xyz;ind,show_plot,save_plot,detrend_data,
                ind_comp  = TL_ind,
                use_mags  = [:mag_1_uc],
                use_vec   = :flux_d,
                plot_diff = true)

# ╔═╡ ab8b23b1-63f8-4306-97a8-2e9d6f2707c4
md"Magnetometer 1 Welch power spectral density (PSD).
"

# ╔═╡ 1eee0857-1278-412b-89fe-f6c3dc094f6f
b5 = plot_frequency(xyz;ind,show_plot,save_plot,detrend_data,
                    field     = :mag_1_uc,
                    freq_type = :PSD)

# ╔═╡ 0d362625-7d94-4131-9254-a336e6a791b1
md"Magnetometer 1 spectrogram.
"

# ╔═╡ 90db7042-37b7-4751-ad7d-b1fabb305a5a
b6 = plot_frequency(xyz;ind,show_plot,save_plot,detrend_data,
                    field     = :mag_1_uc,
                    freq_type = :spec)

# ╔═╡ 1233e336-3f11-44e3-b136-08c724f12e0f
md"## Tolles-Lawson calibration and compensation

Fit the Tolles-Lawson coefficients for uncompensated scalar magnetometers `1-5` with vector (fluxgate) magnetometer `d`.
"

# ╔═╡ 162f9444-146b-41dd-ba2c-729beff44306
begin
	λ       = 0.025   # ridge parameter for ridge regression
	use_vec = :flux_d # selected vector (flux) magnetometer
	flux    = getfield(xyz,use_vec) # load Flux D data
	TL_d_1  = create_TL_coef(flux,xyz.mag_1_uc,TL_ind;λ=λ) # Flux D & Mag 1
	TL_d_2  = create_TL_coef(flux,xyz.mag_2_uc,TL_ind;λ=λ) # Flux D & Mag 2
	TL_d_3  = create_TL_coef(flux,xyz.mag_3_uc,TL_ind;λ=λ) # Flux D & Mag 3
	TL_d_4  = create_TL_coef(flux,xyz.mag_4_uc,TL_ind;λ=λ) # Flux D & Mag 4
	TL_d_5  = create_TL_coef(flux,xyz.mag_5_uc,TL_ind;λ=λ) # Flux D & Mag 5
end;

# ╔═╡ ebf894ec-5140-42cb-b2b5-1b85f96b7a75
md"Get the relevant scalar magnetometer data.
"

# ╔═╡ c54b97cd-565f-4d5a-980a-44a3e1d18355
begin
	A = create_TL_A(flux,ind)
	mag_1_sgl = xyz.mag_1_c[ind]
	mag_1_uc  = xyz.mag_1_uc[ind]
	mag_2_uc  = xyz.mag_2_uc[ind]
	mag_3_uc  = xyz.mag_3_uc[ind]
	mag_4_uc  = xyz.mag_4_uc[ind]
	mag_5_uc  = xyz.mag_5_uc[ind]
end;

# ╔═╡ d83b4b7a-e8c9-4dbe-8682-592f0ac5e1b6
md"Create the Tolles-Lawson `A` matrices & perform Tolles-Lawson compensation.
"

# ╔═╡ 9116a5b5-7b5f-4c38-8a48-e854584a8ada
begin
	mag_1_c = mag_1_uc - detrend(A*TL_d_1;mean_only=true)
	mag_2_c = mag_2_uc - detrend(A*TL_d_2;mean_only=true)
	mag_3_c = mag_3_uc - detrend(A*TL_d_3;mean_only=true)
	mag_4_c = mag_4_uc - detrend(A*TL_d_4;mean_only=true)
	mag_5_c = mag_5_uc - detrend(A*TL_d_5;mean_only=true)
end;

# ╔═╡ 01779ec7-0088-4b37-bda4-7fccf4dbb548
md"Prepare the flight data for the navigation filter, load the map data, & get the map interpolation function & map values along the selected flight line. The map values are then corrected for diurnal & core ([IGRF](https://www.ncei.noaa.gov/products/international-geomagnetic-reference-field)) magnetic fields.
"

# ╔═╡ 3f2dd431-6c5b-403f-9a96-320d8ab6ef17
begin
	traj = get_traj(xyz,ind) # trajectory (GPS) struct
	ins  = get_ins( xyz,ind;N_zero_ll=1) # INS struct, "zero" lat/lon to match first `traj` data point
	mapS = get_map(map_name,df_map) # load map data
	# get map values & map interpolation function
	(map_val,itp_mapS) = get_map_val(mapS,traj;return_itp=true)
	map_val += (xyz.diurnal + xyz.igrf)[ind] # add in diurnal & core (IGRF)
end;

# ╔═╡ 50a6302c-b1ed-4be3-8af7-1c641715b25f
md"Map to magnetometer (standard deviation) errors. Magnetometer `1` is great (stinger), while `2` is unusable & `3-5` are in-between.
"

# ╔═╡ 110e5de7-9011-4ba6-83d6-3443b6845dc6
begin
	println("Mag 1: ",round(std(map_val-mag_1_c),digits=2))
	println("Mag 2: ",round(std(map_val-mag_2_c),digits=2))
	println("Mag 3: ",round(std(map_val-mag_3_c),digits=2))
	println("Mag 4: ",round(std(map_val-mag_4_c),digits=2))
	println("Mag 5: ",round(std(map_val-mag_5_c),digits=2))
end

# ╔═╡ e5adfd84-4727-4839-bef0-9365d035249f
md"## Navigation

Create a navigation filter model. Only the most relevant navigation filter parameters are shown.
"

# ╔═╡ 1c3579bc-bf1f-45ad-bdfb-df4683ace128
(P0,Qd,R) = create_model(traj.dt,traj.lat[1];
                         init_pos_sigma = 0.1,
                         init_alt_sigma = 1.0,
                         init_vel_sigma = 1.0,
                         meas_var       = 5^2, # increase if mag_use is bad
                         fogm_sigma     = 3,
                         fogm_tau       = 180);

# ╔═╡ 690c1b98-4201-4a7a-8c1d-04984c3e0307
md"Run the navigation filter (EKF), determine the Cramér–Rao lower bound (CRLB), & extract output data.
"

# ╔═╡ 74b69225-a541-49b8-98fc-cacbf22f187a
begin
	mag_use = mag_1_c # selected magnetometer, modify & see what happens
	(crlb_out,ins_out,filt_out) = run_filt(traj,ins,mag_use,itp_mapS,:ekf;
	                                       P0,Qd,R,core=true)
end;

# ╔═╡ f01997d2-0ea7-4be3-acd8-533580ccb82d
md"Plotting setup.
"

# ╔═╡ 8167ba6a-fff4-4c79-9ee6-57d6f594bb18
begin
	t0  = traj.tt[1]/60    # [min]
	tt  = traj.tt/60 .- t0 # [min]
	dpi = 200
end;

# ╔═╡ 8c92cea8-5cec-42a6-838d-ebc083e0d9b4
md"Compensated scalar magnetometers.
"

# ╔═╡ e0a21a3c-b48c-458e-a4eb-7a726e77b2b2
begin
	p1 = plot(xlab="time [min]",ylab="magnetic field [nT]",legend=:topleft,dpi=dpi)
	plot!(p1,tt,detrend(mag_1_uc ),lab="SGL raw Mag 1" ,color=:cyan,lw=2)
	plot!(p1,tt,detrend(mag_1_sgl),lab="SGL comp Mag 1",color=:blue,lw=2)
	plot!(p1,tt,detrend(mag_1_c  ),lab="MIT comp Mag 1",color=:red ,lw=2,ls=:dash)
	# plot!(p1,tt,detrend(mag_2_c  ),lab="MIT comp Mag 2",color=:purple) # bad
	plot!(p1,tt,detrend(mag_3_c  ),lab="MIT comp Mag 3",color=:green)
	plot!(p1,tt,detrend(mag_4_c  ),lab="MIT comp Mag 4",color=:black)
	plot!(p1,tt,detrend(mag_5_c  ),lab="MIT comp Mag 5",color=:orange)
	# png(p1,"comp_prof_1") # to save figure
end

# ╔═╡ 4036432b-2bea-4463-908a-5f5dcc5544cf
md"Position (lat & lot) for trajectory (GPS), INS (after zeroing), & navigation filter.
"

# ╔═╡ 228b18b7-6374-4ceb-a2b1-b188aa2f593f
begin
	p2 = plot_map(mapS;map_color=:gray) # map background
	plot_filt!(p2,traj,ins,filt_out;show_plot=false) # overlay GPS, INS, & filter
	plot!(p2,legend=:topleft) # move as needed
end

# ╔═╡ 60c2bb16-a0d4-42f2-ba4a-f42dd70f2611
md"Northing & easting INS error (after zeroing).
"

# ╔═╡ 9bc21e37-9911-4b2e-8460-99ebd8256673
begin
	p3 = plot(xlab="time [min]",ylab="error [m]",legend=:topleft,dpi=dpi)
	plot!(p3,tt,ins_out.n_err,lab="northing")
	plot!(p3,tt,ins_out.e_err,lab="easting")
end

# ╔═╡ de62fdda-a107-4283-9140-43dcf2dfb4bc
(p4,p5) = plot_filt_err(traj,filt_out,crlb_out;show_plot,save_plot);

# ╔═╡ 5a71c032-287a-4c70-90dc-e922cf11c606
md"Northing navigation filter residuals.
"

# ╔═╡ 6439d401-07a5-4d31-81e3-411e3aacd0bb
p4

# ╔═╡ 0cf07d89-6be4-491c-9c07-9ce4d94dee46
md"Easting navigation filter residuals.
"

# ╔═╡ 6c4bf761-18f9-4724-8101-910f4f5ad65e
p5

# ╔═╡ 123a6d91-8950-40ef-804b-86021f9a0207
md"Map values vs magnetometer measurements.
"

# ╔═╡ 4bb44a93-c17d-4667-b160-57b62c4c2914
p6 = plot_mag_map(traj,mag_use,itp_mapS)

# ╔═╡ d813053b-d74a-448c-b25e-465c9e883394
md"## In-flight events
"

# ╔═╡ 4c1c21b1-6ee8-4099-893f-9a473cba2d2d
md"Magnetometers with in-flight event(s) marked. This may be useful for understanding errors in the magnetic signal compared to the map. The full list of pilot-recorded in-flight events is in `df_event`.
"

# ╔═╡ 289d9265-a8a0-410b-a858-3698bd4cae37
begin
	p7 = plot(xlab="time [min]",ylab="magnetic field [nT]",dpi=dpi)
	plot!(p7,tt,mag_1_uc,lab="mag_1_uc")
	# plot!(p7,tt,mag_2_uc,lab="mag_2_uc") # bad
	plot!(p7,tt,mag_3_uc,lab="mag_3_uc")
	plot!(p7,tt,mag_4_uc,lab="mag_4_uc")
	plot!(p7,tt,mag_5_uc,lab="mag_5_uc")
	plot_events!(p7,flight,df_event;t0=t0,t_units=:min)
    display(p7)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
MagNav = "f91b31a4-be4d-40e3-b767-4b8c09c10076"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
CSV = "~0.10.15"
DataFrames = "~1.7.0"
MagNav = "~1.3.0"
Plots = "~1.40.10"
Random = "~1.11.0"
Statistics = "~1.11.1"
"""

# ╔═╡ Cell order:
# ╟─13ac32f0-3d77-11ee-3009-e70e93e2e71b
# ╟─b1d3b1b3-db8d-4bb0-a884-d57f217fef24
# ╠═c18afc43-9df3-45a1-b6e5-283ca8b26872
# ╟─2f1e71ff-8f7f-4704-8e7d-b4fd2846f7ed
# ╠═da75fd96-1e6a-46bc-bcea-cdcc8c5f2a86
# ╠═6824d21a-acca-4657-9a1e-18571ce6bdce
# ╟─a2dafd09-e9f7-4a58-b269-32fee40d602d
# ╠═57aeef2a-2b7e-44f7-b1b7-c40736c38063
# ╠═a750b1aa-04d9-4f7a-a380-219ac69e8a0d
# ╟─1a3aba8d-3aa8-454b-9fb0-708f0bd38c42
# ╠═c192c83b-deb3-4b6e-b0d9-97357e0b554c
# ╟─e001a13c-2662-4c2f-a89a-de6658fa81db
# ╠═7390ee91-6a92-4070-af8b-e3c003c3b5fb
# ╟─f665ea95-dac3-4823-94af-c7ba58cd4401
# ╠═5346a989-cf32-436c-9181-d7aff6dd44a1
# ╟─db7dc866-b889-4fb7-81d4-97cd8435636e
# ╠═3db31fc5-9ec2-4ca2-a5e0-6b478a7b29f9
# ╟─6796c1ee-74a1-4d21-b68d-bd08929a817f
# ╠═d02f285c-389c-4b9d-8fef-243253fcc8bb
# ╟─3f69ee39-d3cb-4458-8381-345b289f5d3f
# ╠═18189264-9abf-464a-9101-3f7f4f312690
# ╟─5401fd57-4ac0-4ee3-ab9c-63746ebfd854
# ╠═15a1e17a-fd89-486d-9ad9-0ae73c0c2b79
# ╟─ee2119b5-b61f-4bad-87ed-abb4c88194bc
# ╠═05ebfbab-deaa-43f3-965a-aeb91fec9e67
# ╟─ab8b23b1-63f8-4306-97a8-2e9d6f2707c4
# ╠═1eee0857-1278-412b-89fe-f6c3dc094f6f
# ╟─0d362625-7d94-4131-9254-a336e6a791b1
# ╠═90db7042-37b7-4751-ad7d-b1fabb305a5a
# ╟─1233e336-3f11-44e3-b136-08c724f12e0f
# ╠═162f9444-146b-41dd-ba2c-729beff44306
# ╟─ebf894ec-5140-42cb-b2b5-1b85f96b7a75
# ╠═c54b97cd-565f-4d5a-980a-44a3e1d18355
# ╟─d83b4b7a-e8c9-4dbe-8682-592f0ac5e1b6
# ╠═9116a5b5-7b5f-4c38-8a48-e854584a8ada
# ╟─01779ec7-0088-4b37-bda4-7fccf4dbb548
# ╠═3f2dd431-6c5b-403f-9a96-320d8ab6ef17
# ╟─50a6302c-b1ed-4be3-8af7-1c641715b25f
# ╠═110e5de7-9011-4ba6-83d6-3443b6845dc6
# ╟─e5adfd84-4727-4839-bef0-9365d035249f
# ╠═1c3579bc-bf1f-45ad-bdfb-df4683ace128
# ╟─690c1b98-4201-4a7a-8c1d-04984c3e0307
# ╠═74b69225-a541-49b8-98fc-cacbf22f187a
# ╟─f01997d2-0ea7-4be3-acd8-533580ccb82d
# ╠═8167ba6a-fff4-4c79-9ee6-57d6f594bb18
# ╟─8c92cea8-5cec-42a6-838d-ebc083e0d9b4
# ╠═e0a21a3c-b48c-458e-a4eb-7a726e77b2b2
# ╟─4036432b-2bea-4463-908a-5f5dcc5544cf
# ╠═228b18b7-6374-4ceb-a2b1-b188aa2f593f
# ╟─60c2bb16-a0d4-42f2-ba4a-f42dd70f2611
# ╠═9bc21e37-9911-4b2e-8460-99ebd8256673
# ╠═de62fdda-a107-4283-9140-43dcf2dfb4bc
# ╟─5a71c032-287a-4c70-90dc-e922cf11c606
# ╠═6439d401-07a5-4d31-81e3-411e3aacd0bb
# ╟─0cf07d89-6be4-491c-9c07-9ce4d94dee46
# ╠═6c4bf761-18f9-4724-8101-910f4f5ad65e
# ╟─123a6d91-8950-40ef-804b-86021f9a0207
# ╠═4bb44a93-c17d-4667-b160-57b62c4c2914
# ╟─d813053b-d74a-448c-b25e-465c9e883394
# ╟─4c1c21b1-6ee8-4099-893f-9a473cba2d2d
# ╠═289d9265-a8a0-410b-a858-3698bd4cae37
# ╟─00000000-0000-0000-0000-000000000001
