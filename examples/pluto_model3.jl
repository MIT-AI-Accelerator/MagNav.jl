### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 22982d8e-240c-11ee-2e0c-bda2a93a4ab0
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

# ╔═╡ e289486a-57ed-4eeb-9ec9-6500f0bc563b
md"# Model 3 Training & Explainability
This file is best viewed in a [Pluto](https://plutojl.org/) notebook. To run it this way, from the MagNav.jl directory, do:
```julia
julia> using Pluto
julia> Pluto.run() # select & open notebook
```

This is a reactive notebook, so feel free to change any parameters of interest, like adding/removing features to the model, changing the number of training epochs, switching between versions of model 3 (scalar is `:m3s`, vector is `:m3v`), or different training & testing lines.
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

# ╔═╡ 3a55962c-bd1b-410c-b98a-3130fc11ee11
md"## Train a (linear) Tolles-Lawson model

Select Flight 1006 (see [readme](https://github.com/MIT-AI-Accelerator/MagNav.jl/blob/master/readmes/Flt1006_readme.txt)), load the flight data, & get the Boolean indices for a specific calibration flight line that is used to fit the Tolles-Lawson coefficients. The full list of SGL flights is in `df_flight`, & the full list of calibration flight line options is in `df_cal`.
"

# ╔═╡ bf9f72f0-c351-48d3-a811-418ee965073c
flight_train = :Flt1006; # select flight, full list in df_flight

# ╔═╡ 9d11d4be-9ba7-44ab-a4ff-f0fdf95b2171
reorient_vec = true; # align vector magnetometers with aircraft body frame

# ╔═╡ 663f7b8c-dfc6-445d-9c54-1141856f2a66
xyz_train = get_XYZ(flight_train, df_flight;
					reorient_vec=reorient_vec, silent=true); # load flight data

# ╔═╡ 99405ae1-580a-4fd3-a860-13cc5b22b045
begin # select a calibration flight line to train Tolles-Lawson on
	println(df_cal[df_cal.flight .== flight_train, :])
	TL_i   = 6 # select first calibration box of 1006.04
	TL_ind = get_ind(xyz_train;tt_lim=[df_cal.t_start[TL_i],df_cal.t_end[TL_i]])
end;

# ╔═╡ 3d438be1-537b-4c13-a422-7f5f48479e62
md"Select magnetometers & parameters to be used with Tolles-Lawson.
"

# ╔═╡ 6d4a87c8-41d7-4478-b52a-4dc5a1ae18ea
begin # try modifying these parameters
	use_mag = :mag_4_uc
	use_vec = :flux_d
	terms   = [:permanent,:induced,:eddy]
	flux    = getfield(xyz_train,use_vec)
	comp_params_lin_init = LinCompParams(model_type   = :TL,
		                                 y_type       = :e,
		                                 use_mag      = use_mag,
		                                 use_vec      = use_vec,
		                                 terms_A      = terms,
		                                 sub_diurnal  = false,
		                                 sub_igrf     = false,
		                                 reorient_vec = reorient_vec)
end;

# ╔═╡ c5b6b439-e962-4c9d-9b02-339657fb266b
begin
	tt = (xyz_train.traj.tt[TL_ind] .- xyz_train.traj.tt[TL_ind][1]) / 60
	p1 = plot(xlab="time [min]",ylab="magnetic field [nT]",dpi=200,
		      ylim=(51000,55000))
	plot!(p1, tt, xyz_train.igrf[TL_ind],     lab="IGRF core field")
	plot!(p1, tt, xyz_train.mag_1_c[TL_ind],  lab="compensated tail stinger")
	plot!(p1, tt, xyz_train.mag_4_uc[TL_ind], lab="uncompensated Mag 4")
	plot!(p1, tt, xyz_train.flux_a.t[TL_ind], lab="vector Flux A, total field")
end

# ╔═╡ ae1acc31-19db-4e94-85dc-e6274186978e
begin # Tolles-Lawson calibration
	(comp_params_lin, _, _, err_TL) =
		comp_train(comp_params_lin_init, xyz_train, TL_ind)
	TL_a_4 = comp_params_lin.model[1]
end;

# ╔═╡ 7a20ef62-f352-4fc7-9061-13fb293bb0bf
md"Select a flight line from Flight 1006 (see [readme](https://github.com/MIT-AI-Accelerator/MagNav.jl/blob/master/readmes/Flt1006_readme.txt)) to compare different compensation models. The full list of navigation-capable flight lines is in `df_nav`.
"

# ╔═╡ 39bdbe6a-52ae-44d2-8e80-2e7d2a75e322
begin
	flight_test = :Flt1006 # select flight, full list in df_flight
	println(df_nav[df_nav.flight .== flight_test, :])
	xyz_test   = get_XYZ(flight_test, df_flight;
	                     reorient_vec=reorient_vec, silent=true) # load flight data
	lines_test = [1006.08]
	ind_test   = get_ind(xyz_test, lines_test, df_nav) # get Boolean indices
end;

# ╔═╡ 7a59dc41-21f1-4a5e-9f8b-06cacca8845b
md"Perform Tolles-Lawson compensation on `lines_test`. The full list of SGL flights is in `df_flight`, the full list of maps is in `df_map`, & the full list of flight lines is in `df_all`.
"

# ╔═╡ 9ed61357-346b-49de-a1ed-8849db041ade
comp_test(comp_params_lin, lines_test, df_all, df_flight, df_map);

# ╔═╡ a9cac83d-03b3-4793-8da4-f45563091bd0
md"## Model 3 training

Create a low-pass (not the default, bandpass)-filtered version of the coefficients to preserve the Earth field, which helps to remove a bias term in the Tolles-Lawson (TL) fitting.

This shows how the bandpass filter causes a bias in the compensation (filtering out the Earth field). That is not a problem for navigating, but it is preferred to not have the neural network learn a simple correction that could be accounted for more simply by Tolles-Lawson.
"

# ╔═╡ f492ee3d-c097-4937-9552-e7d2632a5e50
 # create Tolles-Lawson coefficients with use_vec & use_mag
TL_coef = create_TL_coef(getfield(xyz_train,use_vec),
	                     getfield(xyz_train,use_mag)-xyz_train.mag_1_c, TL_ind;
	                     terms=terms, pass1=0.0, pass2=0.9);

# ╔═╡ 7e15f297-64a4-4ee6-a0ab-65d959354f29
begin # create Tolles-Lawson `A` matrix & perform compensation
	A = create_TL_A(flux,TL_ind)
	mag_1_sgl_TL_train = xyz_train.mag_1_c[TL_ind]
	mag_4_uc_TL_train  = xyz_train.mag_4_uc[TL_ind]
	mag_4_c_bpf        = mag_4_uc_TL_train - A*TL_a_4
	mag_4_c_lpf        = mag_4_uc_TL_train - A*TL_coef
	p2 = plot(xlab="time [min]",ylab="magnetic field [nT]",dpi=200,
		      ylim=(52000,55000))
	plot!(p2, tt, mag_1_sgl_TL_train, lab="ground truth")
	plot!(p2, tt, mag_4_c_lpf,        lab="low-pass filtered TL")
	plot!(p2, tt, mag_4_c_bpf,        lab="bandpass filtered TL")
end

# ╔═╡ 239b60aa-0e97-4871-a9a5-64cd802f4bde
md"### Training data & parameters

Set the training lines (all from the same flight in this example), select a model type, features for the neural network, & intialize all of this using the `NNCompParams` data structure, which keeps all these configuration/learning settings in one place. For model 3, it is important to not normalize Tolles-Lawson `A` matrix, unlike the `x` input data matrix & `y` target vector.
"

# ╔═╡ 21828f95-98f3-4271-9a57-121252065741
begin
	lines_train = [1006.03, 1006.04, 1006.05, 1006.06]
	ind_train   = get_ind(xyz_train, lines_train, df_all) # get Boolean indices
	model_type  = :m3s
	features    = [:mag_4_uc, :lpf_cur_com_1, :lpf_cur_strb, :lpf_cur_outpwr, :lpf_cur_ac_lo, :TL_A_flux_a]
end;

# ╔═╡ 4d75b716-bc93-42f5-8b1a-f5550e7de276
comp_params_init = NNCompParams(features_setup = features,
                                model_type     = model_type,
                                y_type         = :d,
                                use_mag        = use_mag,
                                use_vec        = use_vec,
                                terms          = [:permanent,:induced,:eddy],
                                terms_A        = [:permanent,:induced,:eddy],
                                sub_diurnal    = false,
                                sub_igrf       = false,
                                bpf_mag        = false,
                                reorient_vec   = reorient_vec,
                                norm_type_A    = :none,
                                norm_type_x    = :normalize,
                                norm_type_y    = :normalize,
                                TL_coef        = TL_coef,
                                η_adam         = 0.001,
                                epoch_adam     = 100,
                                epoch_lbfgs    = 0,
                                hidden         = [8,4],
                                batchsize      = 2048,
                                frac_train     = 13/17);

# ╔═╡ 4171fef9-650b-4f37-ae5d-7084d54a0be0
# neural network calibration
(comp_params, y_train, y_train_hat, err_train, _) =
	comp_train(comp_params_init, xyz_train, ind_train;
			   xyz_test = xyz_test,
			   ind_test = ind_test);

# ╔═╡ 0c839441-909d-4095-a4eb-4dfc92eb2121
md"Evaluate test line performance & return additional terms that are useful for visualizing the compensation performance
"

# ╔═╡ 0a1b7102-59e0-4e2a-a45c-5c21d0039b88
# neural network compensation
(TL_perm, TL_induced, TL_eddy, TL_aircraft, B_unit, _, y_nn, _, y, y_hat, _, _) =
	comp_m3_test(comp_params, lines_test, df_nav, df_flight, df_map);

# ╔═╡ 5f88bca1-ecc9-45b2-8164-0f6965b81088
md"## Navigation performance

Ultimately, the compensated magnetometer values are used with a navigation algorithm, an extended Kalman filter (EKF) here. To prepare the flight data & map for the navigation filter, a few more objects are needed:
* `traj`: trajectory (GPS) data structure
* `ins`: INS data structure
* `itp_mapS`: interpolated map to pass to the filter to use for measurement evaluations
* (`P0`, `Qd`, `R`): initialized covariance & noise matrices for the filter
"

# ╔═╡ ed1f8baa-d228-4f66-a979-527f68acccdc
begin
	map_name = df_nav[df_nav.line .== lines_test[1], :map_name][1]
	traj = get_traj(xyz_test,ind_test) # trajectory (GPS) struct
	ins  = get_ins(xyz_test,ind_test;N_zero_ll=1) # INS struct, "zero" lat/lon to match first `traj` data point
	mapS = get_map(map_name,df_map) # load map data
	# get map values & map interpolation function
	(map_val,itp_mapS) = get_map_val(mapS,traj;return_itp=true)
end;

# ╔═╡ 4e01e107-5d58-409d-b08c-23135b4c28ba
md"### Compensate the magnetometer

Compute the compensated magnetometer values consistent with the `:d` setting in `NNCompParams`, that is, a correction has been learned that can be removed from the uncompensated measurement, producing the compensated measurement. The figure below illustrates performance as compared to the uncompensated figure above.

An assumed constant map-magnetometer bias is removed here, which is approximated at the first measurement. The diurnal & core ([IGRF](https://www.ncei.noaa.gov/products/international-geomagnetic-reference-field)) fields must be included to leave only the nearly constant bias, if any.
"

# ╔═╡ 566bb8eb-23d3-473d-8e79-1d02536dfdd5
begin
	mag_4_c = xyz_test.mag_4_uc[ind_test] - y_hat # neural network compensation
	mag_4_c .+= (map_val + (xyz_test.diurnal + xyz_test.igrf)[ind_test] - mag_4_c)[1]
	# first-order Gauss-Markov noise values
	(sigma,tau) = get_autocor(mag_4_c - map_val)
end

# ╔═╡ 987292fe-4e58-4cc9-bfe2-734c0f4174f8
begin
	tt_test = (xyz_test.traj.tt[ind_test] .- xyz_test.traj.tt[ind_test][1]) / 60
	p3 = plot(xlab="time [min]",ylab="magnetic field [nT]",dpi=200,
		      ylim=(52900,53700))
	plot!(p3, tt_test, map_val + (xyz_test.diurnal + xyz_test.igrf)[ind_test],
												   lab="map value")
	plot!(p3, tt_test, xyz_test.mag_1_c[ind_test], lab="compensated tail stinger")
	plot!(p3, tt_test, mag_4_c,                    lab="compensated Mag 4")
end

# ╔═╡ d0fcbd95-002b-4e67-ad06-15ae52f27ede
md"### Navigation

Create a navigation filter model. Only the most relevant navigation filter parameters are shown.
"

# ╔═╡ 70b8e7ff-71ee-489b-817e-1d4ea3004355
(P0,Qd,R) = create_model(traj.dt,traj.lat[1];
                         init_pos_sigma = 0.1,
                         init_alt_sigma = 1.0,
                         init_vel_sigma = 1.0,
                         meas_var       = sigma^2, # increase if mag_use is bad
                         fogm_sigma     = sigma,
                         fogm_tau       = tau);

# ╔═╡ a56f0b50-0b39-4400-928d-f485af206d23
begin
	mag_use = mag_4_c
	(crlb_out,ins_out,filt_out) = run_filt(traj,ins,mag_use,itp_mapS,:ekf;
	                                       P0,Qd,R,core=true)
	drms_out = round(Int,sqrt(mean(filt_out.n_err.^2+filt_out.e_err.^2)))
end;

# ╔═╡ 658524ef-c716-408c-ab57-f1a10459ff24
md"
## Compensation and navigation animation

Make an animation of the navigation performance, keeping track of each component of the compensation terms emitted by model 3, projected into a 2D plane (north/east, removing the vertical components) to make visualization simpler. Ideally, you should see that the knowledge-integrated architecture in model 3 is mostly accounting for variations in the compensation field arising from the map/plane maneuvers in the Tolles-Lawson (TL) component, whereas the neural network (NN) correction is addressing any deviations due to current/voltage fluctuations.

Increase (or decrease) the `skip_every` argument to exclude (or include) more frames, which speeds up (or slows down) the rendering time & decrease (or increase) the file size.
"

# ╔═╡ c469bc3c-434d-452a-8dca-1a96823a8153
g1 = gif_animation_m3(TL_perm, TL_induced, TL_eddy, TL_aircraft, B_unit,
					  y_nn, y, y_hat, xyz_test, filt_out.lat, filt_out.lon;
					  ind=ind_test, skip_every=20, tt_lim=(3.0,9.5))

# ╔═╡ 0cda1d66-6ba5-40d2-89ff-caf574d9a09f
md"Show detailed filter performance using `plot_filt_err`
"

# ╔═╡ 3e9e848e-bf16-428a-98c3-9fee4eddda41
(p4,p5) = plot_filt_err(traj, filt_out, crlb_out);

# ╔═╡ a42f021e-0a11-4589-949a-7a2a56af129a
p4

# ╔═╡ ea334b91-e53d-404f-ae3c-106ba1bb7463
p5

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
MagNav = "~1.2.2"
Plots = "~1.40.9"
Random = "~1.11.0"
Statistics = "~1.11.1"
"""

# ╔═╡ Cell order:
# ╟─e289486a-57ed-4eeb-9ec9-6500f0bc563b
# ╟─b1d3b1b3-db8d-4bb0-a884-d57f217fef24
# ╠═22982d8e-240c-11ee-2e0c-bda2a93a4ab0
# ╟─3a55962c-bd1b-410c-b98a-3130fc11ee11
# ╠═bf9f72f0-c351-48d3-a811-418ee965073c
# ╠═9d11d4be-9ba7-44ab-a4ff-f0fdf95b2171
# ╠═663f7b8c-dfc6-445d-9c54-1141856f2a66
# ╠═99405ae1-580a-4fd3-a860-13cc5b22b045
# ╟─3d438be1-537b-4c13-a422-7f5f48479e62
# ╠═6d4a87c8-41d7-4478-b52a-4dc5a1ae18ea
# ╠═c5b6b439-e962-4c9d-9b02-339657fb266b
# ╠═ae1acc31-19db-4e94-85dc-e6274186978e
# ╟─7a20ef62-f352-4fc7-9061-13fb293bb0bf
# ╠═39bdbe6a-52ae-44d2-8e80-2e7d2a75e322
# ╟─7a59dc41-21f1-4a5e-9f8b-06cacca8845b
# ╠═9ed61357-346b-49de-a1ed-8849db041ade
# ╟─a9cac83d-03b3-4793-8da4-f45563091bd0
# ╠═f492ee3d-c097-4937-9552-e7d2632a5e50
# ╠═7e15f297-64a4-4ee6-a0ab-65d959354f29
# ╟─239b60aa-0e97-4871-a9a5-64cd802f4bde
# ╠═21828f95-98f3-4271-9a57-121252065741
# ╠═4d75b716-bc93-42f5-8b1a-f5550e7de276
# ╠═4171fef9-650b-4f37-ae5d-7084d54a0be0
# ╟─0c839441-909d-4095-a4eb-4dfc92eb2121
# ╠═0a1b7102-59e0-4e2a-a45c-5c21d0039b88
# ╟─5f88bca1-ecc9-45b2-8164-0f6965b81088
# ╠═ed1f8baa-d228-4f66-a979-527f68acccdc
# ╟─4e01e107-5d58-409d-b08c-23135b4c28ba
# ╠═566bb8eb-23d3-473d-8e79-1d02536dfdd5
# ╠═987292fe-4e58-4cc9-bfe2-734c0f4174f8
# ╟─d0fcbd95-002b-4e67-ad06-15ae52f27ede
# ╠═70b8e7ff-71ee-489b-817e-1d4ea3004355
# ╠═a56f0b50-0b39-4400-928d-f485af206d23
# ╟─658524ef-c716-408c-ab57-f1a10459ff24
# ╠═c469bc3c-434d-452a-8dca-1a96823a8153
# ╟─0cda1d66-6ba5-40d2-89ff-caf574d9a09f
# ╠═3e9e848e-bf16-428a-98c3-9fee4eddda41
# ╠═a42f021e-0a11-4589-949a-7a2a56af129a
# ╠═ea334b91-e53d-404f-ae3c-106ba1bb7463
# ╟─00000000-0000-0000-0000-000000000001
