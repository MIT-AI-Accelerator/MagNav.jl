### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ f799c623-3802-491b-9709-6c1a01ac3978
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

# ╔═╡ d9ac0df2-3d79-11ee-0869-73b7f6649d95
md"# Using the MagNav Package with Simulated Data
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

# ╔═╡ 3ffae734-b7b2-45bf-9843-171b6e2deb13
md"## Load map data and create flight data

The built-in [NAMAD](https://mrdata.usgs.gov/magnetic/map-us.html) map is used to create flight data.
"

# ╔═╡ ff29f1c4-74ef-43cb-95be-14b5518e2cc6
begin
	seed!(33)  # ensure create_XYZ0() reproducibility
	t    = 600 # flight time [s]
	mapS = get_map(MagNav.namad) # load map data
	xyz  = create_XYZ0(mapS;alt=mapS.alt,t=t) # create flight data
	traj = xyz.traj # trajectory (GPS) struct
	ins  = xyz.ins  # INS struct
	mapS = map_trim(mapS,traj;pad=10) # trim map for given trajectory (with padding)
	itp_mapS = map_interpolate(mapS)  # map interpolation function
end;

# ╔═╡ a2dafd09-e9f7-4a58-b269-32fee40d602d
md"The `xyz` flight data struct is of type `MagNav.XYZ0` (for the minimum dataset required for MagNav), which is a subtype of `MagNav.XYZ` (the abstract type for any flight data in MagNav.jl). There are 7 fields, which can be accessed using dot notation. These fields are described in the docs, which are easily accessed by searching `MagNav.XYZ0` in the Live Docs in the lower right within Pluto.
"

# ╔═╡ 57aeef2a-2b7e-44f7-b1b7-c40736c38063
typeof(xyz)

# ╔═╡ a750b1aa-04d9-4f7a-a380-219ac69e8a0d
fieldnames(MagNav.XYZ0)

# ╔═╡ 773df1f5-4cfc-42a8-b6c4-6d9d6e40bb0c
md"## Navigation

Create a navigation filter model.
"

# ╔═╡ 49eafe6d-120d-4e9e-8a34-2a5034f7de3f
(P0,Qd,R) = create_model(traj.dt,traj.lat[1]);

# ╔═╡ 9a092951-05fd-47c2-932d-002231135486
md"Run the navigation filter (EKF), determine the Cramér–Rao lower bound (CRLB), & extract output data.
"

# ╔═╡ 81a1f9fd-245d-42ca-bce0-fe78a69009ac
begin
	mag_use = xyz.mag_1_c # selected magnetometer (using compensated mag)
	(crlb_out,ins_out,filt_out) = run_filt(traj,ins,mag_use,itp_mapS,:ekf;P0,Qd,R)
end;

# ╔═╡ 4b40ae5b-6641-4792-89bb-dcceb553dd41
md"Plotting setup.
"

# ╔═╡ 8f417413-6739-4c25-848f-7d47a491b89a
begin
	t0 = traj.tt[1]/60    # [min]
	tt = traj.tt/60 .- t0 # [min]
end;

# ╔═╡ dc5aa915-9037-4792-a3e1-09074431d786
md"Position (lat & lot) for trajectory (GPS), INS (after zeroing), & navigation filter.
"

# ╔═╡ 363b668b-bece-4bcd-8ac4-287b3138fdee
begin
	p1 = plot_map(mapS;map_color=:gray); # map background
	plot_filt!(p1,traj,ins,filt_out;show_plot=false) # overlay GPS, INS, & filter
	plot!(p1,legend=:topleft) # move as needed
end

# ╔═╡ 9f90be51-1d9b-45a0-8ef3-91c93aa9bf2b
md"Northing & easting INS error (after zeroing).
"

# ╔═╡ 3699e96c-48b4-4116-8c33-13cbc64bb3df
begin
	p2 = plot(xlab="time [min]",ylab="error [m]",legend=:topright,dpi=200)
	plot!(p2,tt,ins_out.n_err,lab="northing")
	plot!(p2,tt,ins_out.e_err,lab="easting")
end

# ╔═╡ 7fecb8d4-5b8d-4731-b224-22b9adfad5ee
(p3,p4) = plot_filt_err(traj,filt_out,crlb_out;show_plot=false);

# ╔═╡ 9e6376e6-3280-4f10-8a52-870c8c43f1b2
md"Northing navigation filter residuals.
"

# ╔═╡ 4027a986-1cdf-467b-802d-ca4e963c77f5
p3

# ╔═╡ 893d6521-87e0-422c-9aee-480fb2dcbfba
md"Easting navigation filter residuals.
"

# ╔═╡ c8c89445-acc7-4dfe-8b35-181abbb89f1b
p4

# ╔═╡ ac029f6d-a7b4-40d6-a47c-041f3be99850
md"Display the map or flight paths in Google Earth by uncommenting below to generate a KMZ file (`mapS`) or KML files (`traj`, `ins`, `filt_out`), then open in Google Earth.
"

# ╔═╡ 26810eff-0812-43cc-b1bc-d4f5d7c9542d
begin
	# map2kmz(mapS,"pluto_sim_map")
	# path2kml(traj,"pluto_sim_gps")
	# path2kml(ins,"pluto_sim_ins")
	# path2kml(filt_out,"pluto_sim_filt")
end;

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
MagNav = "~1.2.1"
Plots = "~1.40.9"
Random = "~1.11.0"
Statistics = "~1.11.1"
"""

# ╔═╡ Cell order:
# ╟─d9ac0df2-3d79-11ee-0869-73b7f6649d95
# ╟─b1d3b1b3-db8d-4bb0-a884-d57f217fef24
# ╠═f799c623-3802-491b-9709-6c1a01ac3978
# ╟─3ffae734-b7b2-45bf-9843-171b6e2deb13
# ╠═ff29f1c4-74ef-43cb-95be-14b5518e2cc6
# ╟─a2dafd09-e9f7-4a58-b269-32fee40d602d
# ╠═57aeef2a-2b7e-44f7-b1b7-c40736c38063
# ╠═a750b1aa-04d9-4f7a-a380-219ac69e8a0d
# ╟─773df1f5-4cfc-42a8-b6c4-6d9d6e40bb0c
# ╠═49eafe6d-120d-4e9e-8a34-2a5034f7de3f
# ╟─9a092951-05fd-47c2-932d-002231135486
# ╠═81a1f9fd-245d-42ca-bce0-fe78a69009ac
# ╟─4b40ae5b-6641-4792-89bb-dcceb553dd41
# ╠═8f417413-6739-4c25-848f-7d47a491b89a
# ╟─dc5aa915-9037-4792-a3e1-09074431d786
# ╠═363b668b-bece-4bcd-8ac4-287b3138fdee
# ╟─9f90be51-1d9b-45a0-8ef3-91c93aa9bf2b
# ╠═3699e96c-48b4-4116-8c33-13cbc64bb3df
# ╠═7fecb8d4-5b8d-4731-b224-22b9adfad5ee
# ╟─9e6376e6-3280-4f10-8a52-870c8c43f1b2
# ╠═4027a986-1cdf-467b-802d-ca4e963c77f5
# ╟─893d6521-87e0-422c-9aee-480fb2dcbfba
# ╠═c8c89445-acc7-4dfe-8b35-181abbb89f1b
# ╟─ac029f6d-a7b4-40d6-a47c-041f3be99850
# ╠═26810eff-0812-43cc-b1bc-d4f5d7c9542d
# ╟─00000000-0000-0000-0000-000000000001
