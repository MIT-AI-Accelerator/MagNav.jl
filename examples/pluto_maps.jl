### A Pluto.jl notebook ###
# v0.20.13

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
md"# Magnetic Anomaly Maps
This file is best viewed in a [Pluto](https://plutojl.org/) notebook. To do so, from the MagNav.jl directory, run:
```julia
julia> using Pluto
julia> Pluto.run() # select & open notebook
```

This is a reactive notebook, so feel free to change any parameters.
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
md"## Load Perth map

This is the Perth map (at 800 m) as provided by Sander Geophysics Ltd.
"

# ╔═╡ bf9f72f0-c351-48d3-a811-418ee965073c
begin
	map_gxf = MagNav.ottawa_area_maps_gxf(:Perth)
	p_mapS_800 = map_gxf2h5(map_gxf,800)
	p1 = plot_map(p_mapS_800)
end

# ╔═╡ 438f2f01-5cbe-4088-b365-571de4f9539a
md"## Display Perth map in Google Earth

Display the Perth map in Google Earth by uncommenting below to generate a KMZ file, then open in Google Earth.
"

# ╔═╡ d7d5fa3e-0c00-4d0b-b9e0-b6d8b0e917cf
# map2kmz(p_mapS_800,"Perth")

# ╔═╡ 32d385fd-1a78-47f8-b748-41e77f680da0
md"## Overlay Perth mini-survey

Overlaid on the Perth map are most of the flight lines used to generate the map, which were collected during Flight 1004 (see [readme](https://github.com/MIT-AI-Accelerator/MagNav.jl/blob/master/readmes/Flt1004_readme.txt)) & Flight 1005 (see [readme](https://github.com/MIT-AI-Accelerator/MagNav.jl/blob/master/readmes/Flt1005_readme.txt)). The full list of SGL flights is in `df_flight`.
"

# ╔═╡ 9d11d4be-9ba7-44ab-a4ff-f0fdf95b2171
begin
	xyz_1004 = get_XYZ(:Flt1004,df_flight;silent=true) # load flight data
	xyz_1005 = get_XYZ(:Flt1005,df_flight;silent=true) # load flight data
	lines = [4019,4018,4017,4016,4015,4012,4011,4010,4009,4008,4007,4004,
	         4003,4002,4001,421,419,417,415,413,411,409,408,407,405,403,401]
	for line in lines
	    xyz = line in xyz_1004.line ? xyz_1004 : xyz_1005
	    ind = get_ind(xyz,line,df_nav) # get Boolean indices
	    plot_path!(p1,xyz.traj,ind;show_plot=false,path_color=:black)
	end
	p1
end

# ╔═╡ b21e0015-367e-44e2-89b9-841867b6299d
md"## Load previously processed maps

Eastern Ontario & NAMAD maps have been processed & saved previously. The full list of maps is in `df_map`.
"

# ╔═╡ 663f7b8c-dfc6-445d-9c54-1141856f2a66
begin # e_mask contains Boolean indices for "real" map data (not filled-in)
	e_mapS_395 = get_map(:Eastern_395,df_map) # load map data
	e_mask     = e_mapS_395.mask
end;

# ╔═╡ 99405ae1-580a-4fd3-a860-13cc5b22b045
begin
	xx_lim = extrema(e_mapS_395.xx) .+ (-0.01,0.01)
	yy_lim = extrema(e_mapS_395.yy) .+ (-0.01,0.01)
	n_mapS_395 = upward_fft(map_trim(get_map(MagNav.namad),
				            xx_lim=xx_lim,yy_lim=yy_lim),e_mapS_395.alt)
end;

# ╔═╡ c111d83f-2883-4149-8ee2-53bf4b57640a
md"## Plot all Ottawa area maps

3 maps are overlayed with 3 different color schemes.
"

# ╔═╡ 6d4a87c8-41d7-4478-b52a-4dc5a1ae18ea
begin
	clims = (-500,500)
	dpi   = 50
	p2 = plot_map(n_mapS_395;clims=clims,dpi=dpi,legend=false)    # 395 m
	plot_map!(p2,e_mapS_395;clims=clims,dpi=dpi,map_color=:magma) # 395 m
	plot_map!(p2,p_mapS_800;clims=clims,dpi=dpi,map_color=:gray)  # 800 m
	p2
end

# ╔═╡ 39bdbe6a-52ae-44d2-8e80-2e7d2a75e322
begin
    using MagNav: get_step
	(map_map,px,py) = map_expand(e_mapS_395.map,200)
	px_end    = size(map_map,2) - length(e_mapS_395.xx) - px
	py_end    = size(map_map,1) - length(e_mapS_395.yy) - py
	dx        = get_step(e_mapS_395.xx)
	dy        = get_step(e_mapS_395.yy)
	xx        = [(e_mapS_395.xx[1]-dx*px):dx:(e_mapS_395.xx[end]+dx*px_end);]
	yy        = [(e_mapS_395.yy[1]-dy*py):dy:(e_mapS_395.yy[end]+dy*py_end);]
	(lat,lon) = map_border(e_mapS_395;sort_border=true) # get map border
	p5 = plot_map(map_map,xx,yy;dpi=dpi)
	plot_path!(p5,lat,lon;path_color=:black)
	p5
end

# ╔═╡ 2d18bbeb-ad4e-4c5b-80a3-25bd99f45c73
md"## Plot Eastern Ontario altitude map CDF

Most of the map data was collected at altitudes between 215 & 395 m.
"

# ╔═╡ c5b6b439-e962-4c9d-9b02-339657fb266b
begin # this map contains an additional drape (altitude) map
	e_mapS_drp = get_map(:Eastern_drape,df_map) # load map data
	e_alts  = e_mapS_drp.alt[e_mask]
	alt_avg = round(Int,mean(e_alts))
	alt_med = round(Int,median(e_alts))
	alt_val = 200:500
	alt_cdf = [sum(e_alts .< a) for a in alt_val] / sum(e_mask)
	p3 = plot(alt_val,alt_cdf,xlab="altitude [m]",ylab="fraction [-]",
	          title="altitude map CDF",lab=false,dpi=200)
end

# ╔═╡ 3f6b725b-f1c7-41fb-9854-c737b9698a40
md"## Plot Eastern Ontario altitude map

Minimal areas have map data collected at 395 m or higher (colored spots), so a level map can be generated at 395 m using almost entirely upward continuation & minimal downward continuation.
"

# ╔═╡ ae1acc31-19db-4e94-85dc-e6274186978e
begin
	p4 = plot_map(e_mapS_395;dpi=dpi,legend=false,map_color=:gray)
	e_mapS_395_ = deepcopy(e_mapS_395)
	e_mapS_395_.map[e_mapS_drp.alt .<= 395] .= 0
	plot_map!(p4,e_mapS_395_;dpi=dpi)
	p4
end

# ╔═╡ 504842a2-61e0-4154-a1e9-5182c97b6090
md"## Plot expanded Eastern Ontario map

The original map area is show with a black outline. During upward (or downward) continuation, the map is temporarily expanded with \"wrapped\" edges for a more accurate result.
"

# ╔═╡ 8b5c030a-e0b9-4a7a-a901-a6967edbe70b
md"## Plot combined Eastern Ontario and NAMAD maps together

2 maps can be combined at the same altitude. Here, the Eastern Ontario map is better (higher resolution), but the NAMAD map can be used in the case of navigation near the Eastern Ontario map border. Note that only the southeast part of the NAMAD map is accurate here, as the map coverage ends & the remainder is filled-in (e.g., the northwest corner).
"

# ╔═╡ 9ed61357-346b-49de-a1ed-8849db041ade
begin
	mapS_combined = map_combine(e_mapS_395,n_mapS_395)
	p6 = plot_map(mapS_combined;dpi=dpi,use_mask=false)
	plot_path!(p6,lat,lon;path_color=:black)
	p6
end

# ╔═╡ 1b4d1092-1680-4676-a69d-6acc2549d588
md"## Compare map values on border of Eastern Ontario and NAMAD maps

Map values on 2 maps will be compared for the red path along the border.
"

# ╔═╡ 1927f750-c3de-4c52-8a78-bb7f0f05c21e
begin
	ind_trim   = (rad2deg.(lon) .> -76) .& (rad2deg.(lat) .< 45)
	lon_trim   = lon[ind_trim]
	lat_trim   = lat[ind_trim]
	p7 = plot_map(mapS_combined;dpi=dpi,use_mask=false,map_color=:gray)
	plot_path!(p7,lat,lon;path_color=:black)
	plot_path!(p7,lat_trim,lon_trim;path_color=:red)
	p7
end

# ╔═╡ 8dd4196a-692f-4f4f-8f4a-6d8c39c4e3ac
md" First the `map_interpolate` function is used to create map interpolation objects, which are then evaluated at the given latitudes & longitudes (red path above). The (standard deviation) error of the map values is approximately 100 nT in this case.
"

# ╔═╡ b332569d-28d6-48e3-b080-170e23c1aa3f
begin
	e_itp_mapS = map_interpolate(e_mapS_395)
	n_itp_mapS = map_interpolate(n_mapS_395)
	e_mapS_val = e_itp_mapS.(lat_trim,lon_trim)
	n_mapS_val = n_itp_mapS.(lat_trim,lon_trim)
	mapS_err   = round(Int,std(n_mapS_val - e_mapS_val)) # error
end

# ╔═╡ 7cf91a2c-73a3-4ad5-93f2-6f6908702441
md" The trend of the map values agree, as expected.
"

# ╔═╡ 992dce6d-3bed-428c-84ca-2ee9bc9f0167
begin
    p8 = plot(ylab="map value [nT]",dpi=200)
	plot!(p8,e_mapS_val,lab="Eastern")
	plot!(p8,n_mapS_val,lab="NAMAD")
end

# ╔═╡ 18eecde1-3cc3-4775-a139-c77beffe394e
md"## Create a 3D map

Most functionality in MagNav.jl can use 3D maps, which contain multiple stacked 2D maps with constant altitude spacing between each map level. There are 2 ways to create a 3D map, the first of which is to upward/downward continue to multiple altitudes using the `upward_fft` function. In this case, a filled Perth map at 800 m is stacked with the same map upward continued to 810 m.
"

# ╔═╡ e28ee51a-7d9f-4ce8-a63e-69f2ce654901
begin
	p_mapS3D_1 = upward_fft(p_mapS_800, p_mapS_800.alt .+ [0,10]) # 3D map
	println(p_mapS3D_1.alt) # map altitude levels
	p9 = plot_map(p_mapS3D_1) # plot single map level (default is lowest)
end

# ╔═╡ ee8bbf97-97b0-420d-a29d-7a0a9fb87a41
md"The second way to create a 3D map is to combine multiple 2D maps using the `map_combine` function. In this case, the Eastern Ontario map (395 m) & Perth map (800 m) are stacked into a 3D map.
"

# ╔═╡ eaa6b007-97cf-4178-bb9a-2656632df7c1
begin
	p_mapS3D_2 = map_combine([e_mapS_395,p_mapS_800]; # 3D map
	                         dx       = get_step(p_mapS_800.xx),
	                         dy       = get_step(p_mapS_800.yy),
	                         xx_lim   = extrema(p_mapS_800.xx),
	                         yy_lim   = extrema(p_mapS_800.yy));
	println(p_mapS3D_2.alt) # map altitude levels (default is 3)
	p10 = plot_map(p_mapS3D_2) # plot single map level (default is lowest)
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
MagNav = "~1.3.2"
Plots = "~1.40.14"
Random = "~1.11.0"
Statistics = "~1.11.1"
"""

# ╔═╡ Cell order:
# ╟─e289486a-57ed-4eeb-9ec9-6500f0bc563b
# ╟─b1d3b1b3-db8d-4bb0-a884-d57f217fef24
# ╠═22982d8e-240c-11ee-2e0c-bda2a93a4ab0
# ╟─3a55962c-bd1b-410c-b98a-3130fc11ee11
# ╠═bf9f72f0-c351-48d3-a811-418ee965073c
# ╟─438f2f01-5cbe-4088-b365-571de4f9539a
# ╠═d7d5fa3e-0c00-4d0b-b9e0-b6d8b0e917cf
# ╟─32d385fd-1a78-47f8-b748-41e77f680da0
# ╠═9d11d4be-9ba7-44ab-a4ff-f0fdf95b2171
# ╟─b21e0015-367e-44e2-89b9-841867b6299d
# ╠═663f7b8c-dfc6-445d-9c54-1141856f2a66
# ╠═99405ae1-580a-4fd3-a860-13cc5b22b045
# ╟─c111d83f-2883-4149-8ee2-53bf4b57640a
# ╠═6d4a87c8-41d7-4478-b52a-4dc5a1ae18ea
# ╟─2d18bbeb-ad4e-4c5b-80a3-25bd99f45c73
# ╠═c5b6b439-e962-4c9d-9b02-339657fb266b
# ╟─3f6b725b-f1c7-41fb-9854-c737b9698a40
# ╠═ae1acc31-19db-4e94-85dc-e6274186978e
# ╟─504842a2-61e0-4154-a1e9-5182c97b6090
# ╠═39bdbe6a-52ae-44d2-8e80-2e7d2a75e322
# ╟─8b5c030a-e0b9-4a7a-a901-a6967edbe70b
# ╠═9ed61357-346b-49de-a1ed-8849db041ade
# ╟─1b4d1092-1680-4676-a69d-6acc2549d588
# ╠═1927f750-c3de-4c52-8a78-bb7f0f05c21e
# ╟─8dd4196a-692f-4f4f-8f4a-6d8c39c4e3ac
# ╠═b332569d-28d6-48e3-b080-170e23c1aa3f
# ╟─7cf91a2c-73a3-4ad5-93f2-6f6908702441
# ╠═992dce6d-3bed-428c-84ca-2ee9bc9f0167
# ╟─18eecde1-3cc3-4775-a139-c77beffe394e
# ╠═e28ee51a-7d9f-4ce8-a63e-69f2ce654901
# ╟─ee8bbf97-97b0-420d-a29d-7a0a9fb87a41
# ╠═eaa6b007-97cf-4178-bb9a-2656632df7c1
# ╟─00000000-0000-0000-0000-000000000001
