### A Pluto.jl notebook ###
# v0.20.8

using Markdown
using InteractiveUtils

# ╔═╡ d2d82208-6c6b-4f1d-9798-6bbb884185c8
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

# ╔═╡ f0f58d36-a2a9-4cd6-a549-352a726c76d0
begin
	using MagNav: elasticnet_fit, plsr_fit, linear_test
	using LinearAlgebra
	using Statistics: cov
end;

# ╔═╡ e336e668-4d9f-11ee-2517-b3a384f42e23
md"# Linear Models
This file is best viewed in a [Pluto](https://plutojl.org/) notebook. To run it this way, from the MagNav.jl directory, do:
```julia
julia> using Pluto
julia> Pluto.run() # select & open notebook
```

This is a reactive notebook, so feel free to change any parameters of interest, like adding/removing features to the model, switching between linear models (`:TL`, `:TL_mod`, `:elasticnet`, `:plsr`), or different training & testing lines.
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

# ╔═╡ fa4f5698-8524-47c4-9a79-c5ee8462d517
md"Select magnetometers & parameters for compensation.
"

# ╔═╡ 512d9c80-ce7a-42ad-aff2-98575ccd94d8
begin # try modifying these parameters
	features = [:mag_4_uc,:mag_4_uc_dot,:mag_4_uc_dot4,:TL_A_flux_d]
	use_mag  = :mag_4_uc
	use_vec  = :flux_d
	terms    = [:permanent,:induced,:fdm]
	terms_A  = [:permanent,:induced,:eddy,:bias]
	k_plsr   = 10
end;

# ╔═╡ d92d1039-5f01-4f3b-b53a-dd579e78f136
comp_params_1_init = LinCompParams(features_setup = features,
                                   model_type     = :plsr,
                                   y_type         = :d,
                                   use_mag        = use_mag,
                                   terms          = terms,
                                   k_plsr         = k_plsr);

# ╔═╡ ceea652f-ce1b-4ae9-9b90-0fb538d11b9f
md"Select training & testing flights from Flight 1006 (see [readme](https://github.com/MIT-AI-Accelerator/MagNav.jl/blob/master/readmes/Flt1006_readme.txt)).
"

# ╔═╡ f6822aeb-ff22-4662-8210-f72db9647c65
begin
	lines_train = [1006.03, 1006.04, 1006.05, 1006.06]
	lines_test  = [1006.08]
end;

# ╔═╡ dc519bdb-c501-443a-b7e5-26e6071650fe
md"## PLSR-based calibration and compensation

Perform PLSR-based calibration using training data. The full list of SGL flights is in `df_flight`, the full list of maps is in `df_map`, & the full list of flight lines is in `df_all`.
"

# ╔═╡ dc05737c-4ce8-49aa-809d-e58046fe546c
(comp_params_1,_,_,_,feats) =
    comp_train(comp_params_1_init,lines_train,df_all,df_flight,df_map);

# ╔═╡ a82ddaa5-265b-411a-bff2-17587388e848
md"Perform PLSR-based compensation on testing data. The full list of navigation-capable flight lines is in `df_nav`.
"

# ╔═╡ 8759f021-c416-47b9-8ecf-196197d34919
comp_test(comp_params_1,lines_test,df_nav,df_flight,df_map);

# ╔═╡ 800fbc33-2c64-496a-8a6e-f2c9bf463692
md"Setup data for Tolles-Lawson. `TL_ind` holds the Boolean indices (mask) just for the selected flight data. The full list of calibration flight line options is in `df_cal`.
"

# ╔═╡ 8eebdfaa-22d2-4466-8092-d62a6392c49b
begin
	TL_i   = 6 # select first calibration box of 1006.04
	TL_xyz = get_XYZ(df_cal.flight[TL_i],df_flight;silent=true) # load flight data
	TL_ind = get_ind(TL_xyz;tt_lim=[df_cal.t_start[TL_i],df_cal.t_end[TL_i]])
end;

# ╔═╡ d6efdf23-1a67-4e3e-b8d8-162b706032f6
comp_params_2_init = LinCompParams(features_setup = features,
                                   model_type     = :TL,
                                   y_type         = :d,
                                   use_mag        = use_mag,
                                   use_vec        = use_vec,
                                   terms_A        = terms_A);

# ╔═╡ 8d1ae89d-65ba-4f4b-9fb9-4a717cc8b8f7
md"Perform Tolles-Lawson calibration using training data.
"

# ╔═╡ 5ac7765d-eca9-42bd-b298-28d2ef67ab75
(comp_params_2,_,_,_,_) =
	comp_train(comp_params_2_init,TL_xyz,TL_ind);

# ╔═╡ b204f694-8101-4d14-a497-b5a1a8f8381f
md"Perform Tolles-Lawson compensation on testing data.
"

# ╔═╡ edcd19a1-0479-423c-b4e0-1434de5fb04c
comp_test(comp_params_2,lines_test,df_all,df_flight,df_map);

# ╔═╡ b8ae2d27-ae5c-421b-9b2e-c378bdccf542
md"Get training & testing data & normalize by feature (columns). Typically this is done internally, but shown here to better explain PLSR.
"

# ╔═╡ ed991fd4-05d9-4bee-b58b-547d63e5d199
begin
	(_,x_train,y_train,_,_,l_segs_train) =
	    MagNav.get_Axy(lines_train,df_all,df_flight,df_map,features;
	                   use_mag=use_mag,use_vec=use_vec,terms=terms)
	(_,x_test,y_test,_,_,l_segs_test) =
	    MagNav.get_Axy(lines_test,df_nav,df_flight,df_map,features;
	                   use_mag=use_mag,use_vec=use_vec,terms=terms)
	(x_bias,x_scale,x_train_norm,x_test_norm) = norm_sets(x_train,x_test)
	(y_bias,y_scale,y_train_norm,y_test_norm) = norm_sets(y_train,y_test)
	x   = x_train_norm # for conciseness
	y   = y_train_norm # for conciseness
	x_t = x_test_norm  # for conciseness
	y_t = y_test_norm  # for conciseness
end;

# ╔═╡ 6ab67536-d481-4027-b835-105f14480f1e
md"Training data matrix decomposition & reconstruction with SVD.
"

# ╔═╡ d600ded5-d6dc-43f2-8c00-7b1fb45c6f0f
begin
	(U,S,V) = svd(x)
	Vt      = V'
	x_err   = [std(U[:,1:i]*Diagonal(S[1:i])*Vt[1:i,:]-x) for i in eachindex(S)]
	x_var   = [sum(S[1:i])/sum(S) for i in eachindex(S)]
end;

# ╔═╡ 94faaf70-6eb9-485b-b05e-8685265a612f
md"Plot PLSR error & variance retained.
"

# ╔═╡ 8ecaeb47-af94-47f9-bfed-2c3b5789375e
begin
	p1 = plot(xlab="k (number of compenents)",ylim=(0,1.01),dpi=200)
	plot!(p1,eachindex(S),x_err,lab="error",lc=:blue)
	plot!(p1,eachindex(S),x_var,lab="variance retained" ,lc=:red)
end

# ╔═╡ d46c62a2-9368-4ec3-b305-ab4e56b27f6b
md"Determine training & testing error with different numbers of components.
"

# ╔═╡ e9d26d61-a7dd-4531-9238-0bdfde6d2c05
begin
	k_max     = size(x,2) # maximum number of compenents (features)
	err_train = zeros(k_max)
	err_test  = zeros(k_max)
	coef_set  = plsr_fit(x,y,k_max;return_set=true)
	for k = 1:k_max
	    y_train_hat_norm = vec(x  *coef_set[:,:,k])
	    y_test_hat_norm  = vec(x_t*coef_set[:,:,k])
	    (y_train_hat,y_test_hat) = denorm_sets(y_bias,y_scale,
	                                           y_train_hat_norm,
	                                           y_test_hat_norm)
	    err_train[k] = std(err_segs(y_train_hat,y_train,l_segs_train))
	    err_test[k]  = std(err_segs(y_test_hat ,y_test ,l_segs_test ))
	end
end;

# ╔═╡ 8874fd97-aad0-454b-b425-d66208752449
md"Plot PLSR-based training & testing error with different numbers of components.
"

# ╔═╡ 15839dc6-cbe0-4fb9-b60a-1eed9dd4941c
begin
	p2 = plot(xlab="k (number of compenents)",ylab="PLSR error [nT]",ylim=(0,150))
	plot!(p2,1:k_max,err_train,lab="train")
	plot!(p2,1:k_max,err_test ,lab="test")
end

# ╔═╡ c5e34922-977c-4cad-9e9a-94bb242daa7c
md"## Elastic net-based calibration and compensation
"

# ╔═╡ 3e2d9612-203b-4403-b51e-59480eaf83de
begin
	α = 0.99 # tradeoff between ridge regression (0) & Lasso (1)
	(model,data_norms,y_train_hat,_) =
	    elasticnet_fit(x_train,y_train,α;l_segs=l_segs_train)
	(y_test_hat,_) = linear_test(x_test,y_test,data_norms,model;l_segs=l_segs_test)
	(coef,bias) = model
end;

# ╔═╡ 897eb9a9-e84b-412a-852a-70981b01ddc7
md"Features with largest elastic net coefficients.
"

# ╔═╡ 774ab47b-1fee-4911-b1d9-9725aee11848
feats[sortperm(abs.(coef),rev=true)]

# ╔═╡ 70d67179-a688-4401-8248-ad4a3256f5d4
md"## Principal component analysis (PCA)
Look at how `k_pca` is used in neural network-based calibration & compensation.

Note that reduced-dimension data uses: x \* V = U \* Diagonal(S)
"

# ╔═╡ ee7a6b3e-cd13-44ba-87fa-ccd461c12921
begin
	k_pca = 3 # select k (order reduction factor)
	println("std dev error with k = $k_pca: ",round(x_err[k_pca],digits=2))
	println("var  retained with k = $k_pca: ",round(x_var[k_pca],digits=2))
	(_,S_pca,V_pca) = svd(cov(x))
	x_new   = x*V_pca[:,1:k_pca]
	x_t_new = x_t*V_pca[:,1:k_pca]
	v_scale = V_pca[:,1:k_pca]*inv(Diagonal(sqrt.(S_pca[1:k_pca])))
	x_use   = x*v_scale   #* this could be trained on
	x_t_use = x_t*v_scale #* this could be tested  on
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
MagNav = "~1.3.1"
Plots = "~1.40.13"
Random = "~1.11.0"
Statistics = "~1.11.1"
"""

# ╔═╡ Cell order:
# ╟─e336e668-4d9f-11ee-2517-b3a384f42e23
# ╟─b1d3b1b3-db8d-4bb0-a884-d57f217fef24
# ╠═d2d82208-6c6b-4f1d-9798-6bbb884185c8
# ╠═f0f58d36-a2a9-4cd6-a549-352a726c76d0
# ╟─fa4f5698-8524-47c4-9a79-c5ee8462d517
# ╠═512d9c80-ce7a-42ad-aff2-98575ccd94d8
# ╠═d92d1039-5f01-4f3b-b53a-dd579e78f136
# ╟─ceea652f-ce1b-4ae9-9b90-0fb538d11b9f
# ╠═f6822aeb-ff22-4662-8210-f72db9647c65
# ╟─dc519bdb-c501-443a-b7e5-26e6071650fe
# ╠═dc05737c-4ce8-49aa-809d-e58046fe546c
# ╟─a82ddaa5-265b-411a-bff2-17587388e848
# ╠═8759f021-c416-47b9-8ecf-196197d34919
# ╟─800fbc33-2c64-496a-8a6e-f2c9bf463692
# ╠═8eebdfaa-22d2-4466-8092-d62a6392c49b
# ╠═d6efdf23-1a67-4e3e-b8d8-162b706032f6
# ╟─8d1ae89d-65ba-4f4b-9fb9-4a717cc8b8f7
# ╠═5ac7765d-eca9-42bd-b298-28d2ef67ab75
# ╟─b204f694-8101-4d14-a497-b5a1a8f8381f
# ╠═edcd19a1-0479-423c-b4e0-1434de5fb04c
# ╟─b8ae2d27-ae5c-421b-9b2e-c378bdccf542
# ╠═ed991fd4-05d9-4bee-b58b-547d63e5d199
# ╟─6ab67536-d481-4027-b835-105f14480f1e
# ╠═d600ded5-d6dc-43f2-8c00-7b1fb45c6f0f
# ╟─94faaf70-6eb9-485b-b05e-8685265a612f
# ╠═8ecaeb47-af94-47f9-bfed-2c3b5789375e
# ╟─d46c62a2-9368-4ec3-b305-ab4e56b27f6b
# ╠═e9d26d61-a7dd-4531-9238-0bdfde6d2c05
# ╟─8874fd97-aad0-454b-b425-d66208752449
# ╠═15839dc6-cbe0-4fb9-b60a-1eed9dd4941c
# ╟─c5e34922-977c-4cad-9e9a-94bb242daa7c
# ╠═3e2d9612-203b-4403-b51e-59480eaf83de
# ╟─897eb9a9-e84b-412a-852a-70981b01ddc7
# ╠═774ab47b-1fee-4911-b1d9-9725aee11848
# ╟─70d67179-a688-4401-8248-ad4a3256f5d4
# ╠═ee7a6b3e-cd13-44ba-87fa-ccd461c12921
# ╟─00000000-0000-0000-0000-000000000001
