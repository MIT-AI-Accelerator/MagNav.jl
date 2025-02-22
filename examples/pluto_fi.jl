### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 5e16a321-3321-4d54-899e-594eb670e681
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

# ╔═╡ 66e96968-4daa-11ee-31cd-7b4d78033095
md"# Feature Importance/Selection
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

# ╔═╡ 4e5e2a24-cc3f-4766-a6f5-c94ff1f6ac2f
md"Select magnetometers & parameters for compensation.
"

# ╔═╡ 1c4d1729-19eb-4b2e-8775-a7b45714c9b9
begin # try modifying these parameters
	features = [:mag_4_uc,:mag_4_uc_dot,:mag_4_uc_dot4,:TL_A_flux_a]
	use_mag  = :mag_4_uc
	use_vec  = :flux_d
	terms    = [:p3,:i3,:e3]
end;

# ╔═╡ 2bb45c4c-4d7a-486c-8e6f-2d4e4419ce83
comp_params_init = NNCompParams(features_setup = features,
	                            model_type     = :m1,
                                y_type         = :d,
                                use_mag        = use_mag,
                                use_vec        = use_vec,
                                terms          = terms,
                                epoch_adam     = 100);

# ╔═╡ 951feb00-5557-4558-94cb-7af2af2b8083
md"Select training & testing flights from Flight 1006 (see [readme](https://github.com/MIT-AI-Accelerator/MagNav.jl/blob/master/readmes/Flt1006_readme.txt)).
"

# ╔═╡ 20d06753-85f6-428a-acb7-b9ca3b67b78a
begin
	lines_train = [1006.03, 1006.04, 1006.05, 1006.06]
	lines_test  = [1006.08]
end;

# ╔═╡ 43dbbbb1-e4b9-4ef5-b5f1-b7da32e6f8e1
md"Perform neural network-based calibration using training data & extract trained neural network (NN) compensation model. The full list of SGL flights is in `df_flight`, the full list of maps is in `df_map`, & the full list of flight lines is in `df_all`.
"

# ╔═╡ fd8adf67-48b5-487b-9a14-8f4acf47d1d6
begin
	(comp_params,_,_,_,feats) =
	    comp_train(comp_params_init,lines_train,df_all,df_flight,df_map)
	m = comp_params.model # extract trained NN model
end;

# ╔═╡ 37e7cbc1-abef-4ea5-b767-e169a2294a51
md"Get training & testing data & normalize by feature (columns). Typically this is done internally, but shown here to better explain feature importance/selection. The full list of navigation-capable flight lines is in `df_nav`.
"

# ╔═╡ 0f23b8ae-a153-4018-a2e2-78cf462a03e8
begin
	(_,x_train,y_train,_,_,_) =
	    get_Axy(lines_train,df_all,df_flight,df_map,features;
	            use_mag=use_mag,use_vec=use_vec,terms=terms)
	(_,x_test,y_test,_,_,_) =
	    get_Axy(lines_test ,df_nav,df_flight,df_map,features;
	            use_mag=use_mag,use_vec=use_vec,terms=terms)
	(x_bias,x_scale,x_train_norm,x_test_norm) = norm_sets(x_train,x_test)
	(y_bias,y_scale,y_train_norm,y_test_norm) = norm_sets(y_train,y_test)
end;

# ╔═╡ f2c49313-6599-4f01-ba6c-a19d15733254
begin
	using DataFrames: sort
	N_gsa  = length(y_test_norm) # number of samples to use for explanation
	means  = eval_gsa(m,Float32.(x_test_norm),N_gsa)
	df_gsa = sort(DataFrame(feature=feats,means=means),:means,by=abs,rev=true)
end

# ╔═╡ 876aaa7a-277a-4012-9409-a9d59e069c55
md"## Shapley-based feature importance

Determine & plot Shapley effects.
"

# ╔═╡ fc389949-680c-4837-bb67-7769cdd50370
begin
	N_shap     = length(y_test_norm) # number of samples to use for explanation
	range_shap = 1:12                # (ranked) features to plot
	(df_shap,baseline_shap) = eval_shapley(m,Float32.(x_test_norm),feats,N_shap)
	p1 = plot_shapley(df_shap,baseline_shap,range_shap)
end

# ╔═╡ b1f50703-ccb8-48b6-83ac-25c0a85554ed
md"## Global sensitivity analysis (GSA)-based feature importance

List of most important features.
"

# ╔═╡ f3633d2c-f008-4fae-9ef0-cb2c097e6967
md"## Sparse group Lasso (SGL)-based feature importance

List of most important features.
"

# ╔═╡ 48155392-9bc7-4c80-abc6-98e96434802f
begin
	α_sgl = 0.5
	λ_sgl = 1e-5
	comp_params_sgl_init = NNCompParams(comp_params_init,α_sgl=α_sgl,λ_sgl=λ_sgl)
	comp_params_sgl =
		comp_train(comp_params_sgl_init,lines_train,df_all,df_flight,df_map)[1]
	m_sgl  = comp_params_sgl.model # extract trained NN model
	w_sgl  = comp_params_sgl.data_norms[3]*sparse_group_lasso(m_sgl,1)
	df_sgl = sort(DataFrame(feature=feats,w_norm=w_sgl),:w_norm,by=abs,rev=true)
	# m_sgl_  = comp_params.model # extract trained NN model
	# w_sgl_  = comp_params.data_norms[3]*sparse_group_lasso(m_sgl_,1)
	# df_sgl_ = sort(DataFrame(feature=feats,w_norm=w_sgl_),:w_norm,by=abs,rev=true)
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
MagNav = "~1.2.2"
Plots = "~1.40.9"
Random = "~1.11.0"
Statistics = "~1.11.1"
"""

# ╔═╡ Cell order:
# ╟─66e96968-4daa-11ee-31cd-7b4d78033095
# ╟─b1d3b1b3-db8d-4bb0-a884-d57f217fef24
# ╠═5e16a321-3321-4d54-899e-594eb670e681
# ╟─4e5e2a24-cc3f-4766-a6f5-c94ff1f6ac2f
# ╠═1c4d1729-19eb-4b2e-8775-a7b45714c9b9
# ╠═2bb45c4c-4d7a-486c-8e6f-2d4e4419ce83
# ╟─951feb00-5557-4558-94cb-7af2af2b8083
# ╠═20d06753-85f6-428a-acb7-b9ca3b67b78a
# ╟─43dbbbb1-e4b9-4ef5-b5f1-b7da32e6f8e1
# ╠═fd8adf67-48b5-487b-9a14-8f4acf47d1d6
# ╟─37e7cbc1-abef-4ea5-b767-e169a2294a51
# ╠═0f23b8ae-a153-4018-a2e2-78cf462a03e8
# ╟─876aaa7a-277a-4012-9409-a9d59e069c55
# ╠═fc389949-680c-4837-bb67-7769cdd50370
# ╟─b1f50703-ccb8-48b6-83ac-25c0a85554ed
# ╠═f2c49313-6599-4f01-ba6c-a19d15733254
# ╟─f3633d2c-f008-4fae-9ef0-cb2c097e6967
# ╠═48155392-9bc7-4c80-abc6-98e96434802f
# ╟─00000000-0000-0000-0000-000000000001
