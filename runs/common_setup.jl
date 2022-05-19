##* load MagNav & other packages commonly used in REPL =======================
using Pkg; Pkg.activate("../"); Pkg.instantiate()
using Revise
using MagNav
using BenchmarkTools, DataFrames, Flux, LinearAlgebra, Plots, Zygote
using BSON: @load, @save
using CSV
using DataFrames: sort
using DelimitedFiles: readdlm, writedlm
using Plots: plot, plot!
using Random: rand, randn, randperm, seed!, shuffle
using Statistics: cor, cov, mean, median, std, var

##* load useful DataFrames ===================================================
## SGL flight compensation lines
df_comp = 0.0
@load "dataframes/df_comp.bson" df_comp
df_comp[!,:flight]   = convert.(Symbol,df_comp[!,:flight])
df_comp[!,:map_name] = convert.(Union{Missing,Symbol},df_comp[!,:map_name])

## SGL flight files
df_flight  = 0.0
@load "dataframes/df_flight.bson" df_flight
df_flight[!,:flight]   = convert.(Symbol,df_flight[!,:flight])
df_flight[!,:xyz_type] = convert.(Symbol,df_flight[!,:xyz_type])
df_flight[!,:xyz_h5]   = convert.(String,df_flight[!,:xyz_h5])
#* if you would like to keep the data locally, uncomment the for loop below
#* and make sure the data file locations match up with the xyz_h5 column
for (i,flight) in enumerate(df_flight.flight)
    if df_flight.xyz_type[i] == :XYZ20
        df_flight.xyz_h5[i] = string(MagNav.sgl_2020_train(),"/$(flight)_train.h5")
    end
end

## map files (associated with SGL flights)
df_map  = 0.0
@load "dataframes/df_map.bson" df_map
df_map[!,:map_name] = convert.(Symbol,df_map[!,:map_name])
df_map[!,:map_type] = convert.(Symbol,df_map[!,:map_type])
df_map[!,:map_h5]   = convert.(String,df_map[!,:map_h5])
#* if you would like to keep the maps locally, uncomment the for loop below
#* and make sure the maps file locations match up with the map_h5 column
for (i,map_name) in enumerate(df_map.map_name)
    df_map.map_h5[i] = string(MagNav.ottawa_area_maps(),"/$map_name.h5")
end

## all lines
df_all = 0.0
@load "dataframes/df_all.bson" df_all
df_all[!,:flight] = convert.(Symbol,df_all[!,:flight])

## navigation-capable lines
df_nav  = 0.0
@load "dataframes/df_nav.bson" df_nav
df_nav[!,:flight]   = convert.(Symbol,df_nav[!,:flight])
df_nav[!,:map_name] = convert.(Symbol,df_nav[!,:map_name])
df_nav[!,:map_type] = convert.(Symbol,df_nav[!,:map_type])

## in-flight events
df_event = DataFrame(readdlm("../readmes/pilot_comments.csv",',',skipstart=1),:auto)
rename!(df_event, :x1 => :flight)
rename!(df_event, :x2 => :t)
rename!(df_event, :x3 => :event)
df_event[!,:flight] = Symbol.(df_event[!,:flight])
df_event[!,:t]      = float.( df_event[!,:t])
df_event[!,:event]  = String.(df_event[!,:event])

## all lines for flights 3-6, except 1003.05
# 1003.05 not used because of ~1.5 min data anomaly in mag_4_uc & mag_5_uc
# between times 59641 & 59737, causing ~25x NN comp errors
flts = [:Flt1003,:Flt1004,:Flt1005,:Flt1006]
df_all_3456 = df_all[(df_all.flight .∈ (flts,)) .& (df_all.line.!=1003.05),:];

## navigation-capable lines for flights 3-6
flts = [:Flt1003,:Flt1004,:Flt1005,:Flt1006]
df_nav_3456 = df_nav[(df_nav.flight .∈ (flts,)),:];
