##* load MagNav & other packages commonly used in REPL =======================
#*  adjust the environment activation location & install packages as needed
using Pkg; Pkg.activate("../"); Pkg.instantiate()
using Revise
using MagNav
using BenchmarkTools, CSV, DataFrames, Flux, LinearAlgebra, Plots, Zygote
using BSON: bson, load, @load, @save
using DataFrames: outerjoin, sort
using DelimitedFiles: readdlm, writedlm
using Plots: plot, plot!
using Random: rand, randn, randperm, seed!, shuffle
using Statistics: cor, cov, mean, median, std, var

##* load useful DataFrames ===================================================
## SGL flight compensation lines
df_comp = DataFrame(CSV.File("dataframes/df_comp.csv"))
df_comp[!,:flight]   = Symbol.(df_comp[!,:flight])
df_comp[!,:map_name] = Symbol.(df_comp[!,:map_name]) # not ideal, but ok

## SGL flight data HDF5 files
df_flight = DataFrame(CSV.File("dataframes/df_flight.csv"))
df_flight[!,:flight]   = Symbol.(df_flight[!,:flight])
df_flight[!,:xyz_type] = Symbol.(df_flight[!,:xyz_type])
df_flight[!,:xyz_h5]   = convert.(String,df_flight[!,:xyz_h5])
#* to store/load the data locally uncomment the for loop below
#* and make sure the file locations match up with the xyz_h5 column
for (i,flight) in enumerate(df_flight.flight)
    if df_flight.xyz_type[i] == :XYZ20
        df_flight.xyz_h5[i] = MagNav.sgl_2020_train()*"/$(flight)_train.h5"
    end
end

## map data HDF5 files (associated with SGL flights)
df_map = DataFrame(CSV.File("dataframes/df_map.csv"))
df_map[!,:map_name] = Symbol.(df_map[!,:map_name])
df_map[!,:map_type] = Symbol.(df_map[!,:map_type])
df_map[!,:map_h5]   = convert.(String,df_map[!,:map_h5])
#* to store/load the maps locally, uncomment the for loop below
#* and make sure the file locations match up with the map_h5 column
for (i,map_name) in enumerate(df_map.map_name)
    df_map.map_h5[i] = MagNav.ottawa_area_maps()*"/$map_name.h5"
end

## all lines
df_all = DataFrame(CSV.File("dataframes/df_all.csv"))
df_all[!,:flight] = Symbol.(df_all[!,:flight])

## navigation-capable lines
df_nav = DataFrame(CSV.File("dataframes/df_nav.csv"))
df_nav[!,:flight]   = Symbol.(df_nav[!,:flight])
df_nav[!,:map_name] = Symbol.(df_nav[!,:map_name])
df_nav[!,:map_type] = Symbol.(df_nav[!,:map_type])

## in-flight events
df_event = DataFrame(CSV.File("../readmes/pilot_comments.csv"))
df_event[!,:flight] = Symbol.(df_event[!,:flight])

## all lines for flights 3-6, except 1003.05
# 1003.05 not used because of ~1.5 min data anomaly in mag_4_uc & mag_5_uc
# between times 59641 & 59737, causing ~25x NN comp errors
flts = [:Flt1003,:Flt1004,:Flt1005,:Flt1006]
df_all_3456 = df_all[(df_all.flight .∈ (flts,)) .& (df_all.line.!=1003.05),:]

## navigation-capable lines for flights 3-6
flts = [:Flt1003,:Flt1004,:Flt1005,:Flt1006]
df_nav_3456 = df_nav[(df_nav.flight .∈ (flts,)),:]
