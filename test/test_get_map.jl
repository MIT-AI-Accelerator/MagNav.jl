using MagNav, Test, MAT, DataFrames

test_data_map = joinpath(@__DIR__,"test_data/test_data_map.mat")
map_data  = matopen(test_data_map,"r") do file
    read(file,"map_data")
end

map_map = map_data["map"]
map_xx  = deg2rad.(vec(map_data["xx"]))
map_yy  = deg2rad.(vec(map_data["yy"]))
map_alt = map_data["alt"]

ind = trues(length(map_xx))
ind[51:end] .= false

map_data_badS = Dict("map" => map_map[ind,ind],
                     "xx"  => map_xx[1:10],
                     "yy"  => map_yy[ind],
                     "alt" => map_alt)

map_data_badV = Dict("mapX" => map_map[ind,ind],
                     "mapY" => map_map[ind,ind],
                     "mapZ" => map_map[ind,ind],
                     "xx"   => map_xx[1:10],
                     "yy"   => map_yy[ind],
                     "alt"  => map_alt)

map_data_drpS = Dict("map" => map_map[ind,ind],
                     "xx"  => map_xx[ind],
                     "yy"  => map_yy[ind],
                     "alt" => map_map[ind,ind])

test_data_map_badS = joinpath(@__DIR__,"test_data_map_badS.mat")
map_data  = matopen(test_data_map_badS,"w") do file
    write(file,"map_data",map_data_badS)
end

test_data_map_badV = joinpath(@__DIR__,"test_data_map_badV.mat")
map_data  = matopen(test_data_map_badV,"w") do file
    write(file,"map_data",map_data_badV)
end

test_data_map_drpS = joinpath(@__DIR__,"test_data_map_drpS.mat")
map_data  = matopen(test_data_map_drpS,"w") do file
    write(file,"map_data",map_data_drpS)
end

data_dir         = MagNav.ottawa_area_maps()
Eastern_395_h5   = data_dir*"/Eastern_395.h5"
Eastern_drape_h5 = data_dir*"/Eastern_drape.h5"
Eastern_plot_h5  = data_dir*"/Eastern_plot.h5"
HighAlt_5181_h5  = data_dir*"/HighAlt_5181.h5"
Perth_800_h5     = data_dir*"/Perth_800.h5"
Renfrew_395_h5   = data_dir*"/Renfrew_395.h5"
Renfrew_555_h5   = data_dir*"/Renfrew_555.h5"
Renfrew_drape_h5 = data_dir*"/Renfrew_drape.h5"
Renfrew_plot_h5  = data_dir*"/Renfrew_plot.h5"

# emag2, emm720, & namad all tested elsewhere
map_files = [test_data_map,test_data_map_drpS,
             Eastern_drape_h5,Eastern_drape_h5,Eastern_plot_h5,
             HighAlt_5181_h5,Perth_800_h5,
             Renfrew_395_h5,Renfrew_555_h5,Renfrew_drape_h5,Renfrew_plot_h5]
map_names = [:map_1,:map_2,:map_3,:map_4,:map_5,:map_6,:map_7,:map_8,:map_9,
             :map_10,:map_11]
df_map    = DataFrame(map_h5=map_files,map_name=map_names)

@testset "get_map tests" begin
    for map_file in map_files
        println(map_file)
        @test_nowarn get_map(map_file)
    end
    for map_name in map_names
        println(map_name)
        @test_nowarn get_map(map_name,df_map)
    end
    @test typeof(get_map(test_data_map;map_units=:utm)) <: MagNav.MapS
    @test_throws AssertionError get_map("test")
    @test_throws ErrorException get_map(test_data_map_badS)
    @test_throws ErrorException get_map(test_data_map_badV)
end

mapS   = get_map(test_data_map)
map_h5 = joinpath(@__DIR__,"test_save_map")
comp_params_lin_bson = joinpath(@__DIR__,"test_save_comp_params_lin")
comp_params_nn_bson  = joinpath(@__DIR__,"test_save_comp_params_nn")

@testset "save_map tests" begin
    @test typeof(save_map(mapS,map_h5;map_units=:deg )) <: Nothing
    @test typeof(save_map(mapS,map_h5;map_units=:rad )) <: Nothing
    @test typeof(save_map(mapS,map_h5;map_units=:utm )) <: Nothing
    @test typeof(save_map(mapS,map_h5;map_units=:test)) <: Nothing
end

@testset "save_comp_params tests" begin
    @test_nowarn save_comp_params(LinCompParams(),comp_params_lin_bson)
    @test_nowarn save_comp_params(NNCompParams() ,comp_params_nn_bson)
end

@testset "get_comp_params tests" begin
    @test typeof(get_comp_params(comp_params_lin_bson)) <: MagNav.LinCompParams
    @test typeof(get_comp_params(comp_params_nn_bson )) <: MagNav.NNCompParams
end

map_h5 = MagNav.add_extension(map_h5,".h5")
comp_params_lin_bson = MagNav.add_extension(comp_params_lin_bson,".bson")
comp_params_nn_bson  = MagNav.add_extension(comp_params_nn_bson ,".bson")

rm(test_data_map_badS)
rm(test_data_map_badV)
rm(test_data_map_drpS)
rm(map_h5)
rm(comp_params_lin_bson)
rm(comp_params_nn_bson)
