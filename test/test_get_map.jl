using MagNav, Test, MAT, DataFrames

test_data_map = "test_data/test_data_map.mat"
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

map_data_drpV = Dict("mapX" => map_map[ind,ind],
                     "mapY" => map_map[ind,ind],
                     "mapZ" => map_map[ind,ind],
                     "xx"   => map_xx[ind],
                     "yy"   => map_yy[ind],
                     "alt"  => map_map[ind,ind])

test_data_map_badS = "test_data_map_badS.mat"
map_data  = matopen(test_data_map_badS,"w") do file
    write(file,"map_data",map_data_badS)
end

test_data_map_badV = "test_data_map_badV.mat"
map_data  = matopen(test_data_map_badV,"w") do file
    write(file,"map_data",map_data_badV)
end

test_data_map_drpS = "test_data_map_drpS.mat"
map_data  = matopen(test_data_map_drpS,"w") do file
    write(file,"map_data",map_data_drpS)
end

test_data_map_drpV = "test_data_map_drpV.mat"
map_data  = matopen(test_data_map_drpV,"w") do file
    write(file,"map_data",map_data_drpV)
end

data_dir           = MagNav.ottawa_area_maps()
Eastern_395_h5     = string(data_dir,"/Eastern_395.h5")
Eastern_drape_h5   = string(data_dir,"/Eastern_drape.h5")
Eastern_plot_h5    = string(data_dir,"/Eastern_plot.h5")
HighAlt_5181_h5    = string(data_dir,"/HighAlt_5181.h5")
Perth_800_h5       = string(data_dir,"/Perth_800.h5")
Renfrew_395_h5     = string(data_dir,"/Renfrew_395.h5")
Renfrew_555_h5     = string(data_dir,"/Renfrew_555.h5")
Renfrew_drape_h5   = string(data_dir,"/Renfrew_drape.h5")
Renfrew_plot_h5    = string(data_dir,"/Renfrew_plot.h5")

map_files = [test_data_map,test_data_map_drpS,test_data_map_drpV,
             MagNav.emag2,MagNav.emm720,MagNav.namad,
             Eastern_drape_h5,Eastern_drape_h5,Eastern_plot_h5,
             HighAlt_5181_h5,Perth_800_h5,
             Renfrew_395_h5,Renfrew_555_h5,Renfrew_drape_h5,Renfrew_plot_h5]
map_names = [:map_1,:map_2,:map_3,:map_4,:map_5,:map_6,:map_7,:map_8,:map_9,
             :map_10,:map_11,:map_12,:map_13,:map_14,:map_15]
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
    @test_throws ErrorException get_map("test")
    @test_throws ErrorException get_map(test_data_map_badS)
    @test_throws ErrorException get_map(test_data_map_badV)
end

mapS   = get_map(test_data_map)
map_h5 = "test.h5"

@testset "save_map tests" begin
    @test typeof(save_map(mapS,map_h5;map_units=:deg )) <: Nothing
    @test typeof(save_map(mapS,map_h5;map_units=:rad )) <: Nothing
    @test typeof(save_map(mapS,map_h5;map_units=:utm )) <: Nothing
    @test typeof(save_map(mapS,map_h5;map_units=:test)) <: Nothing
end

rm(test_data_map_badS)
rm(test_data_map_badV)
rm(test_data_map_drpS)
rm(test_data_map_drpV)
rm(map_h5)
