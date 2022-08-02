using MagNav, Test, DataFrames

test_data_map_mat = "test_data/test_data_map.mat"

data_dir          = MagNav.ottawa_area_maps()
Eastern_395_h5    = string(data_dir,"/Eastern_395.h5")
Eastern_drape_h5  = string(data_dir,"/Eastern_drape.h5")
Eastern_plot_h5   = string(data_dir,"/Eastern_plot.h5")
HighAlt_5181_h5   = string(data_dir,"/HighAlt_5181.h5")
Perth_800_h5      = string(data_dir,"/Perth_800.h5")
Renfrew_395_h5    = string(data_dir,"/Renfrew_395.h5")
Renfrew_555_h5    = string(data_dir,"/Renfrew_555.h5")
Renfrew_drape_h5  = string(data_dir,"/Renfrew_drape.h5")
Renfrew_plot_h5   = string(data_dir,"/Renfrew_plot.h5")

map_files = [test_data_map_mat,MagNav.emag2,MagNav.emm720,MagNav.namad,
             Eastern_drape_h5,Eastern_drape_h5,Eastern_plot_h5,
             HighAlt_5181_h5,Perth_800_h5,
             Renfrew_395_h5,Renfrew_555_h5,Renfrew_drape_h5,Renfrew_plot_h5]
map_names = [:map_1,:map_2,:map_3,:map_4,:map_5,:map_6,:map_7,:map_8,:map_9,
             :map_10,:map_11,:map_12,:map_13]
df_map    = DataFrame(map_h5=map_files,map_name=map_names)

@testset "get_map tests" begin
    for map_file in map_files
        @test_nowarn get_map(map_file);
    end
    for map_name in map_names
        @test_nowarn get_map(map_name,df_map);
    end
end
