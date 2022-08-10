using MagNav, Test, MAT, Plots

test_file = "test_data/test_data_grid.mat"
grid_data = matopen(test_file,"r") do file
    read(file,"grid_data")
end

map_file  = "test_data/test_data_map.mat"
mapS      = get_map(map_file)
itp_mapS  = map_interpolate(mapS,:linear) # linear to match MATLAB

traj_file = "test_data/test_data_traj.mat"
traj      = get_traj(traj_file,:traj,silent=true)

mapV = get_map(MagNav.emm720)
mapV = map_trim(mapV,traj)

@testset "map_interpolate tests" begin
    @test itp_mapS(traj.lon[1],traj.lat[1]) ≈ grid_data["itp_map"]
    @test_nowarn map_interpolate(mapS,:quad)
    @test_nowarn map_interpolate(mapS,:cubic)
    @test_throws ErrorException map_interpolate(mapS,:test)
    @test_nowarn map_interpolate(mapV,:X)
    @test_nowarn map_interpolate(mapV,:Y)
    @test_nowarn map_interpolate(mapV,:Z)
    @test_throws ErrorException map_interpolate(mapV,:test)
end

@testset "map_params tests" begin
    @test_nowarn MagNav.map_params(mapS)
    @test_nowarn MagNav.map_params(mapV)
end

@testset "plot_map tests" begin
    @test typeof(plot_map(mapS)) <: Plots.Plot
end

@testset "map_cs tests" begin
    for map_color in [:usgs,:gray,:gray1,:gray2,:plasma,:magma]
        @test typeof(MagNav.map_cs(map_color)) <: Plots.ColorGradient
    end
end

@testset "plot_path tests" begin
    @test typeof(plot_path(traj;show_plot=false)) <: Plots.Plot
end

@testset "map_check tests" begin
    @test map_check(mapS,traj) ≈ true
    @test map_check(mapV,traj) ≈ true
    @test all(map_check([mapS,mapV],traj)) ≈ true
end
