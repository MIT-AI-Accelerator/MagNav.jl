using SafeTestsets

@safetestset "Delta Lat Lon   Test" begin include("delta_lat_lon_test.jl") end
@safetestset "Helpers         Test" begin include("helpers_test.jl") end
@safetestset "MagNav Artifact Test" begin include("MagNav_artifact_test.jl") end
@safetestset "Tolles-Lawson   Test" begin include("TL_test.jl") end
