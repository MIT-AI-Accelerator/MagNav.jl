using SafeTestsets

@safetestset "First (Sample)  Test" begin include("test_1.jl") end
@safetestset "Delta Lat Lon   Test" begin include("delta_lat_lon_test.jl") end
@safetestset "FFT Maps        Test" begin include("fft_maps_test.jl") end
@safetestset "Interpolate Map Test" begin include("gen_interp_map_test.jl") end
@safetestset "Helpers         Test" begin include("helpers_test.jl") end
@safetestset "Tolles-Lawson   Test" begin include("TL_test.jl") end
