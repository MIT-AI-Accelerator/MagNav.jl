##* top-level file for running unit tests
#*  note that all non-hardcoded test data was generated using the MagNav 
#*  MATLAB-companion (run test_baseline.m & copy the test_data folder over)
using SafeTestsets

@safetestset "analysis_util   " begin include("test_analysis_util.jl") end  
@safetestset "create_XYZ0     " begin include("test_create_XYZ0.jl") end
@safetestset "dcm             " begin include("test_dcm.jl") end
@safetestset "ekf_&_crlb      " begin include("test_ekf_&_crlb.jl") end
@safetestset "map_fft         " begin include("test_map_fft.jl") end
@safetestset "map_functions   " begin include("test_map_functions.jl") end
@safetestset "model_functions " begin include("test_model_functions.jl") end
@safetestset "tolles_lawson   " begin include("test_tolles_lawson.jl") end
