##* top-level file for running unit tests
#*  note that all non-hardcoded test data was generated using the MagNav 
#*  MATLAB-companion (run test_baseline.m & copy the test_data folder over)
using Flux, MagNav, SafeTestsets, Zygote # todo: https://github.com/JuliaIO/BSON.jl#notes

@safetestset "analysis_util   " begin include("test_analysis_util.jl") end
@safetestset "baseline_plots  " begin include("test_baseline_plots.jl") end
@safetestset "compensation    " begin include("test_compensation.jl") end
@safetestset "create_XYZ0     " begin include("test_create_XYZ0.jl") end
@safetestset "dcm             " begin include("test_dcm.jl") end
@safetestset "ekf_&_crlb      " begin include("test_ekf_&_crlb.jl") end
@safetestset "ekf_online_nn   " begin include("test_ekf_online_nn.jl") end
@safetestset "ekf_online      " begin include("test_ekf_online.jl") end
@safetestset "eval_filt       " begin include("test_eval_filt.jl") end
@safetestset "get_map         " begin include("test_get_map.jl") end
@safetestset "get_XYZ         " begin include("test_get_XYZ.jl") end
@safetestset "get_XYZ0        " begin include("test_get_XYZ0.jl") end
@safetestset "google_earth    " begin include("test_google_earth.jl") end
@safetestset "map_fft         " begin include("test_map_fft.jl") end
@safetestset "map_functions   " begin include("test_map_functions.jl") end
@safetestset "model_functions " begin include("test_model_functions.jl") end
@safetestset "mpf             " begin include("test_mpf.jl") end
@safetestset "nekf            " begin include("test_nekf.jl") end
@safetestset "tolles_lawson   " begin include("test_tolles_lawson.jl") end
@safetestset "xyz2h5          " begin include("test_xyz2h5.jl") end
