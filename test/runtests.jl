##* top-level file for running unit tests
#*  note that all non-hardcoded test data was generated using the MagNav 
#*  MATLAB-companion (run test_baseline.m & copy the test_data folder over)
using TestItemRunner

@testitem "analysis_util   " begin include("test_analysis_util.jl") end
@testitem "baseline_plots  " begin include("test_baseline_plots.jl") end
# @testitem "compensation    " begin include("test_compensation.jl") end
@testitem "create_XYZ0     " begin include("test_create_XYZ0.jl") end
@testitem "dcm             " begin include("test_dcm.jl") end
@testitem "ekf_&_crlb      " begin include("test_ekf_&_crlb.jl") end
@testitem "ekf_online_nn   " begin include("test_ekf_online_nn.jl") end
@testitem "ekf_online      " begin include("test_ekf_online.jl") end
@testitem "eval_filt       " begin include("test_eval_filt.jl") end
@testitem "get_map         " begin include("test_get_map.jl") end
@testitem "get_XYZ         " begin include("test_get_XYZ.jl") end
@testitem "get_XYZ0        " begin include("test_get_XYZ0.jl") end
@testitem "google_earth    " begin include("test_google_earth.jl") end
@testitem "map_fft         " begin include("test_map_fft.jl") end
@testitem "map_functions   " begin include("test_map_functions.jl") end
@testitem "model_functions " begin include("test_model_functions.jl") end
@testitem "mpf             " begin include("test_mpf.jl") end
@testitem "nekf            " begin include("test_nekf.jl") end
@testitem "tolles_lawson   " begin include("test_tolles_lawson.jl") end
@testitem "xyz2h5          " begin include("test_xyz2h5.jl") end

@run_package_tests
