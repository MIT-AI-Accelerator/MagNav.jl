module MagNav

using Pkg.Artifacts: @artifact_str

data_dir() = joinpath(artifact"flight_data", "flight_data")

greet() = print("Hello World!")

end # module
