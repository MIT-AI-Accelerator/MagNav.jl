using Documenter, MagNav, PlutoStaticHTML

# define directory paths
package_dir  = pkgdir(MagNav)
docs_src_dir = joinpath(package_dir,"docs","src")
folder       = "examples"

# run Pluto notebooks in src_dir & create Markdown files
# done sequentially to avoid recompiling multiple times
println("building example Pluto notebooks")
src_dir = joinpath(package_dir,folder)
bopts   = BuildOptions(src_dir;
                       output_format   = documenter_output,
                       use_distributed = false)
build_notebooks(bopts)

# map sidebar names to Markdown files
notebooks = [
    "Feature Importance"    => "$folder/pluto_fi.md",
    "Linear Models"         => "$folder/pluto_linear.md",
    "Magnetic Anomaly Maps" => "$folder/pluto_maps.md",
    "Model 3"               => "$folder/pluto_model3.md",
    "Using SGL Data"        => "$folder/pluto_sgl.md",
    "Using Simulated Data"  => "$folder/pluto_sim.md",
]

# create folder in docs/src if it doesn't exist
dst_dir = joinpath(docs_src_dir,folder)
isdir(dst_dir) || mkdir(dst_dir)

# move Markdown files into folder in docs/src
for notebook in notebooks
    src = joinpath(package_dir ,notebook[2])
    dst = joinpath(docs_src_dir,notebook[2])
    mv(src,dst;force=true)
end

makedocs(
    modules  = [MagNav],
    sitename = "MagNav.jl",
    format   = Documenter.HTML(
        assets              = ["assets/favicon.ico"],
        mathengine          = Documenter.MathJax3(),
        size_threshold      = nothing,
        size_threshold_warn = nothing,
    ),
    pages = [
        "Home"                      => "index.md",
        "Custom Structs"            => "structs.md",
        "Flight Path & INS Data"    => "data.md",
        "Magnetic Anomaly Maps"     => "maps.md",
        "Aeromagnetic Compensation" => "comp.md",
        "NN-Based Model Diagrams"   => "nncomp.md",
        "Navigation Algorithms"     => "nav.md",
        "Example Pluto Notebooks"   => notebooks,
    ],
    checkdocs = :exports,
    warnonly  = true,
    # remotes   = nothing, #* uncomment for local development
)

deploydocs(
    repo = "github.com/MIT-AI-Accelerator/MagNav.jl.git",
)
