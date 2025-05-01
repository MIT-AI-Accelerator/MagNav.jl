package_dir = normpath(joinpath(@__DIR__,"..")) # package directory

#* uncomment for local development
# using Pkg; Pkg.develop(PackageSpec(path=package_dir)); Pkg.instantiate()

using Documenter, MagNav, PlutoStaticHTML

# run Pluto notebooks in nb_dir & create Markdown files
# done sequentially to avoid recompiling multiple times
folder = "examples"
nb_dir = joinpath(package_dir,folder)
b_opts = PlutoStaticHTML.BuildOptions(nb_dir;
                                      output_format   = documenter_output,
                                      use_distributed = false)
println("building example Pluto notebooks")
PlutoStaticHTML.build_notebooks(b_opts)

# map sidebar names to Markdown files
notebooks = [
    "Feature Importance"    => joinpath(folder,"pluto_fi.md"),
    "Linear Models"         => joinpath(folder,"pluto_linear.md"),
    "Magnetic Anomaly Maps" => joinpath(folder,"pluto_maps.md"),
    "Model 3"               => joinpath(folder,"pluto_model3.md"),
    "Using SGL Data"        => joinpath(folder,"pluto_sgl.md"),
    "Using Simulated Data"  => joinpath(folder,"pluto_sim.md"),
]

# create folder in docs_src_dir if it doesn't exist
docs_src_dir = joinpath(package_dir,"docs","src")
dst_dir      = joinpath(docs_src_dir,folder)
isdir(dst_dir) || mkdir(dst_dir)

# move Markdown files into folder in docs_src_dir
for notebook in notebooks
    src = joinpath(package_dir ,notebook[2])
    dst = joinpath(docs_src_dir,notebook[2])
    println("moving $src to $dst")
    mv(src,dst;force=true)
end

Documenter.makedocs(
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
        "API: Functions"            => "api_functions.md",
        "API: Structs"              => "api_structs.md",
        "Flight Path & INS Data"    => "data.md",
        "Magnetic Anomaly Maps"     => "maps.md",
        "Aeromagnetic Compensation" => "comp.md",
        "NN-Based Model Diagrams"   => "nncomp.md",
        "Navigation Algorithms"     => "nav.md",
        "Example Pluto Notebooks"   => notebooks,
    ],
    checkdocs = :exports,
    warnonly  = true,
)

#* comment for local development
Documenter.deploydocs(
    repo = "github.com/MIT-AI-Accelerator/MagNav.jl.git",
)
