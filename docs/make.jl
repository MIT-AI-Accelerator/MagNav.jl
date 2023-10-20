using Documenter, MagNav, PlutoStaticHTML

# run Pluto notebooks in notebooks_dir & write to Markdown files
# done sequentially to avoid recompiling multiple times
println("building example notebooks")
notebooks_dir = joinpath(pkgdir(MagNav),"examples")
build_options = BuildOptions(notebooks_dir;
                             output_format   = documenter_output,
                             use_distributed = false)
build_notebooks(build_options)

notebooks = [
    "Feature Importance"    => "../../examples/pluto_fi.md",
    "Linear Models"         => "../../examples/pluto_linear.md",
    "Magnetic Anomaly Maps" => "../../examples/pluto_maps.md",
    "Model 3"               => "../../examples/pluto_model3.md",
    "Using SGL Data"        => "../../examples/pluto_sgl.md",
    "Using Simulated Data"  => "../../examples/pluto_sim.md",
]

makedocs(
    modules = [MagNav],
    format  = Documenter.HTML(
        assets     = ["assets/favicon.ico"],
        mathengine = Documenter.MathJax3(),
    ),
    checkdocs = :exports,
    warnonly  = true,
    sitename  = "MagNav.jl",
    pages = [
        "Home"                      => "index.md",
        "Custom Structs"            => "structs.md",
        "Flight Path & INS Data"    => "data.md",
        "Magnetic Anomaly Maps"     => "maps.md",
        "Aeromagnetic Compensation" => "comp.md",
        "NN-Based Model Diagrams"   => "nncomp.md",
        "Navigation Algorithms"     => "nav.md",
        "Example Notebooks"         => notebooks,
    ],
)

deploydocs(
    repo = "github.com/MIT-AI-Accelerator/MagNav.jl.git",
)
