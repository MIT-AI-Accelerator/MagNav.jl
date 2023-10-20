using Documenter, MagNav

makedocs(
    modules = [MagNav],
    format  = Documenter.HTML(
        assets     = ["assets/favicon.ico"],
        mathengine = Documenter.MathJax3(),
    ),
    checkdocs = :exports,
    sitename  = "MagNav.jl",
    pages = [
        "Home"                      => "index.md",
        "Custom Structs"            => "structs.md",
        "Flight Path & INS Data"    => "data.md",
        "Magnetic Anomaly Maps"     => "maps.md",
        "Aeromagnetic Compensation" => "comp.md",
        "NN-Based Model Diagrams"   => "nncomp.md",
        "Navigation Algorithms"     => "nav.md",
    ],
)

deploydocs(
    repo = "github.com/MIT-AI-Accelerator/MagNav.jl.git",
)
