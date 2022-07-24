using Documenter, MagNav

makedocs(
    modules = [MagNav],
    checkdocs = :exports,
    sitename = "MagNav.jl",
    pages = Any["index.md"],
)

deploydocs(
    repo = "github.com/MIT-AI-Accelerator/MagNav.jl.git",
)