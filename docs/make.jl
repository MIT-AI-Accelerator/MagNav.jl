using Documenter, MagNav

makedocs(
    modules = [MagNav],
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
    ),
    checkdocs = :exports,
    sitename = "MagNav.jl",
    pages = Any[
        "index.md",
        "data.md",
        "maps.md",
        "comp.md",
        "nav.md"],
)

deploydocs(
    repo = "github.com/MIT-AI-Accelerator/MagNav.jl.git",
)