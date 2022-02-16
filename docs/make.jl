using Documenter, MagNav

makedocs(
    modules = [MagNav],
    checkdocs = :exports,
    sitename = "MagNav.jl",
    pages = Any["index.md"],
    repo = "https://gitlab.com/gnadt/MagNav.jl/blob/{commit}{path}#{line}"
)
