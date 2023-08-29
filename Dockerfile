# ------------------------------------------------------------------------------
# Build a Docker image containing Julia, Jupyter, Pluto, MagNav.jl, & examples.
# It will take a while to build. The image is large, about 6 GB. This uses:
# https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#jupyter-julia-notebook
# ------------------------------------------------------------------------------
# Build/Run commands via Docker
# docker build --tag magnav .
# docker run -p 8888:8888 magnav
# ------------------------------------------------------------------------------

# Get Julia, Jupyter, & Pluto image, might not use latest Julia
FROM jupyter/julia-notebook:latest

# Add packages & precompile
RUN julia -e 'import Pkg; Pkg.update(); \
    Pkg.add(["MagNav","CSV","DataFrames","Plots"]); \
    Pkg.precompile();'

# Download examples
RUN git clone "https://github.com/MIT-AI-Accelerator/MagNav.jl" $HOME/MagNav.jl && \
    cp -r $HOME/MagNav.jl/examples/. $HOME && \
    rm -r $HOME/MagNav.jl && \
    rm -r $HOME/work

# Fix permissions
RUN fix-permissions $HOME
