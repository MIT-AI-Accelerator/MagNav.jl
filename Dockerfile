# Adapted from https://github.com/andferrari/julia_notebook,
# which was derived from https://github.com/jupyter/docker-stacks
# 
# Licensing terms for this Dockerfile: 
# -------------------------------------------------------------------------------- 
# This project is licensed under the terms of the Modified BSD License
# (also known as New or Revised or 3-Clause BSD), as follows:
# 
# - Copyright (c) 2001-2015, IPython Development Team
# - Copyright (c) 2015-, Jupyter Development Team
# 
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# 
# Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
# 
# Neither the name of the Jupyter Development Team nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -------------------------------------------------------------------------------- 

# Build/Run commands via Docker
# -------------------------------------------------------------------------------- 
# docker build --tag magnav .
# docker run -p 8888:8888 -v `pwd`:/home/jovyan/work magnav
FROM "jupyter/minimal-notebook"

USER root

ENV JULIA_VERSION=1.8.5

# Install Julia
RUN mkdir /opt/julia-${JULIA_VERSION} && \
    cd /tmp && \
    wget -q https://julialang-s3.julialang.org/bin/linux/x64/`echo ${JULIA_VERSION} | cut -d. -f 1,2`/julia-${JULIA_VERSION}-linux-x86_64.tar.gz && \
    tar xzf julia-${JULIA_VERSION}-linux-x86_64.tar.gz -C /opt/julia-${JULIA_VERSION} --strip-components=1 && \
    rm /tmp/julia-${JULIA_VERSION}-linux-x86_64.tar.gz

RUN ln -fs /opt/julia-*/bin/julia /usr/local/bin/julia

USER $NB_UID

# Add packages and precompile
RUN julia -e 'import Pkg; Pkg.update()' && \
    julia -e 'import Pkg; Pkg.add("Plots"); using Plots' && \
    julia -e 'import Pkg; Pkg.add("Distributions"); using Distributions' && \
    julia -e 'import Pkg; Pkg.add("Optim"); using Optim' && \  
    julia -e 'import Pkg; Pkg.add("FFTW"); using FFTW' && \  
    # julia -e 'import Pkg; Pkg.add("StatsPlots"); using StatsPlots' && \  
    julia -e 'import Pkg; Pkg.add("DSP"); using DSP' && \  
    julia -e 'import Pkg; Pkg.add("IJulia"); using IJulia' && \
    julia -e 'import Pkg; Pkg.add("DataFrames"); using DataFrames' && \
    julia -e 'import Pkg; Pkg.add("CSV"); using CSV' && \
    julia -e 'import Pkg; Pkg.add("Flux"); using Flux' && \
    julia -e 'import Pkg; Pkg.add("Zygote"); using Zygote' && \
    julia -e 'import Pkg; Pkg.add("BSON"); using BSON' && \
    julia -e 'import Pkg; Pkg.add("BenchmarkTools"); using BenchmarkTools' && \
    julia -e 'import Pkg; Pkg.add("Revise"); using Revise'

# Install MagNav and add common files
RUN julia -e 'import Pkg; Pkg.add(url="https://github.com/MIT-AI-Accelerator/MagNav.jl"); using MagNav' 
RUN git clone "https://github.com/MIT-AI-Accelerator/MagNav.jl" /home/$NB_USER/work/MagNav.jl
RUN julia -e 'cd("work/MagNav.jl/runs"); include("common_setup.jl");'

# Bring in the notebook for convenience
RUN cp /home/$NB_USER/work/MagNav.jl/runs/Demo.ipynb /home/$NB_USER/Demo.ipynb

RUN fix-permissions /home/$NB_USER
