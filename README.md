# MagNav: airborne Magnetic anomaly Navigation

[![CI](https://github.com/MIT-AI-Accelerator/MagNav.jl/workflows/CI/badge.svg)](https://github.com/MIT-AI-Accelerator/MagNav.jl/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/MIT-AI-Accelerator/MagNav.jl/branch/master/graph/badge.svg)](https://app.codecov.io/gh/MIT-AI-Accelerator/MagNav.jl)
[![docs-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mit-ai-accelerator.github.io/MagNav.jl/stable/)

MagNav.jl contains a full suite of tools for airborne **Mag**netic anomaly **Nav**igation (MagNav), including flight path & INS data import or simulation, mapping, aeromagnetic compensation, and navigation. This package was developed as part of the [DAF-MIT Artificial Intelligence Accelerator](https://aia.mit.edu/). More information on this effort, including a list of relevant publications, is provided on the [challenge problem website](https://magnav.mit.edu/). The package has been tested on the long-term support (LTS) and latest stable versions of Julia, which may be downloaded from [here](https://julialang.org/downloads/). The recommended IDE for Julia is [Visual Studio Code](https://code.visualstudio.com/) (with the Julia extension).

## Installation

From the Julia REPL, run:

```julia
julia> using Pkg
julia> Pkg.add("MagNav")
```

or from the Julia REPL, type `]` to enter Pkg REPL mode and run:

```julia
pkg> add MagNav
```

or clone/download the MagNav.jl repo and from within the local MagNav.jl folder, run:

```julia
julia> using Pkg
julia> Pkg.activate(".")
julia> Pkg.instantiate()
```

NOTE: If any artifacts produce a hash mismatch error while downloading, simply navigate to the 
`.julia/artifacts` folder and manually set the appropriate artifact folder name. For example, the ottawa_area_maps artifact folder name should be `bf360d29207d6468a8cf783269191bda2cf1f679`.

## Usage

For general usage, run:

```julia
julia> using MagNav
```

### Examples

Multiple example Jupyter and Pluto notebooks are in the [`examples`](examples) folder. Jupyter can be run directly in [Visual Studio Code](https://code.visualstudio.com/) (with the Jupyter extension). To start Pluto in a web browser, run:

```julia
julia> using Pkg
julia> Pkg.add("Pluto") # as necessary
julia> using Pluto
julia> Pluto.run()
```

In Pluto, select and open the desired Pluto notebook, and it should run automatically.

### Docker

A Docker image is available, which contains Julia, MagNav.jl, and the example notebooks. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/), search and pull `jtaylormit/magnav`, and run with the host port set to `8888`. Alternatively, from the command line, run:

```
docker pull jtaylormit/magnav
docker run -p 8888:8888 jtaylormit/magnav
```

A Docker container will spin up and provide a URL that then must be opened in a local browser. It will look something like this: `http://127.0.0.1:8888/lab?token=###`. Jupyter notebooks can be opened directly, but for Pluto notebooks, click `Pluto Notebook` from the Launcher, then select and open the desired Pluto notebook. Note that any changes to the notebooks occur inside the Docker container only, so a notebook must be manually downloaded to be saved.

The above Docker image is [hosted on Docker Hub](https://hub.docker.com/r/jtaylormit/magnav), and it is manually and sporadically updated. For the most recent Docker image, from the command line, run:

```
docker pull ghcr.io/mit-ai-accelerator/magnav.jl
docker run -p 8888:8888 ghcr.io/mit-ai-accelerator/magnav.jl
```

## Data

Publicly available flight data can be automatically downloaded within the package itself. This dataset can also be directly downloaded from [here](https://doi.org/10.5281/zenodo.4271803). See the [datasheet](readmes/datasheet_sgl_2020_train.pdf) for high-level information about this dataset. Details of the flight data are described in the readme files within the [`readmes`](readmes) folder.

### Data Sharing Agreement

Please read the full Data Sharing Agreement located [here](readmes/DATA_SHARING_AGREEMENT.md).

By granting You access to Data, the Air Force grants You a limited personal, non-exclusive, non-transferable, non-assignable, and revocable license to copy, modify, publicly display, and use the Data in accordance with this AGREEMENT solely for the purpose of non-profit research, non-profit education, or for government purposes by or on behalf of the U.S. Government. No license is granted for any other purpose, and there are no implied licenses in this Agreement. This Agreement is effective as of the date of approval by Air Force and remains in force for a period of one year from such date, unless terminated earlier or amended in writing. By using Data, You hereby grant an unlimited, irrevocable, world-wide, royalty-free right to The United States Government to use for any purpose and in any manner whatsoever any feedback from You to the Air Force concerning Your use of Data.

## Citation

If this code or data is used in any work, please cite:

```
[DataSet Name] provided by the United States Air Force pursuant to Cooperative Agreement Number FA8750-19-2-1000 - [dates used]
@article{gnadt2023signal,
  title={Signal Enhancement for Magnetic Navigation Challenge Problem},
  author={Gnadt, Albert R. and Belarge, Joseph and Canciani, Aaron and Carl, Glenn and Conger, Lauren and Curro, Joseph and Edelman, Alan and Morales, Peter and Nielsen, Aaron P. and O'Keeffe, Michael F. and Rackauckas, Christopher V. and Taylor, Jonathan and Wollaber, Allan B.},
  archivePrefix={arXiv},
  doi={10.48550/arXiv.2007.12158},
  eprint={2007.12158},
  journal={arXiv},
  pages={1--12},
  url={https://doi.org/10.48550/arXiv.2007.12158},
  year={2023}
}
```
