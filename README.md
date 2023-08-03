# MagNav: airborne Magnetic anomaly Navigation

<p align="left">
    <a href="https://github.com/MIT-AI-Accelerator/MagNav.jl/actions/workflows/ci.yml">
        <img src="https://github.com/MIT-AI-Accelerator/MagNav.jl/workflows/CI/badge.svg" title="CI">
    </a>
    <a href="https://app.codecov.io/gh/MIT-AI-Accelerator/MagNav.jl">
        <img src="https://codecov.io/gh/MIT-AI-Accelerator/MagNav.jl/branch/master/graph/badge.svg" title="codecov">
    </a>
    <a href="https://mit-ai-accelerator.github.io/MagNav.jl/stable/">
        <img src="https://img.shields.io/badge/docs-stable-blue.svg" title="docs-stable">
    </a>
</p>

MagNav.jl contains a full suite of tools for airborne Magnetic anomaly Navigation (MagNav), including flight path & INS data import or simulation, mapping, aeromagnetic compensation, and navigation. Julia source code files are in the [`src`](src) folder and examples are in the [`runs`](runs) folder. This package was developed as part of the [DAF-MIT Artificial Intelligence Accelerator](https://aia.mit.edu/). More information on this effort, including a list of relevant publications, is provided on the [challenge problem website](https://magnav.mit.edu/). The package has been tested on the long-term support (LTS) and latest stable versions of Julia, which may be downloaded from [here](https://julialang.org/downloads/).

The package may be downloaded directly or installed using:

```julia
] add MagNav
```

NOTE: If any artifacts produce a hash mismatch error while downloading, simply navigate to the 
`.julia/artifacts` folder and manually set the appropriate artifact folder name, e.g., `bf360d29207d6468a8cf783269191bda2cf1f679` for the ottawa_area_maps artifact.

## Data

Publicly available flight data can be automatically downloaded within the package itself. This dataset can also be directly downloaded from [here](https://doi.org/10.5281/zenodo.4271803). See the [datasheet](readmes/datasheet_sgl_2020_train.pdf) for high-level information about this dataset. Details of the flight data are described in the readme files within the [`readmes`](readmes) folder.

## Data Sharing Agreement

Please read the full Data Sharing Agreement located [here](readmes/DATA_SHARING_AGREEMENT.md).

By granting You access to Data, the Air Force grants You a limited personal, non-exclusive, non-transferable, non-assignable, and revocable license to copy, modify, publicly display, and use the Data in accordance with this AGREEMENT solely for the purpose of non-profit research, non-profit education, or for government purposes by or on behalf of the U.S. Government. No license is granted for any other purpose, and there are no implied licenses in this Agreement. This Agreement is effective as of the date of approval by Air Force and remains in force for a period of one year from such date, unless terminated earlier or amended in writing. By using Data, You hereby grant an unlimited, irrevocable, world-wide, royalty-free right to The United States Government to use for any purpose and in any manner whatsoever any feedback from You to the Air Force concerning Your use of Data.

## Citation

If this code or data is used in any work, please cite:

```
[DataSet Name] provided by the United States Air Force pursuant to Cooperative Agreement Number FA8750-19-2-1000 - [dates used]
@article{gnadt2020signal,
  title = {Signal Enhancement for Magnetic Navigation Challenge Problem},
  author = {Gnadt, Albert R. and Belarge, Joseph and Canciani, Aaron and Carl, Glenn and Conger, Lauren and Curro, Joseph and Edelman, Alan and Morales, Peter and Nielsen, Aaron P. and O'Keeffe, Michael F. and Rackauckas, Christopher V. and Taylor, Jonathan and Wollaber, Allan B.},
  journal = {arXiv e-prints},
  pages = {arXiv--2007.12158},
  doi = {10.48550/arXiv.2007.12158},
  year = {2020}
}
```
