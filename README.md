# MagNav: airborne Magnetic anomaly Navigation

<p align="left">
    <a href="https://gitlab.com/gnadt/MagNav.jl/-/jobs">
        <img src="https://gitlab.com/gnadt/MagNav.jl/badges/master/pipeline.svg" title="gitlab">
    </a>
    <a href="https://app.codecov.io/gh/MIT-AI-Accelerator/MagNav.jl">
        <img src="https://codecov.io/gh/MIT-AI-Accelerator/MagNav.jl/branch/master/graph/badge.svg" title="codecov">
    </a>
</p>

<!-- https://gnadt.gitlab.io/MagNav.jl/ -->

<!-- - Bickel1979, Canciani2016, Canciani2017, Canciani2021, Gnadt2022, Gnadt2022a -->

<!-- MagNav.jl contains a full suite of tools for performing airborne Magnetic anomaly Navigation, including mapping, compensation, and navigation. Multiple tutorials are provided in the `runs` folder. -->

MagNav.jl contains tools for performing aeromagnetic compensation. A basic set of Julia starter code files are in the `src` folder and a sample run file is in the `runs` folder. This package was developed as part of the [USAF-MIT Artificial Intelligence Accelerator](https://aia.mit.edu/). More information on this effort is provided on the [challenge problem website](https://magnav.mit.edu/). Questions may be sent to [magnav-admins@mit.edu](mailto:magnav-admins@mit.edu).

## Data

Publically available flight data can be automatically downloaded within the package itself. This dataset can also be directly downloaded from [here](https://doi.org/10.5281/zenodo.4271804). See the [datasheet](https://github.com/MIT-AI-Accelerator/MagNav.jl/blob/master/challenge_problem_datasheet.pdf) for high-level information about this dataset. Details of the flight data are described in the readme files within the `readmes` folder.

## Data Sharing Agreement

Please read the full Data Sharing Agreement located [here](https://github.com/MIT-AI-Accelerator/MagNav.jl/blob/master/DATA_SHARING_AGREEMENT.md).

By granting You access to Data, the Air Force grants You a limited personal, non-exclusive, non-transferable, non-assignable, and revocable license to copy, modify, publicly display, and use the Data in accordance with this AGREEMENT solely for the purpose of non-profit research, non-profit education, or for government purposes by or on behalf of the U.S. Government. No license is granted for any other purpose, and there are no implied licenses in this Agreement. This Agreement is effective as of the date of approval by Air Force and remains in force for a period of one year from such date, unless terminated earlier or amended in writing. By using Data, You hereby grant an unlimited, irrevocable, world-wide, royalty-free right to the The United States Government to use for any purpose and in any manner whatsoever any feedback from You to the Air Force concerning Your use of Data.

## Citation

If this code or data is used in any work, please cite:

```
[DataSet Name] provided by the United States Air Force pursuant to Cooperative Agreement Number FA8750-19-2-1000 - [dates used]
@article{gnadt2020signal,
  title={Signal Enhancement for Magnetic Navigation Challenge Problem},
  author={Gnadt, Albert R and Belarge, Joseph and Canciani, Aaron and Conger, Lauren and Curro, Joseph and Edelman, Alan and Morales, Peter and O'Keeffe, Michael F and Taylor, Jonathan and Rackauckas, Christopher},
  journal={arXiv e-prints},
  pages={arXiv--2007},
  year={2020}
}
```
