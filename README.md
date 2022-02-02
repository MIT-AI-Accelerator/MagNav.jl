# Signal Enhancement for Magnetic Navigation Challenge Problem

<p align="left">
    <a href="https://gitlab.com/gnadt/MagNav-jl/-/jobs">
        <img src="https://gitlab.com/gnadt/MagNav-jl/badges/master/pipeline.svg" title="gitlab">
    </a>
<!--     <a href="https://codecov.io/gh/MIT-AI-Accelerator/MagNav.jl">
        <img src="https://codecov.io/gh/MIT-AI-Accelerator/MagNav.jl/branch/master/graph/badge.svg" title="codecov">
    </a> -->
</p>

This is a repository for the signal enhancement for magnetic navigation (MagNav) challenge problem, which was [introduced at JuliaCon 2020](https://youtu.be/QwVO0Xh2Hbg?t=7252). The high-level goal is to use magnetometer (magnetic field) readings recorded from within a cockpit and remove the aircraft magnetic noise to yield a clean magnetic signal. A detailed description of the challenge problem can be found [here](https://arxiv.org/pdf/2007.12158.pdf) and additional MagNav literature can be found [here](https://github.com/MIT-AI-Accelerator/sciml-papers/tree/master/magnav). Please email the challenge problem organizers at [magnav-challenge-organizers@mit.edu](mailto:magnav-challenge-organizers@mit.edu) with any questions.

|Round|Start|End|Winning Team|
|--|--|--|--|
|1|26-Jul-20|28-Aug-20|Ling-Wei Kong, Cheng-Zhen Wang, and Ying-Cheng Lai <br /> Arizona State University ([submission](https://github.com/lw-kong/MagNav))|
|2|TBD|TBD||

## Introduction Videos

- [Magnetic Navigation Overview](https://youtu.be/S3wKHDsHq8A)
- [Challenge Problem Description](https://youtu.be/qLKd1gwJhoA)
- [Challenge Problem Datasets](https://youtu.be/fyEt6XJRvvg)

## Starter Code and Data

A basic set of Julia starter code files are in the `src` folder. This code is largely based on work done by [Maj Canciani](https://apps.dtic.mil/dtic/tr/fulltext/u2/1017870.pdf). This code has only been tested using the latest version of [Julia](https://julialang.org/downloads/). A sample run file is in the `runs` folder, which includes downloading the flight data via artifact (see `Artifacts.toml`). Please see the [datasheet](https://github.com/MIT-AI-Accelerator/MagNav.jl/blob/master/challenge_problem_datasheet.pdf) for high-level information about this dataset. Details of the flight data are described in the readme files within the `readmes` folder. The flight data can also be directly downloaded from [here](https://doi.org/10.5281/zenodo.4271804).

**NOTE**: The `dt` field in each HDF5 file is incorrect. The correct value is 0.1.

## Data Sharing Agreement

Please read the full Data Sharing Agreement located [here](https://github.com/MIT-AI-Accelerator/MagNav.jl/blob/master/DATA_SHARING_AGREEMENT.md).

By granting You access to Data, the Air Force grants You a limited personal, non-exclusive, non-transferable, non-assignable, and revocable license to copy, modify, publicly display, and use the Data in accordance with this AGREEMENT solely for the purpose of non-profit research, non-profit education, or for government purposes by or on behalf of the U.S. Government. No license is granted for any other purpose, and there are no implied licenses in this Agreement. This Agreement is effective as of the date of approval by Air Force and remains in force for a period of one year from such date, unless terminated earlier or amended in writing. By using Data, You hereby grant an unlimited, irrevocable, world-wide, royalty-free right to the The United States Government to use for any purpose and in any manner whatsoever any feedback from You to the Air Force concerning Your use of Data.

## Team Members

The MagNav team is part of the [USAF-MIT Artificial Intelligence Accelerator](https://aia.mit.edu/), a joint collaboration between the United States Air Force, MIT CSAIL, and MIT Lincoln Laboratory. Current team members include:

[MIT Julia Lab](https://julia.mit.edu/) within [MIT CSAIL](https://www.csail.mit.edu/)
- [Albert R. Gnadt](https://gnadt.github.io/) (AeroAstro Graduate Student)
- [Chris V. Rackauckas](https://chrisrackauckas.com/) (Applied Mathematics Instructor)
- [Alan S. Edelman](http://www-math.mit.edu/~edelman/) (Applied Mathematics Professor)

[MIT Lincoln Laboratory](https://www.ll.mit.edu/)
- Jonathan A. Taylor (Group 24)
- Glenn M. Carl (Group 89)
- Allan B. Wollaber (Group 01)

[Air Force Institute of Technology](https://www.afit.edu/)
- Aaron P. Nielsen ([DiDacTex, LLC](https://www.didactex.com/))
- Maj Joseph A. Curro
- Maj Aaron J. Canciani ([NRO](https://www.nro.gov/))

[Air Force @ MIT](https://aia.mit.edu/about-us/)
- TSgt Chasen Milner
- Maj Kyle McAlpin

## Citation

If this dataset is used in any citation, please cite the following work:

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
