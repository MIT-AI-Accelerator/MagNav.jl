# Signal Enhancement for Magnetic Navigation Challenge Problem

This is a repository for the signal enhancement for magnetic navigation (MagNav) challenge problem, which is being introduced at [JuliaCon 2020](https://juliacon.org/2020) and will run from July 26 to August 28. The high-level goal is to take magnetometer (magnetic field) readings from within the cockpit and remove the aircraft magnetic noise to yield a clean magnetic signal. A detailed description of the challenge problem can be found [here](https://arxiv.org/pdf/2007.12158.pdf).

## Introduction Videos

- [Signal Enhancement for Magnetic Navigation Scientific Machine Learning Challenge Problem](https://youtu.be/S3wKHDsHq8A)
- [Signal Enhancement for Magnetic Navigation Challenge Problem Detailed Description](https://youtu.be/qLKd1gwJhoA)
- [Signal Enhancement for Magnetic Navigation Challenge Problem Data Description](https://youtu.be/fyEt6XJRvvg)

## Starter Code

A basic set of starter Julia code files have been provided within the `src` folder. This code is largely based on work done by [Major Canciani](https://apps.dtic.mil/dtic/tr/fulltext/u2/1017870.pdf). This code has only been tested with [Julia 1.4](https://julialang.org/downloads/). A sample run file is located within the `runs` folder, which includes downloading the flight data via artifact (`Artifacts.toml`). Details of the flight data are described in the readme files within the `readmes` folder. The flight data can also be directly downloaded from [here](https://www.dropbox.com/sh/dl/x37yr72x5a5nbz0/AADBt8ioU4Lm7JgEMQvPD7gxa/flight_data.tar.gz).

## Team Members

The MagNav team is part of the [MIT-Air Force Artificial Intelligence Accelerator](https://ai-accelerator.csail.mit.edu/), a joint
collaboration between MIT CSAIL, MIT Lincoln Laboratory, and the US Air Force. Team members include:

[MIT Julia Lab](https://julia.mit.edu/) within [MIT CSAIL](https://www.csail.mit.edu/)
- Albert R. Gnadt (AeroAstro Graduate Student)
- Chris Rackauckas (Applied Mathematics Instructor)
- Alan Edelman (Applied Mathematics Professor)

[MIT Lincoln Laboratory](https://www.ll.mit.edu/)
- Joseph Belarge (Group 46)
- Lauren Conger (Group 46)
- Peter Morales (Group 01)
- Michael F. O'Keeffe (Group 89)
- Jonathan Taylor (Group 52)

[Air Force Institute of Technology](https://www.afit.edu/)
- Major Aaron Canciani
- Major Joseph Curro

Air Force @ MIT
- Major David Jacobs

## Citation

If this dataset is used in any citation, please cite the following work:

```
@article{gnadt2020signal,
  title={Signal Enhancement for Magnetic Navigation Challenge Problem},
  author={Gnadt, Albert R and Belarge, Joseph and Canciani, Aaron and Conger, Lauren and Curro, Joseph and Edelman, Alan and Morales, Peter and O'Keeffe, Michael F and Taylor, Jonathan and Rackauckas, Christopher},
  journal={arXiv e-prints},
  pages={arXiv--2007},
  year={2020}
}
```

## Data Sharing Agreement

By granting You access to Data, the Air Force grants You a limited personal, non-exclusive, non-transferable, non-assignable, and revocable license to copy, modify, publicly display, and use the Data in accordance with this AGREEMENT solely for the purpose of non-profit research, non-profit education, or for government purposes by or on behalf of the U.S. Government. No license is granted for any other purpose, and there are no implied licenses in this Agreement. This Agreement is effective as of the date of approval by Air Force and remains in force for a period of one year from such date, unless terminated earlier or amended in writing. By using Data, You hereby grant an unlimited, irrevocable, world-wide, royalty-free right to the The United States Government to use for any purpose and in any manner whatsoever any feedback from You to the Air Force concerning Your use of Data.
