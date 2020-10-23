# Signal Enhancement for Magnetic Navigation Challenge Problem

This is a repository for the signal enhancement for magnetic navigation (MagNav) challenge problem, which was introduced at [JuliaCon 2020](https://live.juliacon.org/talk/C9FGPP). The high-level goal is to use magnetometer (magnetic field) readings recorded from within a cockpit and remove the aircraft magnetic noise to yield a clean magnetic signal. A detailed description of the challenge problem can be found [here](https://arxiv.org/pdf/2007.12158.pdf) and additional MagNav literature can be found [here](https://github.com/MIT-AI-Accelerator/sciml-papers/tree/master/magnav).

|Round|Start|End|Winning Team|
|--|--|--|--|
|1|26-Jul-20|28-Aug-20|Ling-Wei Kong, Cheng-Zhen Wang, and Ying-Cheng Lai <br /> Arizona State University ([submission](https://github.com/lw-kong/MagNav))|
|2|24-Sep-20|4-Jan-21||

## Introduction Videos

- [Magnetic Navigation Overview](https://youtu.be/S3wKHDsHq8A)
- [Challenge Problem Description](https://youtu.be/qLKd1gwJhoA)
- [Challenge Problem Datasets](https://youtu.be/fyEt6XJRvvg)

## Starter Code

A basic set of starter Julia code files have been provided within the `src` folder. This code is largely based on work done by [Major Canciani](https://apps.dtic.mil/dtic/tr/fulltext/u2/1017870.pdf). This code has only been tested with Julia 1.4 and [1.5](https://julialang.org/downloads/). A sample run file is located within the `runs` folder, which includes downloading the flight data via artifact (`Artifacts.toml`). Details of the flight data are described in the readme files within the `readmes` folder. The flight data can also be directly downloaded from [here](https://www.dropbox.com/sh/dl/x37yr72x5a5nbz0/AADBt8ioU4Lm7JgEMQvPD7gxa/flight_data.tar.gz).

## Team Members

The MagNav team is part of the [USAF-MIT Artificial Intelligence Accelerator](https://aia.mit.edu/), a joint collaboration between the United States Air Force, MIT CSAIL, and MIT Lincoln Laboratory. Current team members include:

[MIT Julia Lab](https://julia.mit.edu/) within [MIT CSAIL](https://www.csail.mit.edu/)
- [Albert R. Gnadt](https://gnadt.github.io/) (AeroAstro Graduate Student)
- [Chris Rackauckas](https://chrisrackauckas.com/) (Applied Mathematics Instructor)
- [Alan Edelman](http://www-math.mit.edu/~edelman/) (Applied Mathematics Professor)

[MIT Lincoln Laboratory](https://www.ll.mit.edu/)
- Joseph Belarge (Group 46)
- Michael F. O'Keeffe (Group 89)
- Jonathan Taylor (Group 52)
- Michael Yee (Group 01)

[Air Force Institute of Technology](https://www.afit.edu/)
- Major Aaron Canciani
- Major Joseph Curro
- Aaron P. Nielsen ([DiDacTex, LLC](https://www.didactex.com))

[Air Force @ MIT](https://aia.mit.edu/about-us/)
- Major David Jacobs

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

## Data Sharing Agreement

AIR FORCE ACCEPTABLE USE AGREEMENT
(ver 20200211)
The following terms comprise the Acceptable Use Policy (AUP) and Data License Agreement (collectively, the "AGREEMENT") for all Air Force Data, herein after referred to as “Data”, made available to You. Certain Data may have additional Supplemental provisions. References to this AGREEMENT shall include any and all relevant Supplemental provisions. “You” or “Your” means an individual who is authorized to accept the terms of this AGREEMENT on behalf of themselves, or a group, an organization, or an institution for research, education, or government purposes.
Notwithstanding this AGREEMENT, the Air Force reserves the right, in its sole discretion, to refuse requests for access to Data and/or to revoke authority to use and access Data by anyone. If You feel Your request is inappropriately denied or access terminated, You may submit relevant information to the Air Force by sending a message to this GitHub Account. The additional information provided will be reviewed and a final decision will be issued by GitHub Account without other means for recourse.  
In consideration for requesting and receiving access to Data under Cooperative Agreement Number FA8750-19-2-1000, You acknowledge that You understand and agree to be bound by the terms and conditions of this AGREEMENT. Any violation of this AGREEMENT may result in the immediate suspension or termination of this AGREEMENT, termination of your access to  Data, and/or other actions authorized by law, such as injunctive or equitable relief. By entering into this AGREEMENT, You acknowledge that You are personally and individually liable and responsible for compliance with this AGREEMENT and are liable for any violations of this AGREEMENT, which is a legally binding contract between You and the United States Government. You may terminate this Agreement by contacting GitHub Account in writing and receiving return written acknowledgement of such request. All Data provided to You remain(s)  “Air Force Data” at all times, as defined by the Committee on National Security Systems (CNSS) Instruction Number 4009, 26 April 2010, and the Department of Defense Instruction (DoDI) 8320.02, Sharing Data, Information, and Technology (IT) Services in the Department of Defense, dated 5 August 2013, and this AGREEMENT does not convey any ownership right to You in any Data under United States copyright or other applicable law, nor any equitable or other claim of right or title of any kind whatsoever. 
1. LICENSE
By granting You access to Data, the Air Force grants You a limited personal, non-exclusive, non-transferable, non-assignable, and revocable license to copy, modify, publicly display, and use the Data in accordance with this AGREEMENT solely for the purpose of non-profit research, non-profit education, or for government purposes by or on behalf of the U.S. Government. No license is granted for any other purpose, and there are no implied licenses in this Agreement. This Agreement is effective as of the date of approval by Air Force and remains in force for a period of one year from such date, unless terminated earlier or amended in writing. By using Data, You hereby grant an unlimited, irrevocable, world-wide, royalty-free right to the The United States Government to use for any purpose and in any manner whatsoever any feedback from You to the Air Force concerning Your use of Data. 
2. GENERAL CONDITIONS
(i) The Data has been anonymized; however, if any portion(s) of such Data incidentally contains information that could amount to non-anonymized Data, You will respect the privacy of persons that may be identified in such Data. For any publication or other disclosure, You will anonymize or de-identify all personally-identifiable information, IP addresses, and other data identified in Supplemental provisions (if any) by using commonly accepted techniques, such as one of the methods recommended by CAIDA (https://www.caida.org/projects/predict/anonymization). You will notify the Air Force using the contact information provided at the end of this AGREEMENT immediately and provide all relevant regarding potential non-anonymized Data. 

(ii) While using Data, You agree not to circumvent any technological measure as prohibited and punishable under Title 17 United States Code § 1201, Circumvention of copyright protection systems, by reverse engineering, decrypting, de-anonymizing, deriving, or otherwise re-identifying any anonymized information. 

(iii) You will not distribute, disclose, transfer or otherwise make available Data to any person other than those employed by, or directed by, You who are assisting or collaborating with You using the Data Other entities with whom You are collaborating in research using the Air Force Data must request access to the Data separately and directly from the Air Force. 

(iv) You agree to expunge any and all copies of all Data upon completion or termination of the particular research, education, or government purpose for which You obtained access under this AGREEMENT, or promptly upon revocation of Your license for any reason by Air Force.  For purposes of this subsection, the date of completion of such particular research, education or government purpose  shall be deemed to include a reasonable period of time that You may need to retain Data  to satisfy scientific reproducibility, peer review obligations, or other use specifically authorized in writing by the Air Force. 

(v) You agree to safeguard all Data using at least the same degree of care that You use for Your own data and any other data of a like nature, but using no less than a reasonable degree of care, and You agree to protect the confidentiality of Data, including any de-anonymized portions of such Data, and/or the privacy of any identifiable person and to prevent its unauthorized disclosure and use. Data is confidential if it is marked as such, if by its nature or content is reasonably distinguishable as confidential, or if You have reasonable cause to believe that its disclosure to a third party would cause harm or damage to the interests to the United States Government or any other person with any legal or equitable interest in the protection of Data. 

(vi) You will notify Air Force using the contact information provided at the end of this AGREEMENT immediately and provide all relevant details concerning the occurrence of any of the following: (a) loss of Data by any means, (b) compromises of confidentiality, security or privacy of Data; or (c) You receive any legal, investigatory, or other government demand to reverse engineer, decrypt, de-anonymize, or otherwise disclose anonymized or confidential Data. 

(vii) You agree that all Data is provided to You “AS IS” without any Air Force representation as to suitability for intended use or warranty whatsoever.  This disclaimer does not affect any obligation the Air Force may have regarding Data specified in a contract other than this AGREEMENT for the performance of such other contract.
(viii) You are free to publish (including web pages, papers published by a third party, and publicly available presentations) or make public or use results derived from this Data.  Results derived from data means any conclusions, observations, theories, analysis or opinions derived in whole, or in-part, from the Data.  You may publish Data only to the extent necessary to discuss the results derived from this Data. You agree that by publishing results derived from this Data, or making public results derived from this Data, or using by any means data from this Data, You will provide Air Force with a copy of (or a link to) the publication and You must cite the Data as follows: 
"[DataSet Name] provided by the United States Air Force pursuant to Cooperative Agreement Number FA8750-19-2-1000 - [dates used]
@article{gnadt2020signal,
  title={Signal Enhancement for Magnetic Navigation Challenge Problem},
  author={Gnadt, Albert R and Belarge, Joseph and Canciani, Aaron and Conger, Lauren and Curro, Joseph and Edelman, Alan and Morales, Peter and O'Keeffe, Michael F and Taylor, Jonathan and Rackauckas, Christopher},
  journal={arXiv e-prints},
  pages={arXiv--2007},
  year={2020}"
