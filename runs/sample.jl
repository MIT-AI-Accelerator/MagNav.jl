## analysis of SGL Flt1002
cd(@__DIR__)
using Pkg; Pkg.activate("../"); Pkg.update(); Pkg.instantiate()
using MagNav
using BenchmarkTools, Statistics
using Plots: plot, plot!

## get flight data
data_dir  = MagNav.data_dir()
data_file = string(data_dir,"/Flt1002_train.h5")
xyz_data  = get_flight_data(data_file);

## sensor locations (from front seat rail)
# Mag  1   tail stinger                      X=-12.01   Y= 0      Z=1.37
# Mag  2   front cabin just aft of cockpit   X= -0.60   Y=-0.36   Z=0
# Mag  3   mid cabin next to INS             X= -1.28   Y=-0.36   Z=0
# Mag  4   rear of cabin on floor            X= -3.53   Y= 0      Z=0
# Mag  5   rear of cabin on ceiling          X= -3.79   Y= 0      Z=1.20
# Flux B   tail at base of stinger           X= -8.92   Y= 0      Z=0.96
# Flux C   rear of cabin port side           X= -4.06   Y= 0.42   Z=0
# Flux D   rear of cabin starboard side      X= -4.06   Y=-0.42   Z=0

## create Tolles-Lawson coefficients
pass1 = 0.1  # first  passband frequency [Hz]
pass2 = 0.9  # second passband frequency [Hz]
fs    = 10.0 # sampling frequency [Hz]
i1    = findfirst(xyz_data.LINE .== 1002.02)
i2    = findlast( xyz_data.LINE .== 1002.02)

TL_coef_1 = create_TL_coef(xyz_data.FLUXB_X[i1:i2],
                           xyz_data.FLUXB_Y[i1:i2],
                           xyz_data.FLUXB_Z[i1:i2],
                           xyz_data.UNCOMPMAG1[i1:i2];
                           pass1=pass1,pass2=pass2,fs=fs);
TL_coef_2 = create_TL_coef(xyz_data.FLUXB_X[i1:i2],
                           xyz_data.FLUXB_Y[i1:i2],
                           xyz_data.FLUXB_Z[i1:i2],
                           xyz_data.UNCOMPMAG2[i1:i2];
                           pass1=pass1,pass2=pass2,fs=fs);
TL_coef_3 = create_TL_coef(xyz_data.FLUXB_X[i1:i2],
                           xyz_data.FLUXB_Y[i1:i2],
                           xyz_data.FLUXB_Z[i1:i2],
                           xyz_data.UNCOMPMAG3[i1:i2];
                           pass1=pass1,pass2=pass2,fs=fs);
TL_coef_4 = create_TL_coef(xyz_data.FLUXB_X[i1:i2],
                           xyz_data.FLUXB_Y[i1:i2],
                           xyz_data.FLUXB_Z[i1:i2],
                           xyz_data.UNCOMPMAG4[i1:i2];
                           pass1=pass1,pass2=pass2,fs=fs);
TL_coef_5 = create_TL_coef(xyz_data.FLUXB_X[i1:i2],
                           xyz_data.FLUXB_Y[i1:i2],
                           xyz_data.FLUXB_Z[i1:i2],
                           xyz_data.UNCOMPMAG5[i1:i2];
                           pass1=pass1,pass2=pass2,fs=fs);

## create Tolles-Lawson A matrix
A = create_TL_A(xyz_data.FLUXB_X,
                xyz_data.FLUXB_Y,
                xyz_data.FLUXB_Z);

## correct magnetometer measurements
mag_1_c = xyz_data.UNCOMPMAG1 - A*TL_coef_1 .+ mean(A*TL_coef_1);
mag_2_c = xyz_data.UNCOMPMAG2 - A*TL_coef_2 .+ mean(A*TL_coef_2);
mag_3_c = xyz_data.UNCOMPMAG3 - A*TL_coef_3 .+ mean(A*TL_coef_3);
mag_4_c = xyz_data.UNCOMPMAG4 - A*TL_coef_4 .+ mean(A*TL_coef_4);
mag_5_c = xyz_data.UNCOMPMAG5 - A*TL_coef_5 .+ mean(A*TL_coef_5);

## plot anomaly field comparison with SGL, Compensation 1
calcIGRF = xyz_data.DCMAG1 - xyz_data.IGRFMAG1; # IGRF offset
tt = xyz_data.TIME[i1:i2]; # time series
p1 = plot(tt,detrend(xyz_data.IGRFMAG1[i1:i2]),
          label="SGL Mag 1",color=:blue,lw=3, dpi=200,
          xlabel="Time [s]",ylabel="Magnetic Field [nT]",
          ylims=(-150,150))
plot!(p1,tt,detrend(mag_1_c[i1:i2]-xyz_data.DIURNAL[i1:i2]-calcIGRF[i1:i2]),
      label="Comp Mag 1",color=:red,lw=3,linestyle=:dash)
# plot!(p1,tt,detrend(mag_2_c[i1:i2]-xyz_data.DIURNAL[i1:i2]-calcIGRF[i1:i2]),
#       label="Comp Mag 2",color=:purple)
plot!(p1,tt,detrend(mag_3_c[i1:i2]-xyz_data.DIURNAL[i1:i2]-calcIGRF[i1:i2]),
      label="Comp Mag 3",color=:green)
plot!(p1,tt,detrend(mag_4_c[i1:i2]-xyz_data.DIURNAL[i1:i2]-calcIGRF[i1:i2]),
      label="Comp Mag 4",color=:black)
plot!(p1,tt,detrend(mag_5_c[i1:i2]-xyz_data.DIURNAL[i1:i2]-calcIGRF[i1:i2]),
      label="Comp Mag 5",color=:orange)
display(p1)
# savefig("comp_prof_1.png")

## gradient of parameter example
grad_flux_b_x = fdm(xyz_data.FLUXB_X);


## 
##* INSTRUCTIONS
# 0) Don't edit artifacts.toml file directly
# 1) Create tarball from folder containing data (tar czf folder.tar.gz folder)
# 2) Save tarball on Dropbox (or other blob storage)
# 3) Download tarball into same folder as this file
# 4) Set tarball_name, tarball_url & artifact_name below
# 5) Run the code shown below

#* SPECIFY INFORMATION HERE
tarball_name  = "flight_data_2020.tar.gz"
tarball_url   = "https://www.dropbox.com/s/ge5np08npvidi3r/flight_data_2020.tar.gz?dl=0"
artifact_name = "flight_data_2020"

##* RUN CODE BELOW
cd(@__DIR__)
using Pkg
using Pkg.Artifacts, Inflate, SHA, Tar

# generate hashes
artifact_toml = normpath(joinpath(@__DIR__,"../Artifacts.toml"))
hash_1   = Base.SHA1(Tar.tree_hash(IOBuffer(inflate_gzip(tarball_name))))
hash_256 = bytes2hex(open(sha256,tarball_name))

# bind artifact
bind_artifact!(artifact_toml,artifact_name,hash_1;
               download_info=[(tarball_url,hash_256)],lazy=true,force=true)

## todo: add other artifacts
using MagNav
data_dir  = MagNav.data_dir()
data_file = string(data_dir,"/FltXXXX.h5")
xyz = get_XYZ20(data_file);
