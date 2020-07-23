# analysis of SGL Flt1002
cd(@__DIR__)
using Pkg; Pkg.activate("../"); Pkg.instantiate()
using MagNav
using Plots
gr()

data_dir  = MagNav.data_dir()
data_file = string(data_dir,"/Flt1002-train.h5")
xyz_data  = get_flight_data(data_file)

# sensor locations (from front seat rail)
# Mag  1   tail stinger                      X=-12.01   Y= 0      Z=1.37
# Mag  2   front cabin just aft of cockpit   X= -0.60   Y=-0.36   Z=0
# Mag  3   mid cabin next to INS             X= -1.28   Y=-0.36   Z=0
# Mag  4   rear of cabin on floor            X= -3.53   Y= 0      Z=0
# Mag  5   rear of cabin on ceiling          X= -3.79   Y= 0      Z=1.20
# Flux B   tail at base of stinger           X= -8.92   Y= 0      Z=0.96
# Flux C   rear of cabin port side           X= -4.06   Y= 0.42   Z=0
# Flux D   rear of cabin starboard side      X= -4.06   Y=-0.42   Z=0

# create Tolles-Lawson coefficients
cp = Dict()
cp[:pass1] = 0.1  # first  passband frequency [Hz]
cp[:pass2] = 0.9  # second passband frequency [Hz]
cp[:fs]    = 10.0 # sampling frequency [Hz]
i1         = findfirst(xyz_data.tie_line .== 1002.02)
i2         = findlast( xyz_data.tie_line .== 1002.02)
TL_coef_1  = create_TL_coef(xyz_data.flux_b_x[i1:i2],
                            xyz_data.flux_b_y[i1:i2],
                            xyz_data.flux_b_z[i1:i2],
                            xyz_data.mag_1_uc[i1:i2];cp...)
TL_coef_2  = create_TL_coef(xyz_data.flux_b_x[i1:i2],
                            xyz_data.flux_b_y[i1:i2],
                            xyz_data.flux_b_z[i1:i2],
                            xyz_data.mag_2_uc[i1:i2];cp...)
TL_coef_3  = create_TL_coef(xyz_data.flux_b_x[i1:i2],
                            xyz_data.flux_b_y[i1:i2],
                            xyz_data.flux_b_z[i1:i2],
                            xyz_data.mag_3_uc[i1:i2];cp...)
TL_coef_4  = create_TL_coef(xyz_data.flux_b_x[i1:i2],
                            xyz_data.flux_b_y[i1:i2],
                            xyz_data.flux_b_z[i1:i2],
                            xyz_data.mag_4_uc[i1:i2];cp...)
TL_coef_5  = create_TL_coef(xyz_data.flux_b_x[i1:i2],
                            xyz_data.flux_b_y[i1:i2],
                            xyz_data.flux_b_z[i1:i2],
                            xyz_data.mag_5_uc[i1:i2];cp...)

# create Tolles-Lawson A matrix
A = create_TL_Amat(xyz_data.flux_b_x,
                   xyz_data.flux_b_y,
                   xyz_data.flux_b_z)

# correct magnetometer measurements
mag_1_c = xyz_data.mag_1_uc - A*TL_coef_1 .+ mean(A*TL_coef_1)
mag_2_c = xyz_data.mag_2_uc - A*TL_coef_2 .+ mean(A*TL_coef_2)
mag_3_c = xyz_data.mag_3_uc - A*TL_coef_3 .+ mean(A*TL_coef_3)
mag_4_c = xyz_data.mag_4_uc - A*TL_coef_4 .+ mean(A*TL_coef_4)
mag_5_c = xyz_data.mag_5_uc - A*TL_coef_5 .+ mean(A*TL_coef_5)

# plot total field comparison with SGL, Compensation 1
# plot(xyz_data.tt[i1:i2],xyz_data.mag_1_c[i1:i2])
# plot!(xyz_data.tt[i1:i2],mag_1_c[i1:i2])

# IGRF offset
calcIGRF = xyz_data.mag_1_dc - xyz_data.mag_1_igrf

# plot anomaly field comparison with SGL, Compensation 1
plot(xyz_data.tt[i1:i2],detrend(xyz_data.mag_1_igrf[i1:i2]),
                        label="SGL Mag 1",color=:blue,lw=3,dpi=500,
                        xlabel="Time [s]",ylabel="Magnetic Field [nT]",
                        ylims=(-150,150))
plot!(xyz_data.tt[i1:i2],detrend(mag_1_c[i1:i2]
                        -xyz_data.diurnal[i1:i2]-calcIGRF[i1:i2]),
                         label="Comp Mag 1",color=:red,lw=3,linestyle=:dash)
# plot!(xyz_data.tt[i1:i2],detrend(mag_2_c[i1:i2]
#                         -xyz_data.diurnal[i1:i2]-calcIGRF[i1:i2]),
#                          label="Mag 2",color=:purple)
plot!(xyz_data.tt[i1:i2],detrend(mag_3_c[i1:i2]
                        -xyz_data.diurnal[i1:i2]-calcIGRF[i1:i2]),
                         label="Comp Mag 3",color=:green)
plot!(xyz_data.tt[i1:i2],detrend(mag_4_c[i1:i2]
                        -xyz_data.diurnal[i1:i2]-calcIGRF[i1:i2]),
                         label="Comp Mag 4",color=:black)
plot!(xyz_data.tt[i1:i2],detrend(mag_5_c[i1:i2]
                        -xyz_data.diurnal[i1:i2]-calcIGRF[i1:i2]),
                         label="Comp Mag 5",color=:orange)

# savefig("comp_prof_1.png")
