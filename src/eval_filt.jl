"""
    run_filt(traj::Traj, ins::INS, meas, itp_mapS, filt_type::Symbol = :ekf;
             P0             = create_P0(),
             Qd             = create_Qd(),
             R              = 1.0,
             num_part       = 1000,
             thresh         = 0.8,
             baro_tau       = 3600.0,
             acc_tau        = 3600.0,
             gyro_tau       = 3600.0,
             fogm_tau       = 600.0,
             date           = get_years(2020,185),
             core::Bool     = false,
             map_alt        = 0,
             x_nn           = nothing,
             m              = nothing,
             y_norms        = nothing,
             terms          = [:permanent,:induced,:eddy,:bias],
             flux::MagV     = MagV([0.0],[0.0],[0.0],[0.0]),
             x0_TL          = ones(eltype(P0),19),
             extract::Bool  = true,
             run_crlb::Bool = true)

Run navigation filter and optionally compute Cramér–Rao lower bound (CRLB).

**Arguments:**
- `traj`:      `Traj` trajectory struct
- `ins`:       `INS` inertial navigation system struct
- `meas`:      scalar magnetometer measurement [nT]
- `itp_mapS`:  scalar map interpolation function (`f(lat,lon)` or `f(lat,lon,alt)`)
- `filt_type`: (optional) filter type {`:ekf`,`:mpf`}
- `P0`:        (optional) initial covariance matrix
- `Qd`:        (optional) discrete time process/system noise matrix
- `R`:         (optional) measurement (white) noise variance
- `num_part`:  (optional) number of particles, only used for `filt_type = :mpf`
- `thresh`:    (optional) resampling threshold fraction {0:1}, only used for `filt_type = :mpf`
- `baro_tau`:  (optional) barometer time constant [s]
- `acc_tau`:   (optional) accelerometer time constant [s]
- `gyro_tau`:  (optional) gyroscope time constant [s]
- `fogm_tau`:  (optional) FOGM catch-all time constant [s]
- `date`:      (optional) measurement date (decimal year) for IGRF [yr]
- `core`:      (optional) if true, include core magnetic field in measurement
- `map_alt`:   (optional) map altitude [m]
- `x_nn`:      (optional) `N` x `Nf` data matrix for neural network (`Nf` is number of features)
- `m`:         (optional) neural network model
- `y_norms`:   (optional) tuple of `y` normalizations, i.e., `(y_bias,y_scale)`
- `terms`:     (optional) Tolles-Lawson terms to use {`:permanent`,`:induced`,`:eddy`,`:bias`}
- `flux`:      (optional) `MagV` vector magnetometer measurement struct
- `x0_TL`:     (optional) initial Tolles-Lawson coefficient states
- `extract`:   (optional) if true, extract output structs
- `run_crlb`:  (optional) if true, compute the Cramér–Rao lower bound (CRLB)

**Returns:**
- if `extract = true`  & `run_crlb = true`
    - `crlb_out`: `CRLBout` Cramér–Rao lower bound extracted output struct
    - `ins_out`:  `INSout`  inertial navigation system extracted output struct
    - `filt_out`: `FILTout` filter extracted output struct
- if `extract = true`  & `run_crlb = false`
    - `filt_out`: `FILTout` filter extracted output struct
- if `extract = false` & `run_crlb = true`
    - `filt_res`: `FILTres` filter results struct
    - `crlb_P`:   Cramér–Rao lower bound non-linear covariance matrix
- if `extract = false` & `run_crlb = false`
    - `filt_res`: `FILTres` filter results struct
"""
function run_filt(traj::Traj, ins::INS, meas, itp_mapS, filt_type::Symbol = :ekf;
                  P0             = create_P0(),
                  Qd             = create_Qd(),
                  R              = 1.0,
                  num_part       = 1000,
                  thresh         = 0.8,
                  baro_tau       = 3600.0,
                  acc_tau        = 3600.0,
                  gyro_tau       = 3600.0,
                  fogm_tau       = 600.0,
                  date           = get_years(2020,185),
                  core::Bool     = false,
                  map_alt        = 0,
                  x_nn           = nothing,
                  m              = nothing,
                  y_norms        = nothing,
                  terms          = [:permanent,:induced,:eddy,:bias],
                  flux::MagV     = MagV([0.0],[0.0],[0.0],[0.0]),
                  x0_TL          = ones(eltype(P0),19),
                  extract::Bool  = true,
                  run_crlb::Bool = true)

    if filt_type == :ekf
        filt_res = ekf(ins,meas,itp_mapS;
                       P0       = P0,
                       Qd       = Qd,
                       R        = R,
                       baro_tau = baro_tau,
                       acc_tau  = acc_tau,
                       gyro_tau = gyro_tau,
                       fogm_tau = fogm_tau,
                       date     = date,
                       core     = core,
                       map_alt  = map_alt);
    elseif filt_type == :ekf_online
        filt_res = ekf_online(ins,meas,flux,itp_mapS,x0_TL,P0,Qd,R;
                              baro_tau = baro_tau,
                              acc_tau  = acc_tau,
                              gyro_tau = gyro_tau,
                              fogm_tau = fogm_tau,
                              date     = date,
                              core     = core,
                              terms    = terms);
    elseif filt_type == :ekf_online_nn
        filt_res = ekf_online_nn(ins,meas,itp_mapS,x_nn,m,y_norms,P0,Qd,R;
                                 baro_tau = baro_tau,
                                 acc_tau  = acc_tau,
                                 gyro_tau = gyro_tau,
                                 fogm_tau = fogm_tau,
                                 date     = date,
                                 core     = core);
    elseif filt_type == :mpf
        filt_res = mpf(ins,meas,itp_mapS;
                       P0       = P0,
                       Qd       = Qd,
                       R        = R,
                       num_part = num_part,
                       thresh   = thresh,
                       baro_tau = baro_tau,
                       acc_tau  = acc_tau,
                       gyro_tau = gyro_tau,
                       fogm_tau = fogm_tau,
                       date     = date,
                       core     = core);
    elseif filt_type == :nekf
        filt_res = nekf(ins,meas,itp_mapS,x_nn,m;
                        P0       = P0,
                        Qd       = Qd,
                        R        = R,
                        baro_tau = baro_tau,
                        acc_tau  = acc_tau,
                        gyro_tau = gyro_tau,
                        fogm_tau = fogm_tau,
                        date     = date,
                        core     = core);
    else
        error("filt_type $filt_type not defined")
    end

    if run_crlb
        crlb_P = crlb(traj,itp_mapS;
                      P0       = P0,
                      Qd       = Qd,
                      R        = R,
                      baro_tau = baro_tau,
                      acc_tau  = acc_tau,
                      gyro_tau = gyro_tau,
                      fogm_tau = fogm_tau,
                      date     = date,
                      core     = core)
        if extract
            return eval_results(traj,ins,filt_res,crlb_P)
        else
            return (filt_res, crlb_P)
        end
    else
        if extract
            return eval_filt(traj,ins,filt_res)
        else
            return (filt_res)
        end
    end
end # function run_filt

"""
    run_filt(traj::Traj, ins::INS, meas, itp_mapS,
             filt_type::Vector{Symbol}; ...)

Run multiple filter models and print results (nothing returned).

**Arguments:**
- `filt_type`: multiple filter types, e.g., [`:ekf`,`:ekf_online_nn`]
"""
function run_filt(traj::Traj, ins::INS, meas, itp_mapS,
                  filt_type::Vector{Symbol};
                  P0         = create_P0(),
                  Qd         = create_Qd(),
                  R          = 1.0,
                  num_part   = 1000,
                  thresh     = 0.8,
                  baro_tau   = 3600.0,
                  acc_tau    = 3600.0,
                  gyro_tau   = 3600.0,
                  fogm_tau   = 600.0,
                  date       = get_years(2020,185),
                  core::Bool = false,
                  map_alt    = 0,
                  x_nn       = nothing,
                  m          = nothing,
                  y_norms    = nothing,
                  terms      = [:permanent,:induced,:eddy,:bias],
                  flux::MagV = MagV([0.0],[0.0],[0.0],[0.0]),
                  x0_TL      = ones(eltype(P0),19))

    for i in eachindex(filt_type)
        @info("running $(filt_type[i]) filter")
        run_filt(traj,ins,meas,itp_mapS,filt_type[i];
                 P0          = P0,
                 Qd          = Qd,
                 R           = R,
                 num_part    = num_part,
                 thresh      = thresh,
                 baro_tau    = baro_tau,
                 acc_tau     = acc_tau,
                 gyro_tau    = gyro_tau,
                 fogm_tau    = fogm_tau,
                 date        = date,
                 core        = core,
                 map_alt     = map_alt,
                 x_nn        = x_nn,
                 m           = m,
                 y_norms     = y_norms,
                 terms       = terms,
                 flux        = flux,
                 x0_TL       = x0_TL,
                 extract     = true,
                 run_crlb    = false)
    end

end # function run_filt

"""
    eval_results(traj::Traj, ins::INS, filt_res::FILTres, crlb_P::Array)

Extract CRLB, INS, & filter results.

**Arguments:**
- `traj`:     `Traj` trajectory struct
- `ins`:      `INS` inertial navigation system struct
- `filt_res`: `FILTres` filter results struct
- `crlb_P`:   Cramér–Rao lower bound non-linear covariance matrix

**Returns:**
- `crlb_out`: `CRLBout` Cramér–Rao lower bound extracted output struct
- `ins_out`:  `INSout`  inertial navigation system extracted output struct
- `filt_out`: `FILTout` filter extracted output struct
"""
function eval_results(traj::Traj, ins::INS, filt_res::FILTres, crlb_P::Array)
    return (eval_crlb(traj,crlb_P),
            eval_ins( traj,ins),
            eval_filt(traj,ins,filt_res))
end # function eval_results

"""
    eval_crlb(traj::Traj, crlb_P::Array)

Extract Cramér–Rao lower bound (CRLB) results.

**Arguments:**
- `traj`:   `Traj` trajectory struct
- `crlb_P`: Cramér–Rao lower bound non-linear covariance matrix

**Returns:**
- `crlb_out`: `CRLBout` Cramér–Rao lower bound extracted output struct
"""
function eval_crlb(traj::Traj, crlb_P::Array)

    N = traj.N
    N_fields = length(fieldnames(CRLBout))
    crlb_out = CRLBout((zeros(Float64,N) for _ = 1:N_fields)...)

    crlb_out.lat_std  .= sqrt.(crlb_P[1,1,:])
    crlb_out.lon_std  .= sqrt.(crlb_P[2,2,:])
    crlb_out.alt_std  .= sqrt.(crlb_P[3,3,:])
    crlb_out.vn_std   .= sqrt.(crlb_P[4,4,:])
    crlb_out.ve_std   .= sqrt.(crlb_P[5,5,:])
    crlb_out.vd_std   .= sqrt.(crlb_P[6,6,:])
    crlb_out.tn_std   .= sqrt.(crlb_P[7,7,:])
    crlb_out.te_std   .= sqrt.(crlb_P[8,8,:])
    crlb_out.td_std   .= sqrt.(crlb_P[9,9,:])
    crlb_out.fogm_std .= sqrt.(crlb_P[18,18,:])

    crlb_out.n_std    .= dlat2dn.(crlb_out.lat_std,traj.lat)
    crlb_out.e_std    .= dlon2de.(crlb_out.lon_std,traj.lat)

    crlb_DRMS = round(Int,sqrt(mean(crlb_out.n_std.^2+crlb_out.e_std.^2)))
    @info("CRLB DRMS error = $crlb_DRMS m")

    return (crlb_out)
end # function eval_crlb

"""
    eval_ins(traj::Traj, ins::INS)

Extract INS results.

**Arguments:**
- `traj`: `Traj` trajectory struct
- `ins`:  `INS` inertial navigation system struct

**Returns:**
- `ins_out`: `INSout` inertial navigation system extracted output struct
"""
function eval_ins(traj::Traj, ins::INS)

    # not doing vn,ve,vd,tn,te,td,ax,ay,az,gx,gy,gz

    N = traj.N
    N_fields = length(fieldnames(INSout))
    ins_out  = INSout((zeros(Float64,N) for _ = 1:N_fields)...)

    if !iszero(ins.P)
        ins_out.lat_std .= sqrt.(ins.P[1,1,:])
        ins_out.lon_std .= sqrt.(ins.P[2,2,:])
        ins_out.alt_std .= sqrt.(ins.P[3,3,:])
        ins_out.n_std   .= dlat2dn.(ins_out.lat_std,ins.lat)
        ins_out.e_std   .= dlon2de.(ins_out.lon_std,ins.lat)
    end

    ins_out.lat_err .= ins.lat - traj.lat
    ins_out.lon_err .= ins.lon - traj.lon
    ins_out.alt_err .= ins.alt - traj.alt
    ins_out.n_err   .= dlat2dn.(ins_out.lat_err,ins.lat)
    ins_out.e_err   .= dlon2de.(ins_out.lon_err,ins.lat)

    ins_DRMS = round(Int,sqrt(mean(ins_out.n_err.^2+ins_out.e_err.^2)))
    @info("INS  DRMS error = $ins_DRMS m")

    return (ins_out)
end # function eval_ins

"""
    eval_filt(traj::Traj, ins::INS, filt_res::FILTres)

Extract filter results.

**Arguments:**
- `traj`:     `Traj` trajectory struct
- `ins`:      `INS` inertial navigation system struct
- `filt_res`: `FILTres` filter results struct

**Returns:**
- `filt_out`: `FILTout` filter extracted output struct
"""
function eval_filt(traj::Traj, ins::INS, filt_res::FILTres)

    N  = traj.N
    dt = traj.dt
    N_fields = length(fieldnames(FILTout))
    filt_out = FILTout(N,dt,(zeros(Float64,N) for _ = 1:N_fields-2)...)

    filt_out.tt   .= traj.tt
    filt_out.lat  .= ins.lat + filt_res.x[1,:]
    filt_out.lon  .= ins.lon + filt_res.x[2,:]
    filt_out.alt  .= ins.alt + filt_res.x[3,:]
    filt_out.vn   .= ins.vn  + filt_res.x[4,:]
    filt_out.ve   .= ins.ve  + filt_res.x[5,:]
    filt_out.vd   .= ins.vd  + filt_res.x[6,:]
    filt_out.tn   .=           filt_res.x[7,:]
    filt_out.te   .=           filt_res.x[8,:]
    filt_out.td   .=           filt_res.x[9,:]
    filt_out.ha   .=           filt_res.x[10,:]
    filt_out.ah   .=           filt_res.x[11,:]
    filt_out.ax   .=           filt_res.x[12,:]
    filt_out.ay   .=           filt_res.x[13,:]
    filt_out.az   .=           filt_res.x[14,:]
    filt_out.gx   .=           filt_res.x[15,:]
    filt_out.gy   .=           filt_res.x[16,:]
    filt_out.gz   .=           filt_res.x[17,:]
    filt_out.fogm .=           filt_res.x[end,:]

    filt_out.lat_std  .= sqrt.(filt_res.P[1,1,:])
    filt_out.lon_std  .= sqrt.(filt_res.P[2,2,:])
    filt_out.alt_std  .= sqrt.(filt_res.P[3,3,:])
    filt_out.vn_std   .= sqrt.(filt_res.P[4,4,:])
    filt_out.ve_std   .= sqrt.(filt_res.P[5,5,:])
    filt_out.vd_std   .= sqrt.(filt_res.P[6,6,:])
    filt_out.tn_std   .= sqrt.(filt_res.P[7,7,:])
    filt_out.te_std   .= sqrt.(filt_res.P[8,8,:])
    filt_out.td_std   .= sqrt.(filt_res.P[9,9,:])
    filt_out.ha_std   .= sqrt.(filt_res.P[10,10,:])
    filt_out.ah_std   .= sqrt.(filt_res.P[11,11,:])
    filt_out.ax_std   .= sqrt.(filt_res.P[12,12,:])
    filt_out.ay_std   .= sqrt.(filt_res.P[13,13,:])
    filt_out.az_std   .= sqrt.(filt_res.P[14,14,:])
    filt_out.gx_std   .= sqrt.(filt_res.P[15,15,:])
    filt_out.gy_std   .= sqrt.(filt_res.P[16,16,:])
    filt_out.gz_std   .= sqrt.(filt_res.P[17,17,:])
    filt_out.fogm_std .= sqrt.(filt_res.P[18,18,:])

    filt_out.n_std   .= dlat2dn.(filt_out.lat_std,filt_out.lat)
    filt_out.e_std   .= dlon2de.(filt_out.lon_std,filt_out.lat)

    filt_out.lat_err .= filt_out.lat - traj.lat
    filt_out.lon_err .= filt_out.lon - traj.lon
    filt_out.alt_err .= filt_out.alt - traj.alt
    filt_out.vn_err  .= filt_out.vn  - traj.vn
    filt_out.ve_err  .= filt_out.ve  - traj.ve
    filt_out.vd_err  .= filt_out.vd  - traj.vd

    n_tilt = zeros(eltype(filt_out.lat),N)
    e_tilt = zeros(eltype(filt_out.lat),N)
    d_tilt = zeros(eltype(filt_out.lat),N)

    for k = 1:N
        tilt_temp = ins.Cnb[:,:,k]*traj.Cnb[:,:,k]'
        n_tilt[k] = tilt_temp[3,2] # ≈ -tilt_temp[2,3]
        e_tilt[k] = tilt_temp[1,3] # ≈ -tilt_temp[3,1]
        d_tilt[k] = tilt_temp[2,1] # ≈ -tilt_temp[1,2]
    end

    filt_out.tn_err .= filt_out.tn - n_tilt
    filt_out.te_err .= filt_out.te - e_tilt
    filt_out.td_err .= filt_out.td - d_tilt

    filt_out.n_err  .= dlat2dn.(filt_out.lat_err,filt_out.lat)
    filt_out.e_err  .= dlon2de.(filt_out.lon_err,filt_out.lat)

    filt_DRMS = round(Int,sqrt(mean(filt_out.n_err.^2+filt_out.e_err.^2)))
    @info("FILT DRMS error = $filt_DRMS m")

    return (filt_out)
end # function eval_filt

"""
    plot_filt!(p1::Plot, traj::Traj, ins::INS, filt_out::FILTout;
               dpi::Int        = 200,
               Nmax::Int       = 5000,
               show_plot::Bool = true,
               save_plot::Bool = false)

Plot flights paths on an existing plot.

**Arguments:**
- `p1`:        plot (i.e., map)
- `traj`:      `Traj` trajectory struct
- `ins`:       `INS` inertial navigation system struct
- `filt_out`:  `FILTout` filter extracted output struct
- `dpi`:       (optional) dots per inch (image resolution)
- `Nmax`:      (optional) maximum number of data points plotted
- `show_plot`: (optional) if true, show plots
- `save_plot`: (optional) if true, save plots with default file names

**Returns:**
- `nothing`: flight paths are plotted on `p1`
"""
function plot_filt!(p1::Plot, traj::Traj, ins::INS, filt_out::FILTout;
                    dpi::Int        = 200,
                    Nmax::Int       = 5000,
                    show_plot::Bool = true,
                    save_plot::Bool = false)

    i    = downsample(1:traj.N,Nmax)
    lon  = rad2deg.([traj.lon;ins.lon;filt_out.lon])
    lat  = rad2deg.([traj.lat;ins.lat;filt_out.lat])
    xlim = get_lim(lon,0.05)
    ylim = get_lim(lat,0.05)

    plot!(p1,xlab="longitude [deg]",ylab="latitude [deg]",dpi=dpi)
    plot!(p1,rad2deg.(traj.lon[i])    ,rad2deg.(traj.lat[i])    ,lab="GPS")
    plot!(p1,rad2deg.(ins.lon[i])     ,rad2deg.(ins.lat[i])     ,lab="INS")
    plot!(p1,rad2deg.(filt_out.lon[i]),rad2deg.(filt_out.lat[i]),lab="MagNav")
    plot!(p1,xlim=xlim,ylim=ylim)
    show_plot && display(p1)
    save_plot && png(p1,"flight_path.png")

    return (nothing)
end # function plot_filt!

"""
    plot_filt(p1::Plot, traj::Traj, ins::INS, filt_out::FILTout;
              dpi::Int        = 200,
              Nmax::Int       = 5000,
              plot_vel::Bool  = false,
              show_plot::Bool = true,
              save_plot::Bool = false)

Plot flights paths on an existing plot and latitudes & longitudes vs time.

**Arguments:**
- `p1`:        plot (i.e., map)
- `traj`:      `Traj` trajectory struct
- `ins`:       `INS` inertial navigation system struct
- `filt_out`:  `FILTout` filter extracted output struct
- `dpi`:       (optional) dots per inch (image resolution)
- `Nmax`:      (optional) maximum number of data points plotted
- `plot_vel`:  (optional) if true, plot velocities
- `show_plot`: (optional) if true, show plots
- `save_plot`: (optional) if true, save plots with default file names

**Returns:**
- `p2`: `p1` with flight paths
- `p3`: latitudes  vs time
- `p4`: longitudes vs time
- `p5`: if `plot_vel = true`, north velocities vs time
- `p6`: if `plot_vel = true`, east  velocities vs time
"""
function plot_filt(p1::Plot, traj::Traj, ins::INS, filt_out::FILTout;
                   dpi::Int        = 200,
                   Nmax::Int       = 5000,
                   plot_vel::Bool  = false,
                   show_plot::Bool = true,
                   save_plot::Bool = false)

    p2 = deepcopy(p1)
    plot_filt!(p2,traj,ins,filt_out;
               dpi       = dpi,
               Nmax      = Nmax,
               show_plot = show_plot,
               save_plot = save_plot)

    i  = downsample(1:traj.N,Nmax)
    tt = (traj.tt[i] .- traj.tt[i][1]) / 60

    p3 = plot(xlab="time [min]",ylab="latitude [deg]",dpi=dpi)
    plot!(p3,tt,rad2deg.(traj.lat[i])    ,lab="GPS")
    plot!(p3,tt,rad2deg.(ins.lat[i])     ,lab="INS")
    plot!(p3,tt,rad2deg.(filt_out.lat[i]),lab="MagNav")
    show_plot && display(p3)
    save_plot && png(p3,"latitude.png")

    p4 = plot(xlab="time [min]",ylab="longitude [deg]",dpi=dpi)
    plot!(p4,tt,rad2deg.(traj.lon[i])    ,lab="GPS")
    plot!(p4,tt,rad2deg.(ins.lon[i])     ,lab="INS")
    plot!(p4,tt,rad2deg.(filt_out.lon[i]),lab="MagNav")
    show_plot && display(p4)
    save_plot && png(p4,"longitude.png")

    if plot_vel
        p5 = plot(xlab="time [min]",ylab="north velocity [m/s]",dpi=dpi)
        plot!(p5,tt,traj.vn[i]    ,lab="GPS")
        plot!(p5,tt,ins.vn[i]     ,lab="INS")
        plot!(p5,tt,filt_out.vn[i],lab="MagNav")
        show_plot && display(p5)
        save_plot && png(p5,"north_velocity.png")

        p6 = plot(xlab="time [min]",ylab="east velocity [m/s]",dpi=dpi)
        plot!(p6,tt,traj.ve[i]    ,lab="GPS")
        plot!(p6,tt,ins.ve[i]     ,lab="INS")
        plot!(p6,tt,filt_out.ve[i],lab="MagNav")
        show_plot && display(p6)
        save_plot && png(p6,"east_velocity.png")
    end

    # p7 = plot(xlab="time [min]",ylab="altitude [m]",dpi=dpi)
    # plot!(p7,tt,traj.alt[i]    ,lab="GPS")
    # plot!(p7,tt,ins.alt[i]     ,lab="INS")
    # plot!(p7,tt,filt_out.alt[i],lab="MagNav")
    # show_plot && display(p7)
    # save_plot && png(p7,"altitude.png")

    # p8 = plot(xlab="time [min]",ylab="down velocity [m/s]",dpi=dpi)
    # plot!(p8,tt,traj.vd[i]    ,lab="GPS")
    # plot!(p8,tt,ins.vd[i]     ,lab="INS")
    # plot!(p8,tt,filt_out.vd[i],lab="MagNav")
    # show_plot && display(p8)
    # save_plot && png(p8,"down_velocity.png")

    # p9 = plot(xlab="time [min]",ylab="tilt [deg]",dpi=dpi)
    # plot!(p9,tt,rad2deg.(filt_out.tn[i]),lab="north")
    # plot!(p9,tt,rad2deg.(filt_out.te[i]),lab="east")
    # plot!(p9,tt,rad2deg.(filt_out.td[i]),lab="down")
    # show_plot && display(p9)
    # save_plot && png(p9,"tilt.png")

    # p10 = plot(xlab="time [min]",ylab="barometer aiding altitude [m]",dpi=dpi)
    # plot!(p10,tt,filt_out.ha[i])
    # show_plot && display(p10)
    # save_plot && png(p10,"barometer_aiding_altitude.png")

    # p11 = plot(xlab="time [min]",ylab="barometer aiding vertical accel [m/s^2]",dpi=dpi)
    # plot!(p11,tt,filt_out.ah[i])
    # show_plot && display(p11)
    # save_plot && png(p11,"barometer_aiding_vertical_accel.png")

    # p12 = plot(xlab="time [min]",ylab="accelerometer bias [m/s^2]",dpi=dpi)
    # plot!(p12,tt,filt_out.ax[i],lab="x")
    # plot!(p12,tt,filt_out.ay[i],lab="y")
    # plot!(p12,tt,filt_out.az[i],lab="z")
    # show_plot && display(p12)
    # save_plot && png(p12,"accelerometer_bias.png")

    # p13 = plot(xlab="time [min]",ylab="gyroscope bias [rad/s]",dpi=dpi)
    # plot!(p13,tt,filt_out.gx[i],lab="x")
    # plot!(p13,tt,filt_out.gy[i],lab="y")
    # plot!(p13,tt,filt_out.gz[i],lab="z")
    # show_plot && display(p13)
    # save_plot && png(p13,"gyroscope_bias.png")

    return plot_vel ? (p2, p3, p4, p5, p6) : (p2, p3, p4)
end # function plot_filt

"""
    plot_filt(traj::Traj, ins::INS, filt_out::FILTout;
              dpi::Int        = 200,
              Nmax::Int       = 5000,
              plot_vel::Bool  = false,
              show_plot::Bool = true,
              save_plot::Bool = false)

Plot flights paths and latitudes & longitudes vs time.

**Arguments:**
- `traj`:      `Traj` trajectory struct
- `ins`:       `INS` inertial navigation system struct
- `filt_out`:  `FILTout` filter extracted output struct
- `dpi`:       (optional) dots per inch (image resolution)
- `Nmax`:      (optional) maximum number of data points plotted
- `plot_vel`:  (optional) if true, plot velocities
- `show_plot`: (optional) if true, show plots
- `save_plot`: (optional) if true, save plots with default file names

**Returns:**
- `p1`: flight paths
- `p2`: latitudes  vs time
- `p3`: longitudes vs time
- `p4`: if `plot_vel = true`, north velocities vs time
- `p5`: if `plot_vel = true`, east  velocities vs time
"""
function plot_filt(traj::Traj, ins::INS, filt_out::FILTout;
                   dpi::Int        = 200,
                   Nmax::Int       = 5000,
                   plot_vel::Bool  = false,
                   show_plot::Bool = true,
                   save_plot::Bool = false)
    p1 = plot()
    plot_filt(p1,traj,ins,filt_out;
              dpi       = dpi,
              Nmax      = Nmax,
              plot_vel  = plot_vel,
              show_plot = show_plot,
              save_plot = save_plot)
end # function plot_filt

"""
    plot_filt_err(traj::Traj, filt_out::FILTout, crlb_out::CRLBout;
                  dpi::Int        = 200,
                  Nmax::Int       = 5000,
                  plot_vel::Bool  = false,
                  show_plot::Bool = true,
                  save_plot::Bool = false)

Plot northing & easting errors vs time.

**Arguments:**
- `traj`:      `Traj` trajectory struct
- `filt_out`:  `FILTout` filter extracted output struct
- `crlb_out`:  `CRLBout` Cramér–Rao lower bound extracted output struct
- `dpi`:       (optional) dots per inch (image resolution)
- `Nmax`:      (optional) maximum number of data points plotted
- `plot_vel`:  (optional) if true, plot velocity errors
- `show_plot`: (optional) if true, show plots
- `save_plot`: (optional) if true, save plots with default file names

**Returns:**
- `p1`: northing errors vs time
- `p2`: easting  errors vs time
- `p3`: if `plot_vel = true`, north velocity errors vs time
- `p4`: if `plot_vel = true`, east  velocity errors vs time
"""
function plot_filt_err(traj::Traj, filt_out::FILTout, crlb_out::CRLBout;
                       dpi::Int        = 200,
                       Nmax::Int       = 5000,
                       plot_vel::Bool  = false,
                       show_plot::Bool = true,
                       save_plot::Bool = false)

    i  = downsample(1:traj.N,Nmax)
    tt = (traj.tt[i] .- traj.tt[i][1]) / 60

    p1 = plot(xlab="time [min]",ylab="northing error [m]",dpi=dpi)
    plot!(p1,tt, filt_out.n_err[i],lab="filter error",lc=:black,lw=2)
    plot!(p1,tt, filt_out.n_std[i],lab="filter 1-σ",lc=:blue)
    plot!(p1,tt,-filt_out.n_std[i],lab=false,lc=:blue)
    plot!(p1,tt, crlb_out.n_std[i],lab="CRLB 1-σ",lc=:red,ls=:dash)
    plot!(p1,tt,-crlb_out.n_std[i],lab=false,lc=:red,ls=:dash)
    show_plot && display(p1)
    save_plot && png(p1,"northing_error.png")

    p2 = plot(xlab="time [min]",ylab="easting error [m]",dpi=dpi)
    plot!(p2,tt, filt_out.e_err[i],lab="filter error",lc=:black,lw=2)
    plot!(p2,tt, filt_out.e_std[i],lab="filter 1-σ",lc=:blue)
    plot!(p2,tt,-filt_out.e_std[i],lab=false,lc=:blue)
    plot!(p2,tt, crlb_out.e_std[i],lab="CRLB 1-σ",lc=:red,ls=:dash)
    plot!(p2,tt,-crlb_out.e_std[i],lab=false,lc=:red,ls=:dash)
    show_plot && display(p2)
    save_plot && png(p2,"easting_error.png")

    if plot_vel

        p3 = plot(xlab="time [min]",ylab="north velocity error [m/s]",dpi=dpi) # , ylim=(-10,10))
        plot!(p3,tt, filt_out.vn_err[i],lab="filter error",lc=:black,lw=2)
        plot!(p3,tt, filt_out.vn_std[i],lab="filter 1-σ",lc=:blue)
        plot!(p3,tt,-filt_out.vn_std[i],lab=false,lc=:blue)
        show_plot && display(p3)
        save_plot && png(p3,"north_velocity_error.png")

        p4 = plot(xlab="time [min]",ylab="east velocity error [m/s]",dpi=dpi) # , ylim=(-10,10))
        plot!(p4,tt, filt_out.ve_err[i],lab="filter error",lc=:black,lw=2)
        plot!(p4,tt, filt_out.ve_std[i],lab="filter 1-σ",lc=:blue)
        plot!(p4,tt,-filt_out.ve_std[i],lab=false,lc=:blue)
        show_plot && display(p4)
        save_plot && png(p4,"east_velocity_error.png")

    end

    # p5 = plot(xlab="time [min]",ylab="altitude error [m]",dpi=dpi)
    # plot!(p5,tt, filt_out.alt_err[i],lab="filter error",lc=:black,lw=2)
    # plot!(p5,tt, filt_out.alt_std[i],lab="filter 1-σ",lc=:blue)
    # plot!(p5,tt,-filt_out.alt_std[i],lab=lab=false,lc=:blue)
    # plot!(p5,tt, crlb_out.alt_std[i],lab="CRLB 1-σ",lc=:red,ls=:dash)
    # plot!(p5,tt,-crlb_out.alt_std[i],lab=false,lc=:red,ls=:dash)
    # show_plot && display(p5)
    # save_plot && png(p5,"altitude_error.png")

    # p6 = plot(xlab="time [min]",ylab="down velocity error [m/s]",dpi=dpi)
    # plot!(p6,tt, filt_out.vd_err[i],lab="filter error",lc=:black,lw=2)
    # plot!(p6,tt, filt_out.vd_std[i],lab="filter 1-σ",lc=:blue)
    # plot!(p6,tt,-filt_out.vd_std[i],lab=false,lc=:blue)
    # show_plot && display(p6)
    # save_plot && png(p6,"down_velocity_error.png")

    # p7 = plot(xlab="time [min]",ylab="north tilt error [deg]",dpi=dpi)
    # plot!(p7,tt, rad2deg.(filt_out.tn_err[i]),lab="filter error",lc=:black,lw=2)
    # plot!(p7,tt, rad2deg.(filt_out.tn_std[i]),lab="filter 1-σ",lc=:blue)
    # plot!(p7,tt,-rad2deg.(filt_out.tn_std[i]),lab=false,lc=:blue)
    # show_plot && display(p7)
    # save_plot && png(p7,"north_tilt_error.png")

    # p8 = plot(xlab="time [min]",ylab="east tilt error [deg]",dpi=dpi)
    # plot!(p8,tt, rad2deg.(filt_out.te_err[i]),lab="filter error",lc=:black,lw=2)
    # plot!(p8,tt, rad2deg.(filt_out.te_std[i]),lab="filter 1-σ",lc=:blue)
    # plot!(p8,tt,-rad2deg.(filt_out.te_std[i]),lab=false,lc=:blue)
    # show_plot && display(p8)
    # save_plot && png(p8,"east_tilt_error.png")

    # p9 = plot(xlab="time [min]",ylab="down tilt error [deg]",dpi=dpi)
    # plot!(p9,tt, rad2deg.(filt_out.td_err[i]),lab="filter error",lc=:black,lw=2)
    # plot!(p9,tt, rad2deg.(filt_out.td_std[i]),lab="filter 1-σ",lc=:blue)
    # plot!(p9,tt,-rad2deg.(filt_out.td_std[i]),lab=false,lc=:blue)
    # show_plot && display(p9)
    # save_plot && png(p9,"down_tilt_error.png")

    # p10 = plot(xlab="time [min]",ylab="barometer aiding altitude σ [m]",dpi=dpi)
    # plot!(p10,tt, filt_out.ha_std[i],lab="filter 1-σ",lc=:blue)
    # plot!(p10,tt,-filt_out.ha_std[i],lab=false,lc=:blue)
    # show_plot && display(p10)
    # save_plot && png(p10,"barometer_aiding_altitude_σ.png")

    # p11 = plot(xlab="time [min]",ylab="barometer aiding vertical accel σ [m/s^2]",dpi=dpi)
    # plot!(p11,tt, filt_out.ah_std[i],lab="filter 1-σ",lc=:blue)
    # plot!(p11,tt,-filt_out.ah_std[i],lab=false,lc=:blue)
    # show_plot && display(p11)
    # save_plot && png(p11,"barometer_aiding_vertical_accel_σ.png")

    # p12 = plot(xlab="time [min]",ylab="accelerometer bias σ [m/s^2]",dpi=dpi)
    # plot!(p12,tt,filt_out.ax_std[i],lab="x")
    # plot!(p12,tt,filt_out.ay_std[i],lab="y")
    # plot!(p12,tt,filt_out.az_std[i],lab="z")
    # show_plot && display(p12)
    # save_plot && png(p12,"accelerometer_bias_σ.png")

    # p13 = plot(xlab="time [min]",ylab="gyroscope bias σ [rad/s]",dpi=dpi)
    # plot!(p13,tt,filt_out.gx_std[i],lab="x")
    # plot!(p13,tt,filt_out.gy_std[i],lab="y")
    # plot!(p13,tt,filt_out.gz_std[i],lab="z")
    # show_plot && display(p13)
    # save_plot && png(p13,"gyroscope_bias_σ.png")

    return plot_vel ? (p1, p2, p3, p4) : (p1, p2)
end # function plot_filt_err

"""
    plot_mag_map(path::Path, mag, itp_mapS;
                 lab::String        = "magnetometer",
                 order::Symbol      = :magmap,
                 dpi::Int           = 200,
                 Nmax::Int          = 5000,
                 detrend_data::Bool = true,
                 show_plot::Bool    = true,
                 save_plot::Bool    = false,
                 plot_png::String   = "mag_vs_map.png")

Plot scalar magnetometer measurements vs map values.

**Arguments:**
- `path`:         `Path` struct, i.e., `Traj` trajectory struct, `INS` inertial navigation system struct, or `FILTout` filter extracted output struct
- `mag`:          scalar magnetometer measurements [nT]
- `itp_mapS`:     scalar map interpolation function (`f(lat,lon)` or `f(lat,lon,alt)`)
- `lab`:          (optional) magnetometer data (legend) label
- `order`:        (optional) plotting order {`:magmap`,`:mapmag`}
- `dpi`:          (optional) dots per inch (image resolution)
- `Nmax`:         (optional) maximum number of data points plotted
- `detrend_data`: (optional) if true, detrend plot data
- `show_plot`:    (optional) if true, show `p1`
- `save_plot`:    (optional) if true, save `p1` as `plot_png`
- `plot_png`:     (optional) plot file name to save (`.png` extension optional)

**Returns:**
- `p1`: scalar magnetometer measurements vs map values
"""
function plot_mag_map(path::Path, mag, itp_mapS;
                      lab::String        = "magnetometer",
                      order::Symbol      = :magmap,
                      dpi::Int           = 200,
                      Nmax::Int          = 5000,
                      detrend_data::Bool = true,
                      show_plot::Bool    = true,
                      save_plot::Bool    = false,
                      plot_png::String   = "mag_vs_map.png")

    i  = downsample(1:path.N,Nmax)
    tt = (path.tt[i] .- path.tt[i][1]) / 60

    map_val = itp_mapS.(path.lat[i],path.lon[i],path.alt[i])
    mag_val = detrend_data ? detrend(mag[i] ) : mag[i]
    map_val = detrend_data ? detrend(map_val) : map_val

    p1 = plot(xlab="time [min]",ylab="magnetic field [nT]",dpi=dpi)
    if order == :magmap
        plot!(p1,tt,mag_val,lab=lab)
        plot!(p1,tt,map_val,lab="anomaly map")
    elseif order == :mapmag
        plot!(p1,tt,map_val,lab="anomaly map")
        plot!(p1,tt,mag_val,lab=lab)
    else
        error("order $order not defined")
    end

    show_plot && display(p1)
    save_plot && png(p1,plot_png)

    return (p1)
end # function plot_mag_map

"""
    plot_mag_map_err(path::Path, mag, itp_mapS;
                     lab::String        = "",
                     dpi::Int           = 200,
                     Nmax::Int          = 5000,
                     detrend_data::Bool = true,
                     show_plot::Bool    = true,
                     save_plot::Bool    = false,
                     plot_png::String   = "mag_map_err.png")

Plot scalar magnetometer measurement vs map value errors.

**Arguments:**
- `path`:         `Path` struct, i.e., `Traj` trajectory struct, `INS` inertial navigation system struct, or `FILTout` filter extracted output struct
- `mag`:          scalar magnetometer measurements [nT]
- `itp_mapS`:     scalar map interpolation function (`f(lat,lon)` or `f(lat,lon,alt)`)
- `lab`:          (optional) data (legend) label
- `dpi`:          (optional) dots per inch (image resolution)
- `Nmax`:         (optional) maximum number of data points plotted
- `detrend_data`: (optional) if true, detrend plot data
- `show_plot`:    (optional) if true, show `p1`
- `save_plot`:    (optional) if true, save `p1` as `plot_png`
- `plot_png`:     (optional) plot file name to save (`.png` extension optional)

**Returns:**
- `p1`: scalar magnetometer measurement vs map value errors
"""
function plot_mag_map_err(path::Path, mag, itp_mapS;
                          lab::String        = "",
                          dpi::Int           = 200,
                          Nmax::Int          = 5000,
                          detrend_data::Bool = true,
                          show_plot::Bool    = true,
                          save_plot::Bool    = false,
                          plot_png::String   = "mag_map_err.png")

    i  = downsample(1:path.N,Nmax)
    tt = (path.tt[i] .- path.tt[i][1]) / 60

    l  = detrend_data ? "detrended " : ""
    p1 = plot(xlab="time [min]",ylab=l*"magnetic signal error [nT]",dpi=dpi)

    f = detrend_data ? detrend : x -> x

    map_val = itp_mapS.(path.lat[i],path.lon[i],path.alt[i])

    plot!(p1,tt,f(mag[i] - map_val),lab=lab)

    show_plot && display(p1)
    save_plot && png(p1,plot_png)

	err = round(std(mag[i] - map_val),digits=2)
    @info("mag-map error standard deviation = $err nT")

    return (p1)
end # function plot_mag_map_err

"""
    get_autocor(x::Vector, dt = 0.1, dt_max = 300.0)

Get autocorrelation of data (e.g., actual - expected measurements).

**Arguments:**
- `x`:      data vector
- `dt`:     (optional) measurement time step [s]
- `dt_max`: (optional) maximum time step to evaluate [s]

**Returns:**
- `sigma`: standard deviation
- `tau`:   autocorrelation decay to e^-1 of `x` [s]
"""
function get_autocor(x::Vector, dt = 0.1, dt_max = 300.0)
    sigma = round(Int,std(x))
    dts   = 0:dt:dt_max
    lags  = round.(Int,dts/dt)
    x_ac  = autocor(x,lags)
    i     = findfirst(x_ac .< exp(-1))
    tau   = i isa Nothing ? dt_max : round(Int,dts[i])
    return (sigma, tau)
end # function get_autocor

"""
    plot_autocor(x::Vector, dt = 0.1, dt_max = 300.0;
                 show_plot::Bool  = true,
                 save_plot::Bool  = false,
                 plot_png::String = "autocor.png")

Plot autocorrelation of data (e.g., actual - expected measurements). Prints out
`σ` = standard deviation & `τ` = autocorrelation decay to e^-1 of `x`.

**Arguments:**
- `x`:         data vector
- `dt`:        (optional) measurement time step [s]
- `dt_max`:    (optional) maximum time step to evaluate [s]
- `show_plot`: (optional) if true, show `p1`
- `save_plot`: (optional) if true, save `p1` as `plot_png`
- `plot_png`:  (optional) plot file name to save (`.png` extension optional)

**Returns:**
- `p1`: plot of autocorrelation of `x`
"""
function plot_autocor(x::Vector, dt = 0.1, dt_max = 300.0;
                      show_plot::Bool  = true,
                      save_plot::Bool  = false,
                      plot_png::String = "autocor.png")

    (sigma,tau) = get_autocor(x,dt,dt_max) # 1002.17: σ = 5, τ ≈ 45

    @info("σ ≈ $sigma")
    @info("τ ≈ $tau")
    tau == dt_max && @info("τ not in range")

    dts  = 0:dt:dt_max
    lags = round.(Int,dts/dt)
    x_ac = autocor(x,lags)
    p1   = plot(dts,x_ac,lab=false);

    show_plot && display(p1)
    save_plot && png(p1,plot_png)

    return (p1)
end # function plot_autocor

"""
    chisq_pdf(x, k::Int = 1)

Probability density function (PDF) of the chi-square distribution.

**Arguments:**
- `x`: chi-square value
- `k`: degrees of freedom

**Returns:**
- `f`: probability density function (PDF) at `x` with `k`
"""
function chisq_pdf(x, k::Int = 1)
    f = x > 0 ? x^(k/2-1) * exp(-x/2) / (2^(k/2) * gamma(k/2)) : 0
    return float(f)
end # function chisq_pdf

"""
    chisq_cdf(x, k::Int = 1)

Cumulative distribution function (CDF) of the chi-square distribution.

**Arguments:**
- `x`: chi-square value
- `k`: degrees of freedom

**Returns:**
- `F`: cumulative distribution function (CDF) at `x` with `k`
"""
function chisq_cdf(x, k::Int = 1)
    F = x > 0 ? gamma_inc.(k/2,x/2)[1] : 0
    return float(F)
end # function chisq_cdf

"""
    chisq_q(P = 0.95, k::Int = 1)

Quantile function (inverse CDF) of the chi-square distribution.

**Arguments:**
- `P`: percentile (`1 - p-value`), i.e., lower incomplete gamma function ratio
- `k`: degrees of freedom

**Returns:**
- `x`: chi-square value
"""
function chisq_q(P = 0.95, k::Int = 1)
    gamma_inc_inv(k/2,P,1-P)*2
end # function chisq_q

"""
    points_ellipse(P; clip = Inf, n::Int = 61)

Internal helper function to create `x` & `y` confidence ellipse points for a
`2` x `2` covariance matrix.

**Arguments:**
- `P`:    `2` x `2` covariance matrix
- `clip`: (optional) clipping radius
- `n`:    (optional) number of confidence ellipse points

**Returns:**
- `x`: x-axis confidence ellipse points
- `y`: y-axis confidence ellipse points
"""
function points_ellipse(P; clip = Inf, n::Int = 61)
    θ = LinRange(0,2*pi,n) # angles around circle

    (eigval,eigvec) = eigen(P); # eigenvalues & eigenvectors
    xy = [cos.(θ) sin.(θ)] * sqrt.(Diagonal(eigval)) * eigvec' # transformation
    x  = xy[:,1]
    y  = xy[:,2]

    # clip data to clipping radius
    r = sqrt.(x.^2 + y.^2) # Euclidian distance
    i = r .> clip
    x[i] .= NaN
    y[i] .= NaN

    return (x, y)
end # function points_ellipse

"""
    conf_ellipse!(p1::Plot, P;
                  μ                    = zeros(eltype(P),2),
                  conf                 = 0.95,
                  clip                 = Inf,
                  n::Int               = 61,
                  lim                  = nothing,
                  margin::Int          = 2,
                  lab::String          = "[m]",
                  axis::Bool           = true,
                  plot_eigax::Bool     = false,
                  bg_color::Symbol     = :white,
                  ce_color::Symbol     = :black,
                  b_e::AbstractBackend = gr())

Internal helper function to plot a confidence ellipse for a `2` x `2`
covariance matrix (2 degrees of freedom) on an existing plot. Visualization of
a 2D confidence interval.

**Arguments:**
- `p1`:         plot (i.e., map)
- `P`:          `2` x `2` covariance matrix
- `μ`:          (optional) confidence ellipse center [`P` units]
- `conf`:       (optional) percentile {0:1}
- `clip`:       (optional) clipping radius [`P` units]
- `n`:          (optional) number of confidence ellipse points
- `lim`:        (optional) plot `x` & `y` limits (`-lim`,`lim`) [`P` units]
- `margin`:     (optional) margin around plot [mm]
- `lab`:        (optional) axis labels
- `axis`:       (optional) if true, show axes
- `plot_eigax`: (optional) if true, show major & minor axes
- `bg_color`:   (optional) background color
- `ce_color`:   (optional) confidence ellipse color
- `b_e`:        (optional) plotting backend

**Returns:**
- `nothing`: confidence ellipse is plotted on `p1`
"""
function conf_ellipse!(p1::Plot, P;
                       μ                    = zeros(eltype(P),2),
                       conf                 = 0.95,
                       clip                 = Inf,
                       n::Int               = 61,
                       lim                  = nothing,
                       margin::Int          = 2,
                       lab::String          = "[m]",
                       axis::Bool           = true,
                       plot_eigax::Bool     = false,
                       bg_color::Symbol     = :white,
                       ce_color::Symbol     = :black,
                       b_e::AbstractBackend = gr())

    (eigval,eigvec) = eigen(P)
    eigax = I*sqrt.(Diagonal(eigval)) * eigvec'

    # check arguments
    xlim = ylim = lim isa Nothing  ? lim : (-lim,lim)
    xlab = ylab = axis             ? lab : ""
    @assert all(real(eigval) .> 0) "P is not positive definite"
    @assert size(P) == (2,2)       "P is size $(size(P)) ≂̸ (2,2)"
    @assert length(μ) == 2         "μ is length $(length(μ)) ≂̸ 2"
    @assert 0 < conf < 1           "conf is not 0 < $conf < 1"

    k  = sqrt(chisq_q(conf,2)) # compute quantile for desired percentile
    b_e # backend
    if lim isa Nothing
        plot!(p1,xlab=xlab,ylab=ylab,
              legend=false,aspect_ratio=:equal,margin=margin*mm,
              axis=axis,xticks=axis,yticks=axis,grid=axis,bg=bg_color)
    else
        plot!(p1,xlim=xlim,ylim=ylim,xlab=xlab,ylab=ylab,
              legend=false,aspect_ratio=:equal,margin=margin*mm,
              axis=axis,xticks=axis,yticks=axis,grid=axis,bg=bg_color)
    end

    (x,y) = points_ellipse(P;clip=clip,n=n)
    plot!(p1, μ[1].+k*x, μ[2].+k*y, lc=ce_color, lw=2)
    if plot_eigax
        plot!(p1,[-k,k]*eigax[1,1],[-k,k]*eigax[1,2],lc=:red,ls=:dash,lw=1)
        plot!(p1,[-k,k]*eigax[2,1],[-k,k]*eigax[2,2],lc=:red,ls=:dash,lw=2)
    end

    return (nothing)
end # function conf_ellipse!

"""
    conf_ellipse(P;
                 μ                    = zeros(eltype(P),2),
                 conf                 = 0.95,
                 clip                 = Inf,
                 n::Int               = 61,
                 lim                  = nothing,
                 margin::Int          = 2,
                 lab::String          = "[m]",
                 axis::Bool           = true,
                 plot_eigax::Bool     = false,
                 bg_color::Symbol     = :white,
                 ce_color::Symbol     = :black,
                 b_e::AbstractBackend = gr())

Internal helper function to plot a confidence ellipse for a `2` x `2` covariance
matrix (2 degrees of freedom). Visualization of a 2D confidence interval.

**Arguments:**
- `P`:          `2` x `2` covariance matrix
- `μ`:          (optional) confidence ellipse center [`P` units]
- `conf`:       (optional) percentile {0:1}
- `clip`:       (optional) clipping radius [`P` units]
- `n`:          (optional) number of confidence ellipse points
- `lim`:        (optional) plot `x` & `y` limits (`-lim`,`lim`) [`P` units]
- `margin`:     (optional) margin around plot [mm]
- `lab`:        (optional) axis labels
- `axis`:       (optional) if true, show axes
- `plot_eigax`: (optional) if true, show major & minor axes
- `bg_color`:   (optional) background color
- `ce_color`:   (optional) confidence ellipse color
- `b_e`:        (optional) plotting backend

**Returns:**
- `p1`: plot of confidence ellipse
"""
function conf_ellipse(P;
                      μ                    = zeros(eltype(P),2),
                      conf                 = 0.95,
                      clip                 = Inf,
                      n::Int               = 61,
                      lim                  = nothing,
                      margin::Int          = 2,
                      lab::String          = "[m]",
                      axis::Bool           = true,
                      plot_eigax::Bool     = false,
                      bg_color::Symbol     = :white,
                      ce_color::Symbol     = :black,
                      b_e::AbstractBackend = gr())
    b_e # backend
    p1 = plot()
    conf_ellipse!(p1,P;
                  μ          = μ,
                  conf       = conf,
                  clip       = clip,
                  n          = n,
                  lim        = lim,
                  margin     = margin,
                  lab        = lab,
                  axis       = axis,
                  plot_eigax = plot_eigax,
                  bg_color   = bg_color,
                  ce_color   = ce_color,
                  b_e        = b_e)
    return (p1)
end # function conf_ellipse

"""
    units_ellipse(P; conf_units::Symbol = :m, lat1 = deg2rad(45))

Internal helper function to convert (position) confidence ellipse units for a
`2` x `2` covariance matrix.

**Arguments:**
- `P`:          `2` x `2` covariance matrix [rad]
- `conf_units`: (optional) confidence ellipse units {`:m`,`:ft`,`:deg`,`:rad`}
- `lat1`:       (optional) nominal latitude [rad], only used if `conf_units = :m` or `:ft`

**Returns:**
- `P`: `2` x `2` covariance matrix with converted units
"""
function units_ellipse(P; conf_units::Symbol = :m, lat1 = deg2rad(45))
    @assert size(P,1) == 2 "P is size $(size(P)) ≂̸ (2,2)"
    @assert size(P,2) == 2 "P is size $(size(P)) ≂̸ (2,2)"

    P = float.(P)

    if conf_units == :deg
        P = rad2deg.(rad2deg.(P)) # deg^2
    elseif conf_units in [:m,:ft]
        l = [dlat2dn(1,lat1),dlon2de(1,lat1)] # m/rad
        conf_units == :ft && (l ./= 0.3048)   # ft/rad
        P = P .* (l*l') # m^2 or ft^2
    elseif conf_units != :rad
        error("$conf_units confidence ellipse units not defined")
    end

    return (P)
end # function units_ellipse

"""
    units_ellipse(filt_res::FILTres, filt_out::FILTout;
                  conf_units::Symbol = :m)

Internal helper function to convert (position) confidence ellipse units for a
`2` x `2` covariance matrix.

**Arguments:**
- `filt_res`:   `FILTres` filter results struct
- `filt_out`:   `FILTout` filter extracted output struct
- `conf_units`: (optional) confidence ellipse units {`:m`,`:ft`,`:deg`,`:rad`}

**Returns:**
- `P`: `2` x `2` covariance matrix with converted units
"""
function units_ellipse(filt_res::FILTres, filt_out::FILTout;
                       conf_units::Symbol = :m)
    units_ellipse(float.(filt_res.P[1:2,1:2,:]);
                  conf_units = conf_units,
                  lat1       = mean(filt_out.lat))
end # function units_ellipse

"""
    gif_ellipse(P, lat1 = deg2rad(45);
                dt                   = 0.1,
                di::Int              = 10,
                speedup::Int         = 60,
                conf_units::Symbol   = :m,
                μ                    = zeros(eltype(P),2),
                conf                 = 0.95,
                clip                 = Inf,
                n::Int               = 61,
                lim                  = 500,
                margin::Int          = 2,
                axis::Bool           = true,
                plot_eigax::Bool     = false,
                bg_color::Symbol     = :white,
                ce_color::Symbol     = :black,
                b_e::AbstractBackend = gr(),
                save_plot::Bool      = false,
                ellipse_gif::String  = "conf_ellipse.gif")

Create a (position) confidence ellipse GIF animation for a `2` x `2` (x `N`)
covariance matrix.

**Arguments:**
- `P`:           `2` x `2` (x `N`) covariance matrix
- `lat1`:        (optional) nominal latitude [rad], only used if `conf_units = :m` or `:ft`
- `dt`:          (optional) measurement time step [s]
- `di`:          (optional) GIF measurement interval (e.g., `di = 10` uses every 10th measurement)
- `speedup`:     (optional) GIF speedup (e.g., `speedup = 60` is 60x speed)
- `conf_units`:  (optional) confidence ellipse units {`:m`,`:ft`,`:deg`,`:rad`}
- `μ`:           (optional) confidence ellipse center [`conf_units`]
- `conf`:        (optional) percentile {0:1}
- `clip`:        (optional) clipping radius [`conf_units`]
- `n`:           (optional) number of confidence ellipse points
- `lim`:         (optional) plot `x` & `y` limits (`-lim`,`lim`) [`conf_units`]
- `margin`:      (optional) margin around plot [mm]
- `axis`:        (optional) if true, show axes
- `plot_eigax`:  (optional) if true, show major & minor axes
- `bg_color`:    (optional) background color
- `ce_color`:    (optional) confidence ellipse color
- `b_e`:         (optional) plotting backend
- `save_plot`:   (optional) if true, save `g1` as `ellipse_gif`
- `ellipse_gif`: (optional) path/name of confidence ellipse GIF file to save (`.gif` extension optional)

**Returns:**
- `g1`: confidence ellipse GIF animation
"""
function gif_ellipse(P, lat1 = deg2rad(45);
                     dt                   = 0.1,
                     di::Int              = 10,
                     speedup::Int         = 60,
                     conf_units::Symbol   = :m,
                     μ                    = zeros(eltype(P),2),
                     conf                 = 0.95,
                     clip                 = Inf,
                     n::Int               = 61,
                     lim                  = 500,
                     margin::Int          = 2,
                     axis::Bool           = true,
                     plot_eigax::Bool     = false,
                     bg_color::Symbol     = :white,
                     ce_color::Symbol     = :black,
                     b_e::AbstractBackend = gr(),
                     save_plot::Bool      = false,
                     ellipse_gif::String  = "conf_ellipse.gif")

    P  = units_ellipse(P;conf_units=conf_units,lat1=lat1)
    a1 = Animation()

    for i = 1:di:size(P,3)
        p1 = conf_ellipse(P[:,:,i];
                          μ          = μ,
                          conf       = conf,
                          clip       = clip,
                          n          = n,
                          lim        = lim,
                          margin     = margin,
                          axis       = axis,
                          plot_eigax = plot_eigax,
                          bg_color   = bg_color,
                          ce_color   = ce_color,
                          b_e        = b_e);
        frame(a1,p1);
    end

    # show or save gif
    ellipse_gif = add_extension(ellipse_gif,".gif")
    fps         = 1/dt/di*speedup
    g1          = save_plot ? gif(a1,ellipse_gif;fps=fps) : gif(a1;fps=fps)

    return (g1)
end # function gif_ellipse

"""
    gif_ellipse(filt_res::FILTres,
                filt_out::FILTout,
                map_map::Map         = mapS_null;
                dt                   = 0.1,
                di::Int              = 10,
                speedup::Int         = 60,
                conf_units::Symbol   = :m,
                μ                    = zeros(eltype(filt_res.P),2),
                conf                 = 0.95,
                clip                 = Inf,
                n::Int               = 61,
                lim                  = 500,
                dpi::Int             = 200,
                margin::Int          = 2,
                axis::Bool           = true,
                plot_eigax::Bool     = false,
                bg_color::Symbol     = :white,
                ce_color::Symbol     = :black,
                map_color::Symbol    = :usgs,
                clims::Tuple         = (),
                b_e::AbstractBackend = gr(),
                save_plot::Bool      = false,
                ellipse_gif::String  = "conf_ellipse.gif")

Create a (position) confidence ellipse GIF animation for a `2` x `2` (x `N`)
covariance matrix.

**Arguments:**
- `filt_res`:    `FILTres` filter results struct
- `filt_out`:    `FILTout` filter extracted output struct
- `map_map`:     (optional) `Map` magnetic anomaly map struct
- `dt`:          (optional) measurement time step [s]
- `di`:          (optional) GIF measurement interval (e.g., `di = 10` uses every 10th measurement)
- `speedup`:     (optional) GIF speedup (e.g., `speedup = 60` is 60x speed)
- `conf_units`:  (optional) confidence ellipse units {`:m`,`:ft`,`:deg`,`:rad`}
- `μ`:           (optional) confidence ellipse center [`conf_units`]
- `conf`:        (optional) percentile {0:1}
- `clip`:        (optional) clipping radius [`conf_units`]
- `n`:           (optional) number of confidence ellipse points
- `lim`:         (optional) plot `x` & `y` limits (`-lim`,`lim`) [`conf_units`]
- `dpi`:         (optional) dots per inch (image resolution)
- `margin`:      (optional) margin around plot [mm]
- `axis`:        (optional) if true, show axes
- `plot_eigax`:  (optional) if true, show major & minor axes
- `bg_color`:    (optional) background color
- `ce_color`:    (optional) confidence ellipse color
- `map_color`:   (optional) filled contour color scheme {`:usgs`,`:gray`,`:gray1`,`:gray2`,`:plasma`,`:magma`}
- `clims`:       (optional) length-`2` map colorbar limits `(cmin,cmax)`
- `b_e`:         (optional) plotting backend
- `save_plot`:   (optional) if true, save `g1` as `ellipse_gif`
- `ellipse_gif`: (optional) path/name of confidence ellipse GIF file to save (`.gif` extension optional)

**Returns:**
- `g1`: confidence ellipse GIF animation
"""
function gif_ellipse(filt_res::FILTres,
                     filt_out::FILTout,
                     map_map::Map         = mapS_null;
                     dt                   = 0.1,
                     di::Int              = 10,
                     speedup::Int         = 60,
                     conf_units::Symbol   = :m,
                     μ                    = zeros(eltype(filt_res.P),2),
                     conf                 = 0.95,
                     clip                 = Inf,
                     n::Int               = 61,
                     lim                  = 500,
                     dpi::Int             = 200,
                     margin::Int          = 2,
                     axis::Bool           = true,
                     plot_eigax::Bool     = false,
                     bg_color::Symbol     = :white,
                     ce_color::Symbol     = :black,
                     map_color::Symbol    = :usgs,
                     clims::Tuple         = (),
                     b_e::AbstractBackend = gr(),
                     save_plot::Bool      = false,
                     ellipse_gif::String  = "conf_ellipse.gif")

    dx = dlon2de(get_step(map_map.xx),mean(map_map.yy))
    dy = dlat2dn(get_step(map_map.yy),mean(map_map.yy))

    if !any(iszero.([dx,dy]) .| isnan.([dx,dy])) & isempty(clims)
        num = ceil(Int,1.5*lim/minimum([dx,dy]))
        x1  = findmin(abs.(map_map.xx.-minimum(filt_out.lon)))[2]
        x2  = findmin(abs.(map_map.xx.-maximum(filt_out.lon)))[2]
        y1  = findmin(abs.(map_map.yy.-minimum(filt_out.lat)))[2]
        y2  = findmin(abs.(map_map.yy.-maximum(filt_out.lat)))[2]
        (x1,x2)   = sort([x1,x2])
        (y1,y2)   = sort([y1,y2])
        (_,clims) = map_clims(map_cs(map_color),map_map.map[y1:y2,x1:x2,1])
    end

    P  = units_ellipse(filt_res,filt_out;conf_units=conf_units)
    a1 = Animation()

    for i = 1:di:size(P,3)

        if !any(iszero.([dx,dy]) .| isnan.([dx,dy]))
            xi     = findmin(abs.(map_map.xx.-filt_out.lon[i]))[2]
            yi     = findmin(abs.(map_map.yy.-filt_out.lat[i]))[2]
            ind_xx = max(xi-num,1):min(xi+num,length(map_map.xx))
            ind_yy = max(yi-num,1):min(yi+num,length(map_map.yy))
            p1     = plot_map(map_map.map[ind_yy,ind_xx,1],
                              map_map.xx[ind_xx],map_map.yy[ind_yy];
                              clims=clims,dpi=dpi,margin=margin,Nmax=10^10,
                              legend=false,axis=false,
                              map_color=map_color,bg_color=bg_color,
                              map_units=:rad,plot_units=conf_units,b_e=b_e)
        else
            p1 = plot()
        end

        conf_ellipse!(p1,P[:,:,i];
                      μ          = μ,
                      conf       = conf,
                      clip       = clip,
                      n          = n,
                      lim        = lim,
                      margin     = margin,
                      axis       = axis,
                      plot_eigax = plot_eigax,
                      bg_color   = bg_color,
                      ce_color   = ce_color,
                      b_e        = b_e);
        frame(a1,p1)
    end

    # show or save gif
    ellipse_gif = add_extension(ellipse_gif,".gif")
    fps         = 1/dt/di*speedup
    g1          = save_plot ? gif(a1,ellipse_gif;fps=fps) : gif(a1;fps=fps)

    return (g1)
end # function gif_ellipse
