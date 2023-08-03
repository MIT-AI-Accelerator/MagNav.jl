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
             nn_x           = nothing,
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
- `itp_mapS`:  scalar map interpolation function
- `filt_type`: (optional) filter type {`:ekf`,`:mpf`}
- `P0`:        (optional) initial covariance matrix
- `Qd`:        (optional) discrete time process/system noise matrix
- `R`:         (optional) measurement (white) noise variance
- `num_part`:  (optional) number of particles (`:mpf` only)
- `thresh`:    (optional) resampling threshold fraction {0:1} (`:mpf` only)
- `baro_tau`:  (optional) barometer time constant [s]
- `acc_tau`:   (optional) accelerometer time constant [s]
- `gyro_tau`:  (optional) gyroscope time constant [s]
- `fogm_tau`:  (optional) FOGM catch-all time constant [s]
- `date`:      (optional) measurement date for IGRF [yr]
- `core`:      (optional) if true, include core magnetic field in measurement
- `map_alt`:   (optional) map altitude [m]
- `nn_x`:      (optional) `x` matrix for neural network
- `m`:         (optional) neural network model
- `y_norms`:   (optional) Tuple of `y` normalizations, i.e., `(y_bias,y_scale)`
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
function run_filt(traj::Traj, ins::INS, meas, itp_mapS, filt_type::Symbol=:ekf;
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
                  nn_x           = nothing,
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
        filt_res = ekf_online_nn(ins,meas,itp_mapS,nn_x,m,y_norms,P0,Qd,R;
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
        filt_res = nekf(ins,meas,itp_mapS,nn_x,m;
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
                  nn_x       = nothing,
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
                 nn_x        = nn_x,
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

Extract CRLB, INS, and filter results.

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

    crlb_out = CRLBout(zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),
                       zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),zeros(N))

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
    @info("CRLB DRMS Error = $crlb_DRMS m")

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

    ins_out = INSout(zeros(N),zeros(N),zeros(N),zeros(N),
                     zeros(N),zeros(N),zeros(N),zeros(N),
                     zeros(N),zeros(N))

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

    N =  traj.N
    dt = traj.dt

    filt_out = FILTout(N,dt    ,zeros(N),zeros(N),zeros(N),zeros(N),
                       zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),
                       zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),
                       zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),
                       zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),
                       zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),
                       zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),
                       zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),
                       zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),
                       zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),zeros(N))

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

    n_tilt = zeros(N)
    e_tilt = zeros(N)
    d_tilt = zeros(N)

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
    @info("FILT DRMS Error = $filt_DRMS m")

    return (filt_out)
end # function eval_filt

"""
    plot_filt!(p1, traj::Traj, ins::INS, filt_out::FILTout;
               vel_plot::Bool  = false,
               dpi::Int        = 200,
               Nmax::Int       = 5000,
               show_plot::Bool = true,
               save_plot::Bool = false)

Plot flights paths on an existing plot and latitude & longitude vs time.

**Arguments:**
- `p1`:        existing plot (e.g., map)
- `traj`:      `Traj` trajectory struct
- `ins`:       `INS` inertial navigation system struct
- `filt_out`:  `FILTout` filter extracted output struct
- `vel_plot`:  (optional) if true, plot velocities
- `dpi`:       (optional) dots per inch (image resolution)
- `Nmax`:      (optional) maximum number of data points plotted
- `show_plot`: (optional) if true, show plot
- `save_plot`: (optional) if true, plots will be saved with default file names

**Returns:**
- `p1`: flight paths are plotted on existing `p1`
- `p2`: latitudes  vs time
- `p3`: longitudes vs time
- `p5`: north velocities vs time
- `p6`: east  velocities vs time
"""
function plot_filt!(p1, traj::Traj, ins::INS, filt_out::FILTout;
                    vel_plot::Bool  = false,
                    dpi::Int        = 200,
                    Nmax::Int       = 5000,
                    show_plot::Bool = true,
                    save_plot::Bool = false)

    i    = downsample(1:traj.N,Nmax)
    tt   = (traj.tt[i] .- traj.tt[i][1]) / 60
    lon  = rad2deg.([traj.lon;ins.lon;filt_out.lon])
    lat  = rad2deg.([traj.lat;ins.lat;filt_out.lat])
    xlim = get_lim(lon,0.05)
    ylim = get_lim(lat,0.05)

    plot!(p1,xlab="longitude [deg]",ylab="latitude [deg]",dpi=dpi)
    plot!(p1,rad2deg.(traj.lon[i])    ,rad2deg.(traj.lat[i])    ,lab="gps")
    plot!(p1,rad2deg.(ins.lon[i])     ,rad2deg.(ins.lat[i])     ,lab="ins")
    plot!(p1,rad2deg.(filt_out.lon[i]),rad2deg.(filt_out.lat[i]),lab="filter")
    plot!(p1,xlim=xlim,ylim=ylim)
    show_plot && display(p1)
    save_plot && png(p1,"flight_path.png")

    p2 = plot(xlab="time [min]",ylab="latitude [deg]",dpi=dpi)
    plot!(p2,tt,rad2deg.(traj.lat[i])    ,lab="gps")
    plot!(p2,tt,rad2deg.(ins.lat[i])     ,lab="ins")
    plot!(p2,tt,rad2deg.(filt_out.lat[i]),lab="filter")
    show_plot && display(p2)
    save_plot && png(p2,"latitude.png")

    p3 = plot(xlab="time [min]",ylab="longitude [deg]",dpi=dpi)
    plot!(p3,tt,rad2deg.(traj.lon[i])    ,lab="gps")
    plot!(p3,tt,rad2deg.(ins.lon[i])     ,lab="ins")
    plot!(p3,tt,rad2deg.(filt_out.lon[i]),lab="filter")
    show_plot && display(p3)
    save_plot && png(p3,"longitude.png")

    # p4 = plot(xlab="time [min]",ylab="altitude [m]",dpi=dpi)
    # plot!(p4,tt,traj.alt[i]    ,lab="gps")
    # plot!(p4,tt,ins.alt[i]     ,lab="ins")
    # plot!(p4,tt,filt_out.alt[i],lab="filter")
    # show_plot && display(p4)
    # save_plot && png(p4,"altitude.png")

    if vel_plot
        p5 = plot(xlab="time [min]",ylab="north velocity [m/s]",dpi=dpi)
        plot!(p5,tt,traj.vn[i]    ,lab="gps")
        plot!(p5,tt,ins.vn[i]     ,lab="ins")
        plot!(p5,tt,filt_out.vn[i],lab="filter")
        show_plot && display(p5)
        save_plot && png(p5,"north_velocity.png")

        p6 = plot(xlab="time [min]",ylab="east velocity [m/s]",dpi=dpi)
        plot!(p6,tt,traj.ve[i]    ,lab="gps")
        plot!(p6,tt,ins.ve[i]     ,lab="ins")
        plot!(p6,tt,filt_out.ve[i],lab="filter")
        show_plot && display(p6)
        save_plot && png(p6,"east_velocity.png")
    end

    # p7 = plot(xlab="time [min]",ylab="down velocity [m/s]",dpi=dpi)
    # plot!(p7,tt,traj.vd[i]    ,lab="gps")
    # plot!(p7,tt,ins.vd[i]     ,lab="ins")
    # plot!(p7,tt,filt_out.vd[i],lab="filter")
    # show_plot && display(p7)
    # save_plot && png(p7,"down_velocity.png")

    # p8 = plot(xlab="time [min]",ylab="tilt [deg]",dpi=dpi)
    # plot!(p8,tt,rad2deg.(filt_out.tn[i]),lab="north")
    # plot!(p8,tt,rad2deg.(filt_out.te[i]),lab="east")
    # plot!(p8,tt,rad2deg.(filt_out.td[i]),lab="down")
    # show_plot && display(p8)
    # save_plot && png(p8,"tilt.png")

    # p9 = plot(xlab="time [min]",ylab="barometer aiding altitude [m]",dpi=dpi)
    # plot!(p9,tt,filt_out.ha[i])
    # show_plot && display(p9)
    # save_plot && png(p9,"barometer_aiding_altitude.png")

    # p10 = plot(xlab="time [min]",ylab="barometer aiding vertical accel [m/s^2]",dpi=dpi)
    # plot!(p10,tt,filt_out.ah[i])
    # show_plot && display(p10)
    # save_plot && png(p10,"barometer_aiding_vertical_accel.png")

    # p11 = plot(xlab="time [min]",ylab="accelerometer bias [m/s^2]",dpi=dpi)
    # plot!(p11,tt,filt_out.ax[i],lab="x")
    # plot!(p11,tt,filt_out.ay[i],lab="y")
    # plot!(p11,tt,filt_out.az[i],lab="z")
    # show_plot && display(p11)
    # save_plot && png(p11,"accelerometer_bias.png")

    # p12 = plot(xlab="time [min]",ylab="gyroscope bias [rad/s]",dpi=dpi)
    # plot!(p12,tt,filt_out.gx[i],lab="x")
    # plot!(p12,tt,filt_out.gy[i],lab="y")
    # plot!(p12,tt,filt_out.gz[i],lab="z")
    # show_plot && display(p12)
    # save_plot && png(p12,"gyroscope_bias.png")

    vel_plot ? (return (p1, p2, p3, p5, p6)) : (return (p1, p2, p3))
end # function plot_filt!

"""
    plot_filt(traj::Traj, ins::INS, filt_out::FILTout;
              vel_plot::Bool  = false,
              dpi::Int        = 200,
              Nmax::Int       = 5000,
              show_plot::Bool = true,
              save_plot::Bool = false)

Plot flights paths and latitude & longitude vs time.

**Arguments:**
- `traj`:      `Traj` trajectory struct
- `ins`:       `INS` inertial navigation system struct
- `filt_out`:  `FILTout` filter extracted output struct
- `vel_plot`:  (optional) if true, plot velocities
- `dpi`:       (optional) dots per inch (image resolution)
- `Nmax`:      (optional) maximum number of data points plotted
- `show_plot`: (optional) if true, show plot
- `save_plot`: (optional) if true, plots will be saved with default file names

**Returns:**
- `p1`: flight paths
- `p2`: latitudes  vs time
- `p3`: longitudes vs time
- `p5`: north velocities vs time
- `p6`: east  velocities vs time
"""
function plot_filt(traj::Traj, ins::INS, filt_out::FILTout;
                   vel_plot::Bool  = false,
                   dpi::Int        = 200,
                   Nmax::Int       = 5000,
                   show_plot::Bool = true,
                   save_plot::Bool = false)
    p1 = plot()
    plot_filt!(p1,traj,ins,filt_out;
               vel_plot  = vel_plot,
               dpi       = dpi,
               Nmax      = Nmax,
               show_plot = show_plot,
               save_plot = save_plot)
end # function plot_filt

"""
    plot_filt_err(traj::Traj, filt_out::FILTout, crlb_out::CRLBout;
                  vel_plot::Bool  = false,
                  dpi::Int        = 200,
                  Nmax::Int       = 5000,
                  show_plot::Bool = true,
                  save_plot::Bool = false)

Plot northing and easting errors vs time.

**Arguments:**
- `traj`:      `Traj` trajectory struct
- `filt_out`:  `FILTout` filter extracted output struct
- `crlb_out`:  `CRLBout` Cramér–Rao lower bound extracted output struct
- `vel_plot`:  (optional) if true, plot velocity errors
- `dpi`:       (optional) dots per inch (image resolution)
- `Nmax`:      (optional) maximum number of data points plotted
- `show_plot`: (optional) if true, show plot
- `save_plot`: (optional) if true, plots will be saved with default file names

**Returns:**
- `p2`: northing errors vs time
- `p3`: easting  errors vs time
- `p5`: north velocity errors vs time
- `p6`: east  velocity errors vs time
"""
function plot_filt_err(traj::Traj, filt_out::FILTout, crlb_out::CRLBout;
                       vel_plot::Bool  = false,
                       dpi::Int        = 200,
                       Nmax::Int       = 5000,
                       show_plot::Bool = true,
                       save_plot::Bool = false)

    i  = downsample(1:traj.N,Nmax)
    tt = (traj.tt[i] .- traj.tt[i][1]) / 60

    p2 = plot(xlab="time [min]",ylab="northing error [m]",dpi=dpi)
    plot!(p2,tt, filt_out.n_err[i],lab="filter error",color=:black,lw=2)
    plot!(p2,tt, filt_out.n_std[i],lab="filter 1-σ",color=:blue)
    plot!(p2,tt,-filt_out.n_std[i],lab=false,color=:blue)
    plot!(p2,tt, crlb_out.n_std[i],lab="CRLB 1-σ",color=:red,ls=:dash)
    plot!(p2,tt,-crlb_out.n_std[i],lab=false,color=:red,ls=:dash)
    show_plot && display(p2)
    save_plot && png(p2,"northing_error.png")

    p3 = plot(xlab="time [min]",ylab="easting error [m]",dpi=dpi)
    plot!(p3,tt, filt_out.e_err[i],lab="filter error",color=:black,lw=2)
    plot!(p3,tt, filt_out.e_std[i],lab="filter 1-σ",color=:blue)
    plot!(p3,tt,-filt_out.e_std[i],lab=false,color=:blue)
    plot!(p3,tt, crlb_out.e_std[i],lab="CRLB 1-σ",color=:red,ls=:dash)
    plot!(p3,tt,-crlb_out.e_std[i],lab=false,color=:red,ls=:dash)
    show_plot && display(p3)
    save_plot && png(p3,"easting_error.png")

    # p4 = plot(xlab="time [min]",ylab="altitude error [m]",dpi=dpi)
    # plot!(p4,tt, filt_out.alt_err[i],lab="filter error",color=:black,lw=2)
    # plot!(p4,tt, filt_out.alt_std[i],lab="filter 1-σ",color=:blue)
    # plot!(p4,tt,-filt_out.alt_std[i],lab=lab=false,color=:blue)
    # # plot!(p4,tt, crlb_out.alt_std[i],lab="CRLB 1-σ",color=:red,ls=:dash)
    # # plot!(p4,tt,-crlb_out.alt_std[i],lab=false,color=:red,ls=:dash)
    # show_plot && display(p4)
    # save_plot && png(p4,"altitude_error.png")

    if vel_plot

        p5 = plot(xlab="time [min]",ylab="north velocity error [m/s]",dpi=dpi) # , ylim=(-10,10))
        plot!(p5,tt, filt_out.vn_err[i],lab="filter error")
        plot!(p5,tt, filt_out.vn_std[i],lab="filter 1-σ",color=:black)
        plot!(p5,tt,-filt_out.vn_std[i],lab=false,color=:black)
        show_plot && display(p5)
        save_plot && png(p5,"north_velocity_error.png")

        p6 = plot(xlab="time [min]",ylab="east velocity error [m/s]",dpi=dpi) # , ylim=(-10,10))
        plot!(p6,tt, filt_out.ve_err[i],lab="filter error")
        plot!(p6,tt, filt_out.ve_std[i],lab="filter 1-σ",color=:black)
        plot!(p6,tt,-filt_out.ve_std[i],lab=false,color=:black)
        show_plot && display(p6)
        save_plot && png(p6,"east_velocity_error.png")

    end

    # p7 = plot(xlab="time [min]",ylab="down velocity error [m/s]",dpi=dpi)
    # plot!(p7,tt, filt_out.vd_err[i],lab="filter error")
    # plot!(p7,tt, filt_out.vd_std[i],lab="filter 1-σ",color=:black)
    # plot!(p7,tt,-filt_out.vd_std[i],lab=false,color=:black)
    # show_plot && display(p7)
    # save_plot && png(p7,"down_velocity_error.png")

    # p8 = plot(xlab="time [min]",ylab="north tilt error [deg]",dpi=dpi)
    # plot!(p8,tt, rad2deg.(filt_out.tn_err[i]),lab="filter error")
    # plot!(p8,tt, rad2deg.(filt_out.tn_std[i]),lab="filter 1-σ",color=:black)
    # plot!(p8,tt,-rad2deg.(filt_out.tn_std[i]),lab=false,color=:black)
    # show_plot && display(p8)
    # save_plot && png(p8,"north_tilt_error.png")

    # p9 = plot(xlab="time [min]",ylab="east tilt error [deg]",dpi=dpi)
    # plot!(p9,tt, rad2deg.(filt_out.te_err[i]),lab="filter error")
    # plot!(p9,tt, rad2deg.(filt_out.te_std[i]),lab="filter 1-σ",color=:black)
    # plot!(p9,tt,-rad2deg.(filt_out.te_std[i]),lab=false,color=:black)
    # show_plot && display(p9)
    # save_plot && png(p9,"east_tilt_error.png")

    # p10 = plot(xlab="time [min]",ylab="down tilt error [deg]",dpi=dpi)
    # plot!(p10,tt, rad2deg.(filt_out.td_err[i]),lab="filter error")
    # plot!(p10,tt, rad2deg.(filt_out.td_std[i]),lab="filter 1-σ",color=:black)
    # plot!(p10,tt,-rad2deg.(filt_out.td_std[i]),lab=false,color=:black)
    # show_plot && display(p10)
    # save_plot && png(p10,"down_tilt_error.png")

    # p11 = plot(xlab="time [min]",ylab="barometer aiding altitude σ [m]",dpi=dpi)
    # plot!(p11,tt, filt_out.ha_std[i],lab="filter 1-σ",color=:black)
    # plot!(p11,tt,-filt_out.ha_std[i],lab=false,color=:black)
    # show_plot && display(p11)
    # save_plot && png(p11,"barometer_aiding_altitude_σ.png")

    # p12 = plot(xlab="time [min]",ylab="barometer aiding vertical accel σ [m/s^2]",dpi=dpi)
    # plot!(p12,tt, filt_out.ah_std[i],lab="filter 1-σ",color=:black)
    # plot!(p12,tt,-filt_out.ah_std[i],lab=false,color=:black)
    # show_plot && display(p12)
    # save_plot && png(p12,"barometer_aiding_vertical_accel_σ.png")

    # p13 = plot(xlab="time [min]",ylab="accelerometer bias σ [m/s^2]",dpi=dpi)
    # plot!(p13,tt,filt_out.ax_std[i],lab="x")
    # plot!(p13,tt,filt_out.ay_std[i],lab="y")
    # plot!(p13,tt,filt_out.az_std[i],lab="z")
    # show_plot && display(p13)
    # save_plot && png(p13,"accelerometer_bias_σ.png")

    # p14 = plot(xlab="time [min]",ylab="gyroscope bias σ [rad/s]",dpi=dpi)
    # plot!(p14,tt,filt_out.gx_std[i],lab="x")
    # plot!(p14,tt,filt_out.gy_std[i],lab="y")
    # plot!(p14,tt,filt_out.gz_std[i],lab="z")
    # show_plot && display(p14)
    # save_plot && png(p14,"gyroscope_bias_σ.png")

    vel_plot ? (return (p2, p3, p5, p6)) : (return (p2, p3))
end # function plot_filt_err

"""
    plot_mag_map(path::Path, mag, itp_mapS;
                 lab::String        = "magnetometer",
                 order::Symbol      = :magmap,
                 dpi::Int           = 200,
                 Nmax::Int          = 5000,
                 detrend_data::Bool = true,
                 save_plot::Bool    = false)

Plot scalar magnetometer measurements vs map values.

**Arguments:**
- `path`:         `Path` struct, i.e., `Traj` trajectory struct, `INS` inertial navigation system struct, or `FILTout` filter extracted output struct
- `mag`:          scalar magnetometer measurements [nT]
- `itp_mapS`:     scalar map interpolation function
- `lab`:          (optional) magnetometer data (legend) label
- `order`:        (optional) plotting order {`:magmap`,`:mapmag`}
- `dpi`:          (optional) dots per inch (image resolution)
- `Nmax`:         (optional) maximum number of data points plotted
- `detrend_data`: (optional) if true, plot data will be detrended
- `save_plot`:    (optional) if true, `p1` will be saved with default file names

**Returns:**
- `p1`: scalar magnetometer measurements vs map values
"""
function plot_mag_map(path::Path, mag, itp_mapS;
                      lab::String        = "magnetometer",
                      order::Symbol      = :magmap,
                      dpi::Int           = 200,
                      Nmax::Int          = 5000,
                      detrend_data::Bool = true,
                      save_plot::Bool    = false)

    i  = downsample(1:path.N,Nmax)
    tt = (path.tt[i] .- path.tt[i][1]) / 60

    # custom itp_mapS from map cache, using median path position
    if typeof(itp_mapS) == Map_Cache # not the most accurate, but it runs
        itp_mapS = get_cached_map(itp_mapS,median(path.lat[i]),
                                           median(path.lon[i]),
                                           median(path.alt[i]))
    end

    if length(size(itp_mapS)) == 2
        map_val = itp_mapS.(path.lon[i],path.lat[i])
    elseif length(size(itp_mapS)) == 3
        map_val = itp_mapS.(path.lon[i],path.lat[i],path.alt[i])
    end

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

    save_plot && png(p1,"mag_vs_map.png")

    return (p1)
end # function plot_mag_map

"""
    plot_mag_map_err(path::Path, mag, itp_mapS;
                     lab::String        = "",
                     dpi::Int           = 200,
                     Nmax::Int          = 5000,
                     detrend_data::Bool = true,
                     save_plot::Bool    = false)

Plot scalar magnetometer measurement vs map value errors.

**Arguments:**
- `path`:         `Path` struct, i.e., `Traj` trajectory struct, `INS` inertial navigation system struct, or `FILTout` filter extracted output struct
- `mag`:          scalar magnetometer measurements [nT]
- `itp_mapS`:     scalar map interpolation function
- `lab`:          (optional) data (legend) label
- `dpi`:          (optional) dots per inch (image resolution)
- `Nmax`:         (optional) maximum number of data points plotted
- `detrend_data`: (optional) if true, plot data will be detrended
- `save_plot`:    (optional) if true, `p1` will be saved with default file names

**Returns:**
- `p1`: scalar magnetometer measurement vs map value errors
"""
function plot_mag_map_err(path::Path, mag, itp_mapS;
                          lab::String        = "",
                          dpi::Int           = 200,
                          Nmax::Int          = 5000,
                          detrend_data::Bool = true,
                          save_plot::Bool    = false)

    i  = downsample(1:path.N,Nmax)
    tt = (path.tt[i] .- path.tt[i][1]) / 60

    l  = detrend_data ? "detrended " : ""
    p1 = plot(xlab="time [min]",ylab=l*"magnetic signal error [nT]",dpi=dpi)

    f = detrend_data ? detrend : x -> x

    # custom itp_mapS from map cache, using median path position
    if typeof(itp_mapS) == Map_Cache # not the most accurate, but it runs
        itp_mapS = get_cached_map(itp_mapS,median(path.lat[i]),
                                           median(path.lon[i]),
                                           median(path.alt[i]))
    end

    if length(size(itp_mapS)) == 2
        map_val = itp_mapS.(path.lon[i],path.lat[i])
    elseif length(size(itp_mapS)) == 3
        map_val = itp_mapS.(path.lon[i],path.lat[i],path.alt[i])
    end

    plot!(p1,tt,f(mag[i] - map_val),lab=lab)

    save_plot && png(p1,"mag_map_err.png")

	err = round(std(mag[i] - map_val),digits=2)
    @info("mag-map error standard deviation = $err nT")

    return (p1)
end # function plot_mag_map_err

"""
    get_autocor(x::Vector, dt=0.1, dt_max=300.0)

Get autocorrelation of data (e.g., actual - expected measurements).

**Arguments:**
- `x`:      input data
- `dt`:     (optional) measurement time step [s]
- `dt_max`: (optional) maximum time step to evaluate [s]

**Returns:**
- `sigma`: standard deviation
- `tau`:   autocorrelation decay to e^-1 of `x` [s]
"""
function get_autocor(x::Vector, dt=0.1, dt_max=300.0)
    sigma = round(Int,std(x))
    dts   = 0:dt:dt_max
    lags  = round.(Int,dts/dt)
    x_ac  = autocor(x,lags)
    i     = findfirst(x_ac .< exp(-1))
    tau   = i !== nothing ? round(Int,dts[i]) : dt_max
    return (sigma, tau)
end # function get_autocor

"""
    plot_autocor(x::Vector, dt=0.1, dt_max=300.0; show_plot::Bool=false)

Plot autocorrelation of data (e.g., actual - expected measurements). Prints out
`σ` = standard deviation & `τ` = autocorrelation decay to e^-1 of `x`.

**Arguments:**
- `x`:         input data
- `dt`:        (optional) measurement time step [s]
- `dt_max`:    (optional) maximum time step to evaluate [s]
- `show_plot`: (optional) if true, `p1` will be shown

**Returns:**
- `p1`: plot of autocorrelation of `x`
"""
function plot_autocor(x::Vector, dt=0.1, dt_max=300.0; show_plot::Bool=false)

    (sigma,tau) = get_autocor(x,dt,dt_max) # 1002.17: σ = 5, τ ≈ 45

    @info("σ ≈ $sigma")
    @info("τ ≈ $tau")
    tau == dt_max && @info("τ not in range")

    dts  = 0:dt:dt_max
    lags = round.(Int,dts/dt)
    x_ac = autocor(x,lags)
    p1   = plot(dts,x_ac,lab=false);
    show_plot && display(p1)

    return (p1)
end # function plot_autocor

"""
    chisq_pdf(x, k::Int=1)

Probability density function (PDF) of the chi-square distribution.

**Arguments:**
- `x`: chi-square value
- `k`: degrees of freedom

**Returns:**
- `f`: probability density function (PDF) at `x` with `k`
"""
function chisq_pdf(x, k::Int=1)
    f = x > 0 ? x^(k/2-1) * exp(-x/2) / (2^(k/2) * gamma(k/2)) : 0
    return (float(f))
end # function chisq_pdf

"""
    chisq_cdf(x, k::Int=1)

Cumulative distribution function (CDF) of the chi-square distribution.

**Arguments:**
- `x`: chi-square value
- `k`: degrees of freedom

**Returns:**
- `F`: cumulative distribution function (CDF) at `x` with `k`
"""
function chisq_cdf(x, k::Int=1)
    F = x > 0 ? gamma_inc.(k/2,x/2)[1] : 0
    return (float(F))
end # function chisq_cdf

"""
    chisq_q(P=0.95, k::Int=1)

Quantile function (inverse CDF) of the chi-square distribution.

**Arguments:**
- `P`: percentile (`1 - p-value`), i.e., lower incomplete gamma function ratio
- `k`: degrees of freedom

**Returns:**
- `x`: chi-square value
"""
function chisq_q(P=0.95, k::Int=1)
    gamma_inc_inv(k/2,P,1-P)*2
end # function chisq_q

"""
    points_ellipse(P; clip=Inf, n::Int=61)

Internal helper function to create `x` and `y` confidence ellipse points for a
`2`x`2` covariance matrix.

**Arguments:**
- `P`:    `2`x`2` covariance matrix
- `clip`: (optional) clipping radius
- `n`:    (optional) number of confidence ellipse points

**Returns:**
- `x`: x-axis confidence ellipse points
- `y`: y-axis confidence ellipse points
"""
function points_ellipse(P; clip=Inf, n::Int=61)
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
    conf_ellipse!(p1, P;
                  μ                = zeros(2),
                  conf             = 0.95,
                  clip             = Inf,
                  n::Int           = 61,
                  lim              = nothing,
                  margin::Int      = 2,
                  lab::String      = "[m]",
                  axis::Bool       = true,
                  plot_eigax::Bool = false,
                  bg_color::Symbol = :white,
                  ce_color::Symbol = :black,
                  b_e              = gr())

Internal helper function to plot a confidence ellipse for a `2`x`2` covariance
matrix (2 degrees of freedom). Visualization of a 2D confidence interval.

**Arguments:**
- `p1`:         existing plot (e.g., map)
- `P`:          `2`x`2` covariance matrix
- `μ`:          (optional) confidence ellipse center (in same units as `P`)
- `conf`:       (optional) percentile {0:1}
- `clip`:       (optional) clipping radius (in same units as `P`)
- `n`:          (optional) number of confidence ellipse points
- `lim`:        (optional) `x` & `y` plotting limits (in same units as `P`)
- `margin`:     (optional) margin around plot [mm]
- `lab`:        (optional) axis labels
- `axis`:       (optional) if true, show axes
- `plot_eigax`: (optional) if true, show major and minor axes
- `bg_color`:   (optional) background color
- `ce_color`:   (optional) confidence ellipse color
- `b_e`:        (optional) plotting backend

**Returns:**
- `nothing`: confidence ellipse is plotted on `p1`
"""
function conf_ellipse!(p1, P;
                       μ                = zeros(2),
                       conf             = 0.95,
                       clip             = Inf,
                       n::Int           = 61,
                       lim              = nothing,
                       margin::Int      = 2,
                       lab::String      = "[m]",
                       axis::Bool       = true,
                       plot_eigax::Bool = false,
                       bg_color::Symbol = :white,
                       ce_color::Symbol = :black,
                       b_e              = gr())

    (eigval,eigvec) = eigen(P)
    eigax = I*sqrt.(Diagonal(eigval)) * eigvec'

    # check arguments
    xlim = ylim = lim === nothing  ? lim : (-lim,lim)
    xlab = ylab = axis             ? lab : nothing
    @assert all(real(eigval) .> 0) "P is not positive definite"
    @assert size(P) == (2,2)       "P is size $(size(P)) ≂̸ (2,2)"
    @assert length(μ) == 2         "μ is length $length(μ) ≂̸ 2"
    @assert 0 < conf < 1           "conf = $conf < 0 or conf = $conf > 1"

    k  = sqrt(chisq_q(conf,2)) # compute quantile for desired percentile
    b_e # backend
    if lim === nothing
        plot!(p1,xlab=xlab,ylab=ylab,
              legend=false,aspect_ratio=:equal,margin=margin*mm,
              axis=axis,xticks=axis,yticks=axis,grid=axis,bg=bg_color)
    else
        plot!(p1,xlim=xlim,ylim=ylim,xlab=xlab,ylab=ylab,
              legend=false,aspect_ratio=:equal,margin=margin*mm,
              axis=axis,xticks=axis,yticks=axis,grid=axis,bg=bg_color)
    end

    (x,y) = points_ellipse(P;clip=clip,n=n)
    plot!(p1, μ[1].+k*x, μ[2].+k*y, lw=2, color=ce_color)
    if plot_eigax
        plot!(p1,[-k,k]*eigax[1,1],[-k,k]*eigax[1,2],ls=:dash,lw=1,lc=:red)
        plot!(p1,[-k,k]*eigax[2,1],[-k,k]*eigax[2,2],ls=:dash,lw=2,lc=:red)
    end

end # function conf_ellipse!

"""
    conf_ellipse(P;
                 μ                = zeros(2),
                 conf             = 0.95,
                 clip             = Inf,
                 n::Int           = 61,
                 lim              = nothing,
                 margin::Int      = 2,
                 lab::String      = "[m]",
                 axis::Bool       = true,
                 plot_eigax::Bool = false,
                 bg_color::Symbol = :white,
                 ce_color::Symbol = :black,
                 b_e              = gr())

Internal helper function to plot a confidence ellipse for a `2`x`2` covariance
matrix (2 degrees of freedom). Visualization of a 2D confidence interval.

**Arguments:**
- `P`:          `2`x`2` covariance matrix
- `μ`:          (optional) confidence ellipse center (in same units as `P`)
- `conf`:       (optional) percentile {0:1}
- `clip`:       (optional) clipping radius (in same units as `P`)
- `n`:          (optional) number of confidence ellipse points
- `lim`:        (optional) `x` & `y` plotting limits (in same units as `P`)
- `margin`:     (optional) margin around plot [mm]
- `lab`:        (optional) axis labels
- `axis`:       (optional) if true, show axes
- `plot_eigax`: (optional) if true, show major and minor axes
- `bg_color`:   (optional) background color
- `ce_color`:   (optional) confidence ellipse color
- `b_e`:        (optional) plotting backend

**Returns:**
- `p1`: plot of confidence ellipse
"""
function conf_ellipse(P;
                      μ                = zeros(2),
                      conf             = 0.95,
                      clip             = Inf,
                      n::Int           = 61,
                      lim              = nothing,
                      margin::Int      = 2,
                      lab::String      = "[m]",
                      axis::Bool       = true,
                      plot_eigax::Bool = false,
                      bg_color::Symbol = :white,
                      ce_color::Symbol = :black,
                      b_e              = gr())
    b_e # backend
    p1 = plot()
    conf_ellipse!(p1, P;
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
    units_ellipse(P; conf_units::Symbol=:m, lat1=deg2rad(45))

Internal helper function to convert (position) confidence ellipse units for a
`2`x`2` covariance matrix.

**Arguments:**
- `P`:          `2`x`2` covariance matrix [rad]
- `conf_units`: (optional) confidence ellipse units {`:m`,`:ft`,`:deg`,`:rad`}
- `lat1`:       (optional) nominal latitude [rad], only used if `conf_units = :m` or `:ft`

**Returns:**
- `P`: `2`x`2` covariance matrix with converted units
"""
function units_ellipse(P; conf_units::Symbol=:m, lat1=deg2rad(45))
    @assert size(P,1) == 2 "P is size $(size(P)) ≂̸ (2,2)"
    @assert size(P,2) == 2 "P is size $(size(P)) ≂̸ (2,2)"

    P = deepcopy(P)

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
    units_ellipse(filt_res::FILTres, filt_out::FILTout; conf_units::Symbol=:m)

Internal helper function to convert (position) confidence ellipse units for a
`2`x`2` covariance matrix.

**Arguments:**
- `filt_res`:   `FILTres` filter results struct
- `filt_out`:   `FILTout` filter extracted output struct
- `conf_units`: (optional) confidence ellipse units {`:m`,`:ft`,`:deg`,`:rad`}

**Returns:**
- `P`: `2`x`2` covariance matrix with converted units
"""
function units_ellipse(filt_res::FILTres, filt_out::FILTout; conf_units::Symbol=:m)
    units_ellipse(deepcopy(filt_res.P[1:2,1:2,:]);
                  conf_units = conf_units,
                  lat1       = mean(filt_out.lat))
end # function units_ellipse

"""
    gif_ellipse(P,
                ellipse_gif::String = "conf_ellipse.gif",
                lat1                = deg2rad(45);
                dt                  = 0.1,
                di::Int             = 10,
                speedup::Int        = 60,
                conf_units::Symbol  = :m,
                μ                   = zeros(2),
                conf                = 0.95,
                clip                = Inf,
                n::Int              = 61,
                lim                 = 500,
                margin::Int         = 2,
                axis::Bool          = true,
                plot_eigax::Bool    = false,
                bg_color::Symbol    = :white,
                ce_color::Symbol    = :black,
                b_e                 = gr())

Create a (position) confidence ellipse GIF animation for a `2`x`2` (x`N`)
covariance matrix.

**Arguments:**
- `P`:           `2`x`2` (x`N`) covariance matrix
- `ellipse_gif`: (optional) path/name of confidence ellipse GIF file to save (`.gif` extension optional)
- `lat1`:        (optional) nominal latitude [rad], only used if `conf_units = :m` or `:ft`
- `dt`:          (optional) measurement time step [s]
- `di`:          (optional) GIF measurement interval (e.g., `di = 10` uses every 10th measurement)
- `speedup`:     (optional) GIF speedup (e.g., `speedup = 60` is 60x speed)
- `conf_units`:  (optional) confidence ellipse units {`:m`,`:ft`,`:deg`,`:rad`}
- `μ`:           (optional) confidence ellipse center [`conf_units`]
- `conf`:        (optional) percentile {0:1}
- `clip`:        (optional) clipping radius [`conf_units`]
- `n`:           (optional) number of confidence ellipse points
- `lim`:         (optional) `x` & `y` plotting limits [`conf_units`]
- `margin`:      (optional) margin around plot [mm]
- `axis`:        (optional) if true, show axes
- `plot_eigax`:  (optional) if true, show major and minor axes
- `bg_color`:    (optional) background color
- `ce_color`:    (optional) confidence ellipse color
- `b_e`:         (optional) plotting backend

**Returns:**
- `g1`: confidence ellipse GIF animation
"""
function gif_ellipse(P,
                     ellipse_gif::String = "conf_ellipse.gif",
                     lat1                = deg2rad(45);
                     dt                  = 0.1,
                     di::Int             = 10,
                     speedup::Int        = 60,
                     conf_units::Symbol  = :m,
                     μ                   = zeros(2),
                     conf                = 0.95,
                     clip                = Inf,
                     n::Int              = 61,
                     lim                 = 500,
                     margin::Int         = 2,
                     axis::Bool          = true,
                     plot_eigax::Bool    = false,
                     bg_color::Symbol    = :white,
                     ce_color::Symbol    = :black,
                     b_e                 = gr())

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

    ellipse_gif = add_extension(ellipse_gif,".gif")
    g1 = gif(a1,ellipse_gif;fps=1/dt/di*speedup);

    return (g1)
end # function gif_ellipse

"""
    gif_ellipse(filt_res::FILTres,
                filt_out::FILTout,
                ellipse_gif::String = "conf_ellipse.gif",
                map_map::Map        = mapS_null;
                dt                  = 0.1,
                di::Int             = 10,
                speedup::Int        = 60,
                conf_units::Symbol  = :m,
                μ                   = zeros(2),
                conf                = 0.95,
                clip                = Inf,
                n::Int              = 61,
                lim                 = 500,
                dpi::Int            = 200,
                margin::Int         = 2,
                axis::Bool          = true,
                plot_eigax::Bool    = false,
                bg_color::Symbol    = :white,
                ce_color::Symbol    = :black,
                map_color::Symbol   = :usgs,
                clims::Tuple        = (0,0),
                b_e                 = gr())

Create a (position) confidence ellipse GIF animation for a `2`x`2` (x`N`)
covariance matrix.

**Arguments:**
- `filt_res`:    `FILTres` filter results struct
- `filt_out`:    `FILTout` filter extracted output struct
- `ellipse_gif`: (optional) path/name of confidence ellipse GIF file to save (`.gif` extension optional)
- `map_map`:     (optional) `Map` magnetic anomaly map struct
- `dt`:          (optional) measurement time step [s]
- `di`:          (optional) GIF measurement interval (e.g., `di = 10` uses every 10th measurement)
- `speedup`:     (optional) GIF speedup (e.g., `speedup = 60` is 60x speed)
- `conf_units`:  (optional) confidence ellipse units {`:m`,`:ft`,`:deg`,`:rad`}
- `μ`:           (optional) confidence ellipse center [`conf_units`]
- `conf`:        (optional) percentile {0:1}
- `clip`:        (optional) clipping radius [`conf_units`]
- `n`:           (optional) number of confidence ellipse points
- `lim`:         (optional) `x` & `y` plotting limits [`conf_units`]
- `dpi`:         (optional) dots per inch (image resolution)
- `margin`:      (optional) margin around plot [mm]
- `axis`:        (optional) if true, show axes
- `plot_eigax`:  (optional) if true, show major and minor axes
- `bg_color`:    (optional) background color
- `ce_color`:    (optional) confidence ellipse color
- `map_color`:   (optional) filled contour color scheme {`:usgs`,`:gray`,`:gray1`,`:gray2`,`:plasma`,`:magma`}
- `clims`:       (optional) map color scale limits
- `b_e`:         (optional) plotting backend

**Returns:**
- `g1`: confidence ellipse gif
"""
function gif_ellipse(filt_res::FILTres,
                     filt_out::FILTout,
                     ellipse_gif::String = "conf_ellipse.gif",
                     map_map::Map        = mapS_null;
                     dt                  = 0.1,
                     di::Int             = 10,
                     speedup::Int        = 60,
                     conf_units::Symbol  = :m,
                     μ                   = zeros(2),
                     conf                = 0.95,
                     clip                = Inf,
                     n::Int              = 61,
                     lim                 = 500,
                     dpi::Int            = 200,
                     margin::Int         = 2,
                     axis::Bool          = true,
                     plot_eigax::Bool    = false,
                     bg_color::Symbol    = :white,
                     ce_color::Symbol    = :black,
                     map_color::Symbol   = :usgs,
                     clims::Tuple        = (0,0),
                     b_e                 = gr())

    dlat = get_step(map_map.yy)
    dlon = get_step(map_map.xx)
    dn   = dlat2dn(dlat,mean(map_map.yy))
    de   = dlon2de(dlon,mean(map_map.yy))

    if (dlon != 0) & (dlat != 0) & !isnan(dlon) & !isnan(dlat) & (clims == (0,0))
        num = ceil(Int,1.5*lim/minimum([dn,de]))
        y1  = findmin(abs.(map_map.yy.-minimum(filt_out.lat)))[2]
        y2  = findmin(abs.(map_map.yy.-maximum(filt_out.lat)))[2]
        x1  = findmin(abs.(map_map.xx.-minimum(filt_out.lon)))[2]
        x2  = findmin(abs.(map_map.xx.-maximum(filt_out.lon)))[2]
        (y1,y2)   = sort([y1,y2])
        (x1,x2)   = sort([x1,x2])
        (_,clims) = map_clims(map_cs(map_color),map_map.map[y1:y2,x1:x2,1])
    end

    P  = units_ellipse(filt_res,filt_out;conf_units=conf_units)
    a1 = Animation()

    for i = 1:di:size(P,3)

        if (dlon != 0) & (dlat != 0) & !isnan(dlon) & !isnan(dlat)
            xi   = findmin(abs.(map_map.xx.-filt_out.lon[i]))[2]
            yi   = findmin(abs.(map_map.yy.-filt_out.lat[i]))[2]
            xind = max(xi-num,1):min(xi+num,length(map_map.xx))
            yind = max(yi-num,1):min(yi+num,length(map_map.yy))
            p1   = plot_map(map_map.map[yind,xind,1],
                            map_map.xx[xind],map_map.yy[yind];
                            clims=clims,dpi=dpi,margin=margin,Nmax=10^10,
                            legend=false,axis=false,
                            map_color=map_color,bg_color=bg_color,
                            map_units=:rad,plot_units=conf_units,b_e=b_e);
        else
            p1 = plot();
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

    ellipse_gif = add_extension(ellipse_gif,".gif")
    g1 = gif(a1,ellipse_gif;fps=1/dt/di*speedup)

    return (g1)
end # function gif_ellipse
