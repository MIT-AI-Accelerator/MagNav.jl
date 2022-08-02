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
             date           = 2020+185/366,
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
- `itp_mapS`:  scalar map grid interpolation
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
- `y_norms`:   (optional) Tuple of `y` normalizations, i.e. `(y_bias,y_scale)`
- `terms`:     (optional) Tolles-Lawson terms to use {`:permanent`,`:induced`,`:eddy`,`:bias`}
- `flux`:      (optional) `MagV` vector magnetometer measurement struct
- `x0_TL`:     (optional) initial Tolles-Lawson coefficient states
- `extract`:   (optional) if true, extract output structs
- `run_crlb`:  (optional) if true, compute the Cramér–Rao lower bound (CRLB)

**Returns:**
- if `extract == true`  & `run_crlb = true`
    - `crlb_out`: `CRLBout` Cramér–Rao lower bound extracted output struct
    - `ins_out`:  `INSout`  inertial navigation system extracted output struct
    - `filt_out`: `FILTout` filter extracted output struct
- if `extract == true`  & `run_crlb = false`
    - `filt_out`: `FILTout` filter extracted output struct
- if `extract == false` & `run_crlb = true`
    - `filt_res`: `FILTres` filter results struct
    - `crlb_P`:   Cramér–Rao lower bound non-linear covariance matrix
- if `extract == false` & `run_crlb = false`
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
                  date           = 2020+185/366,
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

Run multiple filter models and print results (nothing returned.)

**Arguments:**
- `filt_type`: multiple filter types, e.g. [`:ekf`,`:ekf_online_nn`]
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
                  date       = 2020+185/366,
                  core::Bool = false,
                  der_mapS   = map_itp(zeros(2,2),[-pi,pi],[-pi/2,pi/2]),
                  map_alt    = 0,
                  nn_x       = nothing,
                  m          = nothing,
                  y_norms    = nothing,
                  terms      = [:permanent,:induced,:eddy,:bias],
                  flux::MagV = MagV([0.0],[0.0],[0.0],[0.0]),
                  x0_TL      = ones(eltype(P0),19))

    for i in eachindex(filt_type)
        @info ("running $(filt_type[i]) filter")
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
