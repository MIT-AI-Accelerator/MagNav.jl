"""
    create_P0(lat1 = deg2rad(45);
              init_pos_sigma   = 3.0,
              init_alt_sigma   = 0.001,
              init_vel_sigma   = 0.01,
              init_att_sigma   = deg2rad(0.01),
              ha_sigma         = 0.001,
              a_hat_sigma      = 0.01,
              acc_sigma        = 0.000245,
              gyro_sigma       = 0.00000000727,
              fogm_sigma       = 3.0,
              vec_sigma        = 1000.0,
              vec_states::Bool = false,
              fogm_state::Bool = true,
              P0_TL            = [])

Internal helper function to create P0, initial covariance matrix.

**Arguments:**
- `lat1`:           initial approximate latitude [rad]
- `init_pos_sigma`: (optional) initial position uncertainty [m]
- `init_alt_sigma`: (optional) initial altitude uncertainty [m]
- `init_vel_sigma`: (optional) initial velocity uncertainty [m/s]
- `init_att_sigma`: (optional) initial attitude uncertainty [rad]
- `ha_sigma`:       (optional) barometer aiding altitude bias [m]
- `a_hat_sigma`:    (optional) barometer aiding vertical accel bias [m/s^2]
- `acc_sigma`:      (optional) accelerometer bias [m/s^2]
- `gyro_sigma`:     (optional) gyroscope bias [rad/s]
- `fogm_sigma`:     (optional) FOGM catch-all bias [nT]
- `vec_sigma`:      (optional) vector magnetometer noise std dev
- `vec_states`:     (optional) if true, include vector magnetometer states
- `fogm_state`:     (optional) if true, include FOGM catch-all bias state
- `P0_TL`:          (optional) initial Tolles-Lawson covariance matrix

**Returns:**
- `P0`: initial covariance matrix
"""
function create_P0(lat1 = deg2rad(45);
                   init_pos_sigma   = 3.0,
                   init_alt_sigma   = 0.001,
                   init_vel_sigma   = 0.01,
                   init_att_sigma   = deg2rad(0.01),
                   ha_sigma         = 0.001,
                   a_hat_sigma      = 0.01,
                   acc_sigma        = 0.000245,
                   gyro_sigma       = 0.00000000727,
                   fogm_sigma       = 3.0,
                   vec_sigma        = 1000.0,
                   vec_states::Bool = false,
                   fogm_state::Bool = true,
                   P0_TL            = [])

    nx_TL   = size(P0_TL,1)
    nx_vec  = vec_states ? 3 : 0
    nx_fogm = fogm_state ? 1 : 0

    nx = 17 + nx_TL + nx_vec + nx_fogm

    P0 = zeros(Float64,nx,nx) # initial covariance matrix

    P0[1,1]   = dn2dlat(init_pos_sigma,lat1)^2
    P0[2,2]   = de2dlon(init_pos_sigma,lat1)^2
    P0[3,3]   = init_alt_sigma^2
    P0[4,4]   = init_vel_sigma^2
    P0[5,5]   = init_vel_sigma^2
    P0[6,6]   = init_vel_sigma^2
    P0[7,7]   = init_att_sigma^2
    P0[8,8]   = init_att_sigma^2
    P0[9,9]   = init_att_sigma^2
    P0[10,10] = ha_sigma^2
    P0[11,11] = a_hat_sigma^2
    P0[12,12] = acc_sigma^2
    P0[13,13] = acc_sigma^2
    P0[14,14] = acc_sigma^2
    P0[15,15] = gyro_sigma^2
    P0[16,16] = gyro_sigma^2
    P0[17,17] = gyro_sigma^2

    i1 = 18
    i2 = i1 + nx_TL
    i3 = i2 + nx_vec
    nx_TL   > 0 && (P0[i1:i2-1,i1:i2-1] = P0_TL)
    nx_vec  > 0 && (P0[i2:i3-1,i2:i3-1] = Diagonal(repeat([vec_sigma^2],nx_vec)))
    nx_fogm > 0 && (P0[i3     ,i3     ] = fogm_sigma^2)

    return (P0)
end # function create_P0

"""
    create_Qd(dt = 0.1;
              VRW_sigma        = 0.000238,
              ARW_sigma        = 0.000000581,
              baro_sigma       = 1.0,
              acc_sigma        = 0.000245,
              gyro_sigma       = 0.00000000727,
              fogm_sigma       = 3.0,
              vec_sigma        = 1000.0,
              TL_sigma         = [],
              baro_tau         = 3600.0,
              acc_tau          = 3600.0,
              gyro_tau         = 3600.0,
              fogm_tau         = 600.0,
              vec_states::Bool = false,
              fogm_state::Bool = true)

Internal helper function to create Qd, discrete time process/system noise matrix.

**Arguments:**
- `dt`:         measurement time step [s]
- `VRW_sigma`:  (optional) velocity random walk [m/s^2 /sqrt(Hz)]
- `ARW_sigma`:  (optional) angular  random walk [rad/s /sqrt(Hz)]
- `baro_sigma`: (optional) barometer bias [m]
- `acc_sigma`:  (optional) accelerometer bias [m/s^2]
- `gyro_sigma`: (optional) gyroscope bias [rad/s]
- `fogm_sigma`: (optional) FOGM catch-all bias [nT]
- `vec_sigma`:  (optional) vector magnetometer noise std dev
- `TL_sigma`:   (optional) Tolles-Lawson coefficients estimate std dev
- `baro_tau`:   (optional) barometer time constant [s]
- `acc_tau`:    (optional) accelerometer time constant [s]
- `gyro_tau`:   (optional) gyroscope time constant [s]
- `fogm_tau`:   (optional) FOGM catch-all time constant [s]
- `vec_states`: (optional) if true, include vector magnetometer states
- `fogm_state`: (optional) if true, include FOGM catch-all bias state

**Returns:**
- `Qd`: discrete time process/system noise matrix
"""
function create_Qd(dt = 0.1;
                   VRW_sigma        = 0.000238,
                   ARW_sigma        = 0.000000581,
                   baro_sigma       = 1.0,
                   acc_sigma        = 0.000245,
                   gyro_sigma       = 0.00000000727,
                   fogm_sigma       = 3.0,
                   vec_sigma        = 1000.0,
                   TL_sigma         = [],
                   baro_tau         = 3600.0,
                   acc_tau          = 3600.0,
                   gyro_tau         = 3600.0,
                   fogm_tau         = 600.0,
                   vec_states::Bool = false,
                   fogm_state::Bool = true)

    VRW_var    = VRW_sigma^2    # velocity random walk noise variance
    ARW_var    = ARW_sigma^2    # angular  random walk noise variance
    baro_drive = 2*baro_sigma^2 / baro_tau  # barometer      driving noise
    acc_drive  = 2*acc_sigma^2  / acc_tau   # accelerometer  driving noise
    gyro_drive = 2*gyro_sigma^2 / gyro_tau  # gyroscope      driving noise
    fogm_drive = 2*fogm_sigma^2 / fogm_tau  # FOGM catch-all driving noise
    TL_var     = TL_sigma.^2    # Tolles-Lawson coefficient noise variance
    vec_var    = vec_sigma^2    # vector magnetometer noise variance

    nx_TL   = size(TL_sigma,1)
    nx_vec  = vec_states ? 3 : 0
    nx_fogm = fogm_state ? 1 : 0

    Q = [repeat([1e-30  ],3);
         repeat([VRW_var],3);
         repeat([ARW_var],3);
         baro_drive;
         1e-30;
         repeat([acc_drive ],3);
         repeat([gyro_drive],3);
         zeros(Float64,nx_TL+nx_vec+nx_fogm)]

    i1 = 18
    i2 = i1 + nx_TL
    i3 = i2 + nx_vec
    nx_TL   > 0 && (Q[i1:i2-1] = vec(TL_var))
    nx_vec  > 0 && (Q[i2:i3-1] = repeat([vec_var],3))
    nx_fogm > 0 && (Q[i3     ] = fogm_drive)

    Qd = Diagonal(Q)*dt # discrete time process/system noise matrix

    return (Qd)
end # function create_Qd

"""
    create_model(dt = 0.1, lat1 = deg2rad(45);
                 init_pos_sigma   = 3.0,
                 init_alt_sigma   = 0.001,
                 init_vel_sigma   = 0.01,
                 init_att_sigma   = deg2rad(0.01),
                 meas_var         = 3.0^2,
                 VRW_sigma        = 0.000238,
                 ARW_sigma        = 0.000000581,
                 baro_sigma       = 1.0,
                 ha_sigma         = 0.001,
                 a_hat_sigma      = 0.01,
                 acc_sigma        = 0.000245,
                 gyro_sigma       = 0.00000000727,
                 fogm_sigma       = 3.0,
                 vec_sigma        = 1000.0,
                 TL_sigma         = [],
                 baro_tau         = 3600.0,
                 acc_tau          = 3600.0,
                 gyro_tau         = 3600.0,
                 fogm_tau         = 600.0,
                 vec_states::Bool = false,
                 fogm_state::Bool = true,
                 P0_TL            = [])

Create magnetic navigation model for EKF or MPF.

**Arguments:**
- `dt`:             measurement time step [s]
- `lat1`:           initial approximate latitude [rad]
- `init_pos_sigma`: (optional) initial position uncertainty [m]
- `init_alt_sigma`: (optional) initial altitude uncertainty [m]
- `init_vel_sigma`: (optional) initial velocity uncertainty [m/s]
- `init_att_sigma`: (optional) initial attitude uncertainty [rad]
- `meas_var`:       (optional) measurement (white) noise variance [nT^2]
- `VRW_sigma`:      (optional) velocity random walk [m/s^2 /sqrt(Hz)]
- `ARW_sigma`:      (optional) angular  random walk [rad/s /sqrt(Hz)]
- `baro_sigma`:     (optional) barometer bias [m]
- `ha_sigma`:       (optional) barometer aiding altitude bias [m]
- `a_hat_sigma`:    (optional) barometer aiding vertical accel bias [m/s^2]
- `acc_sigma`:      (optional) accelerometer bias [m/s^2]
- `gyro_sigma`:     (optional) gyroscope bias [rad/s]
- `fogm_sigma`:     (optional) FOGM catch-all bias [nT]
- `vec_sigma`:      (optional) vector magnetometer noise std dev
- `TL_sigma`:       (optional) Tolles-Lawson coefficients estimate std dev
- `baro_tau`:       (optional) barometer time constant [s]
- `acc_tau`:        (optional) accelerometer time constant [s]
- `gyro_tau`:       (optional) gyroscope time constant [s]
- `fogm_tau`:       (optional) FOGM catch-all time constant [s]
- `vec_states`:     (optional) if true, include vector magnetometer states
- `fogm_state`:     (optional) if true, include FOGM catch-all bias state
- `P0_TL`:          (optional) initial Tolles-Lawson covariance matrix

**Returns:**
- `P0`: initial covariance matrix
- `Qd`: discrete time process/system noise matrix
- `R`:  measurement (white) noise variance
"""
function create_model(dt = 0.1, lat1 = deg2rad(45);
                      init_pos_sigma   = 3.0,
                      init_alt_sigma   = 0.001,
                      init_vel_sigma   = 0.01,
                      init_att_sigma   = deg2rad(0.01),
                      meas_var         = 3.0^2,
                      VRW_sigma        = 0.000238,
                      ARW_sigma        = 0.000000581,
                      baro_sigma       = 1.0,
                      ha_sigma         = 0.001,
                      a_hat_sigma      = 0.01,
                      acc_sigma        = 0.000245,
                      gyro_sigma       = 0.00000000727,
                      fogm_sigma       = 3.0,
                      vec_sigma        = 1000.0,
                      TL_sigma         = [],
                      baro_tau         = 3600.0,
                      acc_tau          = 3600.0,
                      gyro_tau         = 3600.0,
                      fogm_tau         = 600.0,
                      vec_states::Bool = false,
                      fogm_state::Bool = true,
                      P0_TL            = [])

    P0 = create_P0(lat1;
                   init_pos_sigma = init_pos_sigma,
                   init_alt_sigma = init_alt_sigma,
                   init_vel_sigma = init_vel_sigma,
                   init_att_sigma = init_att_sigma,
                   ha_sigma       = ha_sigma,
                   a_hat_sigma    = a_hat_sigma,
                   acc_sigma      = acc_sigma,
                   gyro_sigma     = gyro_sigma,
                   fogm_sigma     = fogm_sigma,
                   vec_sigma      = vec_sigma,
                   vec_states     = vec_states,
                   fogm_state     = fogm_state,
                   P0_TL          = P0_TL)

    Qd = create_Qd(dt;
                   VRW_sigma  = VRW_sigma,
                   ARW_sigma  = ARW_sigma,
                   baro_sigma = baro_sigma,
                   acc_sigma  = acc_sigma,
                   gyro_sigma = gyro_sigma,
                   fogm_sigma = fogm_sigma,
                   vec_sigma  = vec_sigma,
                   TL_sigma   = TL_sigma,
                   baro_tau   = baro_tau,
                   acc_tau    = acc_tau,
                   gyro_tau   = gyro_tau,
                   fogm_tau   = fogm_tau,
                   vec_states = vec_states,
                   fogm_state = fogm_state)

    R = meas_var # measurement (white) noise variance [nT^2]

    return (P0, Qd, R)
end # function create_model

"""
    get_pinson(nx::Int, lat, vn, ve, vd, fn, fe, fd, Cnb;
               baro_tau         = 3600.0,
               acc_tau          = 3600.0,
               gyro_tau         = 3600.0,
               fogm_tau         = 600.0,
               vec_states::Bool = false,
               fogm_state::Bool = true,
               k1=3e-2, k2=3e-4, k3=1e-6)

Internal helper function to get `nx` x `nx` Pinson dynamics matrix.
States (errors) are:

|**Num**|**State**|**Units**|**Description**
|:--|:--|:--|:--
|  1 |   lat | rad   | latitude
|  2 |   lon | rad   | longitude
|  3 |   alt | m     | altitude
|  4 |    vn | m/s   | north velocity
|  5 |    ve | m/s   | east  velocity
|  6 |    vd | m/s   | down  velocity
|  7 |    tn | rad   | north tilt (attitude)
|  8 |    te | rad   | east  tilt (attitude)
|  9 |    td | rad   | down  tilt (attitude)
| 10 |    ha | m     | barometer aiding altitude
| 11 | a_hat | m/s^2 | barometer aiding vertical acceleration
| 12 |    ax | m/s^2 | x accelerometer
| 13 |    ay | m/s^2 | y accelerometer
| 14 |    az | m/s^2 | z accelerometer
| 15 |    gx | rad/s | x gyroscope
| 16 |    gy | rad/s | y gyroscope
| 17 |    gz | rad/s | z gyroscope
|end |     S | nT    | FOGM catch-all

**Arguments:**
- `nx`:         total state dimension
- `lat`:        latitude       [rad]
- `vn`:         north velocity [m/s]
- `ve`:         east  velocity [m/s]
- `vd`:         down  velocity [m/s]
- `fn`:         north specific force [m/s^2]
- `fe`:         east  specific force [m/s^2]
- `fd`:         down  specific force [m/s^2]
- `Cnb`:        direction cosine matrix (body to navigation) [-]
- `baro_tau`:   (optional) barometer      time constant [s]
- `acc_tau`:    (optional) accelerometer  time constant [s]
- `gyro_tau`:   (optional) gyroscope      time constant [s]
- `fogm_tau`:   (optional) FOGM catch-all time constant [s]
- `vec_states`: (optional) if true, include vector magnetometer states
- `fogm_state`: (optional) if true, include FOGM catch-all bias state
- `k1`:         (optional) barometer aiding constant [1/s]
- `k2`:         (optional) barometer aiding constant [1/s^2]
- `k3`:         (optional) barometer aiding constant [1/s^3]

**Returns:**
- `F`: `nx` x `nx` Pinson matrix
"""
function get_pinson(nx::Int, lat, vn, ve, vd, fn, fe, fd, Cnb;
                    baro_tau         = 3600.0,
                    acc_tau          = 3600.0,
                    gyro_tau         = 3600.0,
                    fogm_tau         = 600.0,
                    vec_states::Bool = false,
                    fogm_state::Bool = true,
                    k1=3e-2, k2=3e-4, k3=1e-6)

    # for 40 states:
    #  19-37 TL   [-]     Tolles-Lawson coefficients
    #  38    Bx   [nT]    x vector magnetometer
    #  39    By   [nT]    y vector magnetometer
    #  40    Bz   [nT]    z vector magnetometer

    tan_l = tan(lat)
    cos_l = cos(lat)
    sin_l = sin(lat)

    F = zeros(Float64,nx,nx)

    F[1,3]   = -vn / r_earth^2
    F[1,4]   =  1  / r_earth

    F[2,1]   =  ve * tan_l / (r_earth * cos_l)
    F[2,3]   = -ve         / (cos_l*r_earth^2)
    F[2,5]   =  1          / (r_earth * cos_l)

    F[3,3]   = -k1
    F[3,6]   = -1
    F[3,10]  =  k1

    F[4,1]   = -ve * (2*ω_earth*cos_l + ve / (r_earth*cos_l^2))
    F[4,3]   = (ve^2*tan_l - vn*vd) / r_earth^2
    F[4,4]   =  vd / r_earth
    F[4,5]   = -2*(ω_earth*sin_l + ve*tan_l / r_earth)
    F[4,6]   =  vn / r_earth
    F[4,8]   = -fd
    F[4,9]   =  fe

    F[5,1]   =  2*ω_earth*(vn*cos_l - vd*sin_l) + vn*ve / (r_earth * cos_l^2)
    F[5,3]   = -ve*((vn*tan_l + vd) / r_earth^2)
    F[5,4]   =  2*ω_earth*sin_l + ve * tan_l / r_earth
    F[5,5]   =      (vn*tan_l + vd) / r_earth
    F[5,6]   =  2*ω_earth*cos_l + ve / r_earth
    F[5,7]   =  fd
    F[5,9]   = -fn

    F[6,1]   =  2*ω_earth*ve*sin_l
    F[6,3]   =  (vn^2 + ve^2) / r_earth^2 + k2
    F[6,4]   = -2*vn / r_earth
    F[6,5]   = -2*(ω_earth*cos_l + ve / r_earth)
    F[6,7]   = -fe
    F[6,8]   =  fn
    F[6,10]  = -k2
    F[6,11]  =  1

    F[7,1]   = -ω_earth*sin_l
    F[7,3]   = -ve^2 / r_earth^2
    F[7,5]   =  1 / r_earth
    F[7,8]   = -ω_earth*sin_l - ve*tan_l / r_earth
    F[7,9]   =  vn / r_earth

    F[8,3]   =  vn / r_earth^2
    F[8,4]   = -1 / r_earth
    F[8,7]   =  ω_earth*sin_l + ve*tan_l / r_earth
    F[8,9]   =  ω_earth*cos_l + ve / r_earth

    F[9,1]   = -ω_earth*cos_l - ve / (r_earth*cos_l^2)
    F[9,3]   =  ve*tan_l / r_earth^2
    F[9,5]   = -tan_l / r_earth
    F[9,7]   = -vn / r_earth
    F[9,8]   = -ω_earth*cos_l - ve / r_earth

    F[10,10] = -1 / baro_tau

    F[11,3]  =  k3
    F[11,10] = -k3

    F[12,12] = -1 / acc_tau
    F[13,13] = -1 / acc_tau
    F[14,14] = -1 / acc_tau
    F[15,15] = -1 / gyro_tau
    F[16,16] = -1 / gyro_tau
    F[17,17] = -1 / gyro_tau

    F[4:6,12:14] =  Cnb
    F[7:9,15:17] = -Cnb

    if vec_states
        nx_fogm = fogm_state ? 1 : 0
        F[end-nx_fogm-2,end-nx_fogm-2] = -1e9
        F[end-nx_fogm-1,end-nx_fogm-1] = -1e9
        F[end-nx_fogm  ,end-nx_fogm  ] = -1e9
    end

    fogm_state && (F[end,end] = -1 / fogm_tau)

    return (F)
end # function get_pinson

"""
    get_Phi(nx::Int, lat, vn, ve, vd, fn, fe, fd, Cnb,
            baro_tau, acc_tau, gyro_tau, fogm_tau, dt;
            vec_states::Bool = false,
            fogm_state::Bool = true)

Internal helper function to get Pinson matrix exponential.

**Arguments:**
- `nx`:         total state dimension
- `lat`:        latitude       [rad]
- `vn`:         north velocity [m/s]
- `ve`:         east  velocity [m/s]
- `vd`:         down  velocity [m/s]
- `fn`:         north specific force [m/s^2]
- `fe`:         east  specific force [m/s^2]
- `fd`:         down  specific force [m/s^2]
- `Cnb`:        direction cosine matrix (body to navigation) [-]
- `baro_tau`:   barometer      time constant [s]
- `acc_tau`:    accelerometer  time constant [s]
- `gyro_tau`:   gyroscope      time constant [s]
- `fogm_tau`:   FOGM catch-all time constant [s]
- `dt`:         measurement time step [s]
- `vec_states`: (optional) if true, include vector magnetometer states
- `fogm_state`: (optional) if true, include FOGM catch-all bias state

**Returns:**
- `Phi`: Pinson matrix exponential
"""
function get_Phi(nx::Int, lat, vn, ve, vd, fn, fe, fd, Cnb,
                 baro_tau, acc_tau, gyro_tau, fogm_tau, dt;
                 vec_states::Bool = false,
                 fogm_state::Bool = true)

    exponential!(get_pinson(nx,lat,vn,ve,vd,fn,fe,fd,Cnb;
                            baro_tau   = baro_tau,
                            acc_tau    = acc_tau,
                            gyro_tau   = gyro_tau,
                            fogm_tau   = fogm_tau,
                            vec_states = vec_states,
                            fogm_state = fogm_state) * dt)

    # #* note: slightly more allocations & slightly slower
    # sparse(exponential!(get_pinson(nx,lat,vn,ve,vd,fn,fe,fd,Cnb;
    #                                baro_tau   = baro_tau,
    #                                acc_tau    = acc_tau,
    #                                gyro_tau   = gyro_tau,
    #                                fogm_tau   = fogm_tau,
    #                                vec_states = vec_states,
    #                                fogm_state = fogm_state) * dt))

end # function get_Phi

"""
    get_H(itp_mapS, x::Vector, lat, lon, alt;
          date       = get_years(2020,185),
          core::Bool = false)

Internal helper function to get expected magnetic measurement Jacobian (gradient here).

**Arguments:**
- `itp_mapS`: scalar map interpolation function (`f(lat,lon)` or `f(lat,lon,alt)`)
- `x`:        states [lat, lon, ... S]
- `lat`:      latitude  [rad]
- `lon`:      longitude [rad]
- `alt`:      altitude  [m]
- `date`:     (optional) measurement date (decimal year) for IGRF [yr]
- `core`:     (optional) if true, include core magnetic field in measurement

**Returns:**
- `H`: expected magnetic measurement Jacobian [nT/rad]
"""
function get_H(itp_mapS, x::Vector, lat, lon, alt;
               date       = get_years(2020,185),
               core::Bool = false)
    if core
        return ([(igrf_grad(lat+x[1],lon+x[2],alt+x[3];date=date) +
                 map_grad(itp_mapS,lat+x[1],lon+x[2],alt+x[3])); zero(x)[4:end-1]; 1])
    else
        return ([map_grad(itp_mapS,lat+x[1],lon+x[2],alt+x[3]) ; zero(x)[4:end-1]; 1])
    end
end # function get_H

"""
    get_h(itp_mapS, x::Array, lat, lon, alt;
          date       = get_years(2020,185),
          core::Bool = false)

Internal helper function to get expected magnetic measurement using magnetic
anomaly map interpolation function, FOGM catch-all, and optionally IGRF for
core magnetic field.

**Arguments:**
- `itp_mapS`: scalar map interpolation function (`f(lat,lon)` or `f(lat,lon,alt)`)
- `x`:        states [lat, lon, ... S]
- `lat`:      latitude  [rad]
- `lon`:      longitude [rad]
- `alt`:      altitude  [m]
- `date`:     (optional) measurement date (decimal year) for IGRF [yr]
- `core`:     (optional) if true, include core magnetic field in measurement

**Returns:**
- `h`: expected magnetic measurement [nT]
"""
function get_h(itp_mapS, x::Array, lat, lon, alt;
               date       = get_years(2020,185),
               core::Bool = false)

    map_val = itp_mapS.(lat.+x[1,:],lon.+x[2,:],alt.+x[3,:])

    if core
        return (map_val .+ x[end,:] .+ norm.(igrf.(date,alt.+x[3,:],lat.+x[1,:],
                                                   lon.+x[2,:],Val(:geodetic))))
    else
        return (map_val .+ x[end,:])
    end

end # function get_h

"""
    get_h(itp_mapS, der_mapS, x::Array, lat, lon, alt, map_alt;
          date       = get_years(2020,185),
          core::Bool = false)

Internal helper function to get expected magnetic measurement using magnetic
anomaly map interpolation function, FOGM catch-all, and optionally IGRF for
core magnetic field. Includes vertical derivative information, currently only
for ~constant HAE.

**Arguments:**
- `itp_mapS`: scalar map interpolation function (`f(lat,lon)` or `f(lat,lon,alt)`)
- `der_mapS`: scalar map vertical derivative map interpolation function (`f(lat,lon)` or (`f(lat,lon,alt)`)
- `x`:        states [lat, lon, ... S]
- `lat`:      latitude  [rad]
- `lon`:      longitude [rad]
- `alt`:      altitude  [m]
- `map_alt`:  map altitude [m]
- `date`:     (optional) measurement date (decimal year) for IGRF [yr]
- `core`:     (optional) if true, include core magnetic field in measurement

**Returns:**
- `h`: expected magnetic measurement [nT]
"""
function get_h(itp_mapS, der_mapS, x::Array, lat, lon, alt, map_alt;
               date       = get_years(2020,185),
               core::Bool = false)

    map_val = itp_mapS.(lat.+x[1,:],lon.+x[2,:],alt.+x[3,:])
    der_val = der_mapS.(lat.+x[1,:],lon.+x[2,:],alt.+x[3,:])

    if core
        return (map_val .+ x[end,:] .+ der_val .* (alt.+x[3,:] .- map_alt) .+
                norm.(igrf.(date,alt.+x[3,:],lat.+x[1,:],lon.+x[2,:],Val(:geodetic))))
    else
        return (map_val .+ x[end,:] .+ der_val .* (alt.+x[3,:] .- map_alt))
    end

end # function get_h

"""
    map_grad(itp_mapS, lat, lon, alt; δ = 1.0f-8)

Internal helper function to get local map gradient.

**Arguments:**
- `itp_mapS`: scalar map interpolation function (`f(lat,lon)` or `f(lat,lon,alt)`)
- `lat`:      latitude  [rad]
- `lon`:      longitude [rad]
- `alt`:      altitude  [m]
- `δ`:        (optional) finite difference map step size [rad]

**Returns:**
- `mapS_grad`: local scalar map gradient: δmap/δlat [nT/rad], δmap/δlon [nT/rad], δmap/δalt [nT/m]
"""
function map_grad(itp_mapS, lat, lon, alt; δ = 1.0f-8)
    dlat = dlon = δ
    dalt = dlat2dn(δ,lat)
    return ([(itp_mapS(lat+dlat,lon,alt) - itp_mapS(lat-dlat,lon,alt)) /2/dlat,
             (itp_mapS(lat,lon+dlon,alt) - itp_mapS(lat,lon-dlon,alt)) /2/dlon,
             (itp_mapS(lat,lon,alt+dalt) - itp_mapS(lat,lon,alt-dalt)) /2/dalt])
end # function map_grad

"""
    igrf_grad(lat, lon, alt; date = get_years(2020,185), δ = 1.0f-8)

Internal helper function to get core magnetic field gradient using IGRF model.

**Arguments:**
- `lat`:  latitude  [rad]
- `lon`:  longitude [rad]
- `alt`:  altitude  [m]
- `date`: (optional) measurement date (decimal year) for IGRF [yr]
- `δ`:    (optional) finite difference map step size [rad]

**Returns:**
- `core_grad`: local core magnetic field gradient: δmap/δlat [nT/rad], δmap/δlon [nT/rad], δmap/δalt [nT/m]
"""
function igrf_grad(lat, lon, alt; date = get_years(2020,185), δ = 1.0f-8)
    dlat = dlon = δ
    dalt = dlat2dn(δ,lat)
    return ([(norm(igrf(date,alt,lat+dlat,lon,Val(:geodetic))) -
              norm(igrf(date,alt,lat-dlat,lon,Val(:geodetic)))) /2/dlat,
             (norm(igrf(date,alt,lat,lon+dlon,Val(:geodetic))) -
              norm(igrf(date,alt,lat,lon-dlon,Val(:geodetic)))) /2/dlon,
             (norm(igrf(date,alt+dalt,lat,lon,Val(:geodetic))) -
              norm(igrf(date,alt-dalt,lat,lon,Val(:geodetic)))) /2/dalt])
end # function igrf_grad

"""
    fogm(sigma, tau, dt, N)

First-order Gauss-Markov stochastic process.
Represents unmeasureable time-correlated errors.

**Arguments:**
- `sigma`: FOGM catch-all bias
- `tau`:   FOGM catch-all time constant [s]
- `dt`:    measurement time step [s]
- `N`:     number of samples (instances)

**Returns:**
- `x`: FOGM data
"""
function fogm(sigma, tau, dt, N)

    x    = zeros(Float64,N)
    x[1] = sigma*randn(Float64)
    Phi  = exp(-dt/tau)
    Q    = 2*sigma^2/tau
    Qd   = Q*dt

    for i = 2:N
        x[i] = Phi*x[i-1] + sqrt(Qd)*randn(Float64)
    end

    return (x)
end # function fogm

"""
    chol(M)

Internal helper function to get the Cholesky factorization of matrix `M`,
ignoring roundoff errors that cause the matrix to be (slightly) non-symmetric
or indefinite. The matrix is assumed to be exactly Hermitian and only the
entries in the upper triangle are used.
"""
chol(M) = cholesky(Hermitian(collect(M))).U
