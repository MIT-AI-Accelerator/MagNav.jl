MagNav.jl flight data fields. Data sampled at 10Hz. Includes magnetic sensors, 
inertial navigation system, avionics, and electrical currents and voltages. 
(WGS-84) is elevation above WGS-84 ellipsoid.

================================================================================
       Field    Units    Description
--------------------------------------------------------------------------------
           N        -    number of instances
          DT        s    time step
        LINE        -    line number
         FLT        -    flight number
        TIME        s    fiducial seconds past midnight UTC
       UTM_X        m    x-coordinate, WGS-84 UTM zone 18N
       UTM_Y        m    y-coordinate, WGS-84 UTM zone 18N
       UTM_Z        m    z-coordinate, GPS altitude (WGS-84)
       MSL_Z        m    z-coordinate, GPS altitude above EGM2008 Geoid
         LAT      deg    latitude, WGS-84
        LONG      deg    longitude, WGS-84
        BARO        m    barometric altimeter
       RADAR        m    filtered radar altimeter
        TOPO        m    radar topography (WGS-84)
         DEM        m    digital elevation model from SRTM (WGS-84)
       DRAPE        m    planned survey drape (WGS-84)
       PITCH      deg    INS-computed aircraft pitch
        ROLL      deg    INS-computed aircraft roll
     AZIMUTH      deg    INS-computed aircraft yaw
     DIURNAL       nT    measured diurnal
    COMPMAG1       nT    Mag 1: compensated magnetic field
     LAGMAG1       nT    Mag 1: lag-corrected magnetic field
      DCMAG1       nT    Mag 1: diurnal-corrected magnetic field
    IGRFMAG1       nT    Mag 1: IGRF & diurnal-corrected magnetic field
  UNCOMPMAG1       nT    Mag 1: uncompensated magnetic field
  UNCOMPMAG2       nT    Mag 2: uncompensated magnetic field
  UNCOMPMAG3       nT    Mag 3: uncompensated magnetic field
  UNCOMPMAG4       nT    Mag 4: uncompensated magnetic field
  UNCOMPMAG5       nT    Mag 5: uncompensated magnetic field
     FLUXA_X       nT    Flux A: fluxgate x-axis
     FLUXA_Y       nT    Flux A: fluxgate y-axis
     FLUXA_Z       nT    Flux A: fluxgate z-axis
   FLUXA_TOT       nT    Flux A: fluxgate total
     FLUXB_X       nT    Flux B: fluxgate x-axis
     FLUXB_Y       nT    Flux B: fluxgate y-axis
     FLUXB_Z       nT    Flux B: fluxgate z-axis
   FLUXB_TOT       nT    Flux B: fluxgate total
     FLUXC_X       nT    Flux C: fluxgate x-axis
     FLUXC_Y       nT    Flux C: fluxgate y-axis
     FLUXC_Z       nT    Flux C: fluxgate z-axis
   FLUXC_TOT       nT    Flux C: fluxgate total
     FLUXD_X       nT    Flux D: fluxgate x-axis
     FLUXD_Y       nT    Flux D: fluxgate y-axis
     FLUXD_Z       nT    Flux D: fluxgate z-axis
   FLUXD_TOT       nT    Flux D: fluxgate 
     OGS_MAG       nT    OGS survey diurnal-corrected, levelled, magnetic field
     OGS_HGT        m    OGS survey, GPS altitude (WGS-84)
   INS_ACC_X    m/s^2    INS x-acceleration
   INS_ACC_Y    m/s^2    INS y-acceleration
   INS_ACC_Z    m/s^2    INS z-acceleration
  INS_WANDER      rad    INS-computed wander angle (ccw from north)
     INS_LAT      rad    INS-computed latitude
     INS_LON      rad    INS-computed longitude
     INS_HGT        m    INS-computed altitude (WGS-84)
   INS_VEL_N      m/s    INS-computed north velocity
   INS_VEL_W      m/s    INS-computed west velocity
   INS_VEL_V      m/s    INS-computed vertical (up) velocity
     PITCHRT    deg/s    avionics-computed pitch rate
      ROLLRT    deg/s    avionics-computed roll rate
       YAWRT    deg/s    avionics-computed yaw rate
    LONG_ACC        g    avionics-computed longitudinal (forward) acceleration
     LAT_ACC        g    avionics-computed lateral (starboard) acceleration
    NORM_ACC        g    avionics-computed normal (vertical) acceleration
     TRUE_AS      m/s    avionics-computed true airspeed
     PITOT_P      kPa    avionics-computed pitot pressure
    STATIC_P      kPa    avionics-computed static pressure
       TOT_P      kPa    avionics-computed total pressure
    CUR_COM1        A    current sensor: aircraft radio 1
    CUR_ACHi        A    current sensor: air conditioner fan high
    CUR_ACLo        A    current sensor: air conditioner fan low
    CUR_TANK        A    current sensor: cabin fuel pump
    CUR_FLAP        A    current sensor: flap motor
    CUR_STRB        A    current sensor: strobe lights
  CUR_SRVO_O        A    current sensor: INS outer servo
  CUR_SRVO_M        A    current sensor: INS middle servo
  CUR_SRVO_I        A    current sensor: INS inner servo
    CUR_IHTR        A    current sensor: INS heater
   CUR_ACPWR        A    current sensor: aircraft power
  CUR_OUTPWR        A    current sensor: system output power
    CUR_BAT1        A    current sensor: battery 1
    CUR_BAT2        A    current sensor: battery 2
     V_ACPWR        V    voltage sensor: aircraft power
    V_OUTPWR        V    voltage sensor: system output power
      V_BAT1        V    voltage sensor: battery 1
      V_BAT2        V    voltage sensor: battery 2
      V_RESp        V    voltage sensor: resolver board (+)
      V_RESn        V    voltage sensor: resolver board (-)
     V_BACKp        V    voltage sensor: backplane (+)
     V_BACKn        V    voltage sensor: backplane (-)
     V_GYRO1        V    voltage sensor: gyroscope 1
     V_GYRO2        V    voltage sensor: gyroscope 2
      V_ACCp        V    voltage sensor: INS accelerometers (+)
      V_ACCn        V    voltage sensor: INS accelerometers (-)
     V_BLOCK        V    voltage sensor: block
      V_BACK        V    voltage sensor: backplane
     V_SERVO        V    voltage sensor: servos
      V_CABT        V    voltage sensor: cabinet
       V_FAN        V    voltage sensor: air conditioner fan
================================================================================

Notes on specific flight data fields:

RADAR
- unavailable at some times due to flown altitude exceeding instrument range

PITCH, ROLL, AZIMUTH
- convention: yaw clockwise from north, then pitch up, then roll to starboard

OGS_MAG & OGS_ALT
- included for reference only
- sampled from original Ontario Geological Survey flown in 2013
- unavailable at some times due to flown position not over original OGS survey

INS_ACC_X, INS_ACC_Y, INS_ACC_Z
- AIRGrav (airborne gravimeter) system (http://www.sgl.com/Gravity.html)
- 3-axis gimballed INS, run in local-level, wander-angle mechanization
- corrected for temperature, misalignment, and scale factor
- digitizers used for accelerometers are sigma-delta type and require low-pass 
  filtering to increase resolution
- INS_ACC_X: nominally horizontal, ccw from north by INS_WANDER [rad]
- INS_ACC_Y: nominally horizontal, 90 [deg] ccw from INS_ACC_X
- INS_ACC_Z: nominally vertical (up), includes gravity term (~navigation frame)

INS_HGT
- combined GPS/inertial altitude computed in real time, contains lag error

LONG_ACC, LAT_ACC, NORM_ACC
- Garmin avionics module 010-0G600-00 (https://www.garmin.com/en-US/p/6427)
- NORM_ACC does not include gravity term (body frame)

================================================================================

Magnetometer/Fluxgate positions in reference to front seat rail [m]
- positive directions: forward (X), port (Y), up (Z)

  Sensor   Description                      X       Y       Z
-------------------------------------------------------------------
   Mag 1   Tail stinger                  -12.01    0       1.37
   Mag 2   Front cabin, aft of cockpit    -0.60   -0.36    0
   Mag 3   Mid cabin, near INS            -1.28   -0.36    0
   Mag 4   Rear cabin, floor              -3.53    0       0
   Mag 5   Rear cabin, ceiling            -3.79    0       1.20
  Flux A   Mid cabin, near fuel tank      -3.27   -0.60    0
  Flux B   Tail, base of stinger          -8.92    0       0.96
  Flux C   Rear cabin, port               -4.06    0.42    0
  Flux D   Rear cabin, starboard          -4.06   -0.42    0
