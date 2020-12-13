
File format as follows:

        Name    Size    Units     Null       Description

        LINE     8        -        -         Line Number XXXX.YY where XXXX is line number and YY is segment number
         FLT     5        -        -         Flight Number
        YEAR     5        -        -         Year
         DOY     4        -        -         Day of year
        TIME     9        s        *         Fiducial Seconds Past Midnight UTC
       UTM-X    11        m        *         X coordinate, WGS-84 UTM ZONE 18N
       UTM-Y    11        m        *         Y coordinate, WGS-84 UTM ZONE 18N
       UTM-Z     8        m        *         Z coordinate, GPS Elevation (above WGS-84 Ellipsoid)
       MSL-Z     8        m        *         Z coordinate, GPS Elevation (above EGM2008 Geoid)
         LAT    13      deg        *         Latitude, WGS-84
        LONG    13      deg        *         Longitude, WGS-84
        BARO     8        m        *         Barometric Altimeter
       RADAR     8        m        *         Filtered Radar Altimeter*
        TOPO     8        m        *         Radar Topography (above WGS-84 Ellipsoid)
         DEM     8        m        *         Digital Elevation Model from SRTM (above WGS-84 Ellipsoid)
       DRAPE     8        m        *         Planned Survey Drape (above WGS-84 Ellipsoid)
       PITCH     7      deg        *         INS computed aircraft pitch
        ROLL     7      deg        *         INS computed aircraft roll
     AZIMUTH     7      deg        *         INS computed aircraft azimuth
     DIURNAL    11       nT        *         Measured Diurnal
    COMPMAG1    11       nT        *         Mag 1: Compensated Airborne Magnetic Field
     LAGMAG1    11       nT        *         Mag 1: Lag Corrected Airborne Magnetic Field
      DCMAG1    11       nT        *         Mag 1: Diurnal Corrected Airborne Magnetic Field
    IGRFMAG1    11       nT        *         Mag 1: IGRF and Diurnal Corrected Airborne Magnetic Field
  UNCOMPMAG1    11       nT        *         Mag 1: Uncompensated Airborne Magnetic Field
  UNCOMPMAG2    11       nT        *         Mag 2: Uncompensated Airborne Magnetic Field
  UNCOMPMAG3    11       nT        *         Mag 3: Uncompensated Airborne Magnetic Field
  UNCOMPMAG4    11       nT        *         Mag 4: Uncompensated Airborne Magnetic Field
  UNCOMPMAG5    11       nT        *         Mag 5: Uncompensated Airborne Magnetic Field
     FLUXB_X    11       nT        *         Flux B: Fluxgate X-axis
     FLUXB_Y    11       nT        *         Flux B: Fluxgate Y-axis
     FLUXB_Z    11       nT        *         Flux B: Fluxgate Z-axis
   FLUXB_TOT    11       nT        *         Flux B: Fluxgate Total
     FLUXC_X    11       nT        *         Flux C: Fluxgate X-axis
     FLUXC_Y    11       nT        *         Flux C: Fluxgate Y-axis
     FLUXC_Z    11       nT        *         Flux C: Fluxgate Z-axis
   FLUXC_TOT    11       nT        *         Flux C: Fluxgate Total
     FLUXD_X    11       nT        *         Flux D: Fluxgate X-axis
     FLUXD_Y    11       nT        *         Flux D: Fluxgate Y-axis
     FLUXD_Z    11       nT        *         Flux D: Fluxgate Z-axis
   FLUXD_TOT    11       nT        *         Flux D: Fluxgate Total
     OGS_MAG    11       nT        *         OGS Survey Diurnal Corrected, Levelled, Airborne Magnetic Field**
     OGS_HGT     8        m        *         OGS Survey Flown Height, GPS Elevation (above WGS-84 Ellipsoid)**
   INS_ACC_X    12     m/s2        *         INS X Acceleration
   INS_ACC_Y    12     m/s2        *         INS Y Acceleration
   INS_ACC_Z    12     m/s2        *         INS Z Acceleration
  INS_WANDER    12      rad        *         INS Computed wander angle (ccw from North)
     INS_LAT    13      rad        *         INS Computed Latitude
     INS_LON    13      rad        *         INS Computed Longitude
     INS_HGT     8        m        *         INS Computed Height (above WGS-84 Ellipsoid)
   INS_VEL_N    12      m/s        *         INS Computed North Velocity
   INS_VEL_W    12      m/s        *         INS Computed West Velocity
   INS_VEL_V    12      m/s        *         INS Computed Vertical Velocity
     PITCHRT    10    deg/s        *         Avionics Computed Pitch Rate
      ROLLRT    10    deg/s        *         Avionics Computed Roll Rate
       YAWRT    10    deg/s        *         Avionics Computed Yaw Rate
    LONG_ACC    10     m/s2        *         Avionics Computed Longitudinal Acceleration
     LAT_ACC    10     m/s2        *         Avionics Computed Lateral Acceleration
    NORM_ACC    10     m/s2        *         Avionics Computed Normal (Vertical) Acceleration
     TRUE_AS    10      m/s        *         Avionics Computed True Airspeed
     PITOT_P    10      kPa        *         Avionics Computed Pitot (Impact) Pressure
    STATIC_P    10      kPa        *         Avionics Computed Static Pressure
       TOT_P    10      kPa        *         Avionics Computed Total Pressure
    CUR_COM1    10        A        *         Current Sensor: COM1 Aircraft Radio
    CUR_ACHi    10        A        *         Current Sensor: Air Conditioner Fan High
    CUR_ACLo    10        A        *         Current Sensor: Air Conditioner Fan Low
    CUR_TANK    10        A        *         Current Sensor: Cabin Fuel Pump
    CUR_FLAP    10        A        *         Current Sensor: Flap Motor
    CUR_STRB    10        A        *         Current Sensor: Strobe Lights
  CUR_SRVO_O    10        A        *         Current Sensor: INS Outer Servo
  CUR_SRVO_M    10        A        *         Current Sensor: INS Middle Servo
  CUR_SRVO_I    10        A        *         Current Sensor: INS Inner Servo
    CUR_IHTR    10        A        *         Current Sensor: INS Heater
   CUR_ACPWR    10        A        *         Current Sensor: Aircraft Power
  CUR_OUTPWR    10        A        *         Current Sensor: System Output Power
    CUR_BAT1    10        A        *         Current Sensor: Battery 1
    CUR_BAT2    10        A        *         Current Sensor: Battery 2
     V_ACPWR    10        V        *         Voltage Sensor: Aircraft Power
    V_OUTPWR    10        V        *         Voltage Sensor: System Output Power
      V_BAT1    10        V        *         Voltage Sensor: Battery 1
      V_BAT2    10        V        *         Voltage Sensor: Battery 2
      V_RESp    10        V        *         Voltage Sensor: Resolver Board +
      V_RESn    10        V        *         Voltage Sensor: Resolver Board -
     V_BACKp    10        V        *         Voltage Sensor: Backplane +
     V_BACKn    10        V        *         Voltage Sensor: Backplane -
     V_GYRO1    10        V        *         Voltage Sensor: Gyro 1
     V_GYRO2    10        V        *         Voltage Sensor: Gyro 2
      V_ACCp    10        V        *         Voltage Sensor: INS Accelerometers +
      V_ACCn    10        V        *         Voltage Sensor: INS Accelerometers -
     V_BLOCK    10        V        *         Voltage Sensor: Block
      V_BACK    10        V        *         Voltage Sensor: Backplane
     V_SERVO    10        V        *         Voltage Sensor: Servos
      V_CABT    10        V        *         Voltage Sensor: Cabinet
       V_FAN    10        V        *         Voltage Sensor: Cooling Fan
==========================================================================================================

Notes:

The file contains 10Hz data recordings from magnetic sensors, inertial navigation system, avionics, electrical power

*   No altimeter data is available for some time ranges due to altitudes being greater than the range of the instrument.
**  OGS Survey mag and flown height are included for reference. The data provided here have been sampled from the original
    data as provided from the Ontario Geological Survey web pages. It is only available for portions of flight within the
    previously flown survey area.  The magnetic data grids sampled are: REMAG83.GRD for the Western ("Renfrew") section,
    and EOMAG83.GRD for the Eastern ("Eastern Ontario") section.  The height data was obtained from the OGS line database,
    then re-gridded and later re-sampled to flown lines internally by SGL.
    Flown and compiled in 2013 by Goldack Airborne Surveys.


Magnetometer/Fluxgate Definitions and Position:
Positions are defined by a vector measured from the aircraft reference point (front seat rail) to the instrument.
X is positive in aircraft fwd direction
Y is positive to port (left facing fwd)
Z is positive in up direction

Mag 1   Located in tail stinger                    (X=-12.0135, Y=0.0038, Z=1.3671)
Mag 2   Located in front cabin just aft of cockpit (X=-0.60, Y=-0.36, Z=0)
Mag 3   Located mid cabin next to INS              (X=-1.28, Y=-0.36, Z=0)
Mag 4   Located at rear of cabin on floor          (X=-3.53, Y=0, Z=0)
Mag 5   Located at rear of cabin on ceiling        (X=-3.79, Y=0, Z=1.2)

Flux B  Located in tail at base of stinger         (X=-8.92, Y=0, Z=0.96)
Flux C  Located at rear of cabin port side         (X=-4.06, Y=0.42, Z=0)
Flux D  Located at rear of cabin starboard side    (X=-4.06, Y=-0.42, Z=0)
