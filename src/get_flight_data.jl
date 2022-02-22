"""
    get_flight_data(h5_file::String)

Get xyz data from saved xyz HDF5 file.

**Arguments:**
- `h5_file`: path/name of saved xyz HDF5 file

**Returns:**
- `XYZ`: XYZ data struct
"""
function get_flight_data(h5_file::String; tt_sort::Bool=true)

    println("")
    println("> Reading in file: ", h5_file)

    h5_data     = h5open(h5_file,"r")

    N           = h5read(h5_file, "N")
    DT          = 0.1 # h5read(h5_file, "dt") # error in h5_file
    
    if tt_sort
        ind = sortperm(read_check(N,h5_data,h5_file, "tt"))
    else
        ind = 1:N
    end

    LINE        = read_check(N,h5_data,h5_file, "line")[ind]
    FLT         = read_check(N,h5_data,h5_file, "flight")[ind]
    TIME        = read_check(N,h5_data,h5_file, "tt")[ind]
    UTM_X       = read_check(N,h5_data,h5_file, "utm_x")[ind]
    UTM_Y       = read_check(N,h5_data,h5_file, "utm_y")[ind]
    UTM_Z       = read_check(N,h5_data,h5_file, "utm_z")[ind]
    MSL_Z       = read_check(N,h5_data,h5_file, "msl")[ind]
    LAT         = read_check(N,h5_data,h5_file, "lat")[ind]
    LONG        = read_check(N,h5_data,h5_file, "lon")[ind]

    BARO        = read_check(N,h5_data,h5_file, "baro")[ind]
    RADAR       = read_check(N,h5_data,h5_file, "radar")[ind]
    TOPO        = read_check(N,h5_data,h5_file, "topo")[ind]
    DEM         = read_check(N,h5_data,h5_file, "dem")[ind]
    DRAPE       = read_check(N,h5_data,h5_file, "drape")[ind]

    PITCH       = read_check(N,h5_data,h5_file, "ins_pitch")[ind]
    ROLL        = read_check(N,h5_data,h5_file, "ins_roll")[ind]
    AZIMUTH     = read_check(N,h5_data,h5_file, "ins_yaw")[ind]
    DIURNAL     = read_check(N,h5_data,h5_file, "diurnal")[ind]

    COMPMAG1    = read_check(N,h5_data,h5_file, "mag_1_c")[ind]
    LAGMAG1     = read_check(N,h5_data,h5_file, "mag_1_lag")[ind]
    DCMAG1      = read_check(N,h5_data,h5_file, "mag_1_dc")[ind]
    IGRFMAG1    = read_check(N,h5_data,h5_file, "mag_1_igrf")[ind]
    UNCOMPMAG1  = read_check(N,h5_data,h5_file, "mag_1_uc")[ind]

    UNCOMPMAG2  = read_check(N,h5_data,h5_file, "mag_2_uc")[ind]
    UNCOMPMAG3  = read_check(N,h5_data,h5_file, "mag_3_uc")[ind]
    UNCOMPMAG4  = read_check(N,h5_data,h5_file, "mag_4_uc")[ind]
    UNCOMPMAG5  = read_check(N,h5_data,h5_file, "mag_5_uc")[ind]

    FLUXA_X     = read_check(N,h5_data,h5_file, "flux_a_x")[ind]
    FLUXA_Y     = read_check(N,h5_data,h5_file, "flux_a_y")[ind]
    FLUXA_Z     = read_check(N,h5_data,h5_file, "flux_a_z")[ind]
    FLUXA_TOT   = read_check(N,h5_data,h5_file, "flux_a_t")[ind]

    FLUXB_X     = read_check(N,h5_data,h5_file, "flux_b_x")[ind]
    FLUXB_Y     = read_check(N,h5_data,h5_file, "flux_b_y")[ind]
    FLUXB_Z     = read_check(N,h5_data,h5_file, "flux_b_z")[ind]
    FLUXB_TOT   = read_check(N,h5_data,h5_file, "flux_b_t")[ind]

    FLUXC_X     = read_check(N,h5_data,h5_file, "flux_c_x")[ind]
    FLUXC_Y     = read_check(N,h5_data,h5_file, "flux_c_y")[ind]
    FLUXC_Z     = read_check(N,h5_data,h5_file, "flux_c_z")[ind]
    FLUXC_TOT   = read_check(N,h5_data,h5_file, "flux_c_t")[ind]

    FLUXD_X     = read_check(N,h5_data,h5_file, "flux_d_x")[ind]
    FLUXD_Y     = read_check(N,h5_data,h5_file, "flux_d_y")[ind]
    FLUXD_Z     = read_check(N,h5_data,h5_file, "flux_d_z")[ind]
    FLUXD_TOT   = read_check(N,h5_data,h5_file, "flux_d_t")[ind]

    OGS_MAG     = read_check(N,h5_data,h5_file, "ogs_mag")[ind]
    OGS_HGT     = read_check(N,h5_data,h5_file, "ogs_alt")[ind]

    INS_ACC_X   = read_check(N,h5_data,h5_file, "ins_acc_x")[ind]
    INS_ACC_Y   = read_check(N,h5_data,h5_file, "ins_acc_y")[ind]
    INS_ACC_Z   = read_check(N,h5_data,h5_file, "ins_acc_z")[ind]
    INS_WANDER  = read_check(N,h5_data,h5_file, "ins_wander")[ind]
    INS_LAT     = read_check(N,h5_data,h5_file, "ins_lat")[ind]
    INS_LON     = read_check(N,h5_data,h5_file, "ins_lon")[ind]
    INS_HGT     = read_check(N,h5_data,h5_file, "ins_alt")[ind]
    INS_VEL_N   = read_check(N,h5_data,h5_file, "ins_vn")[ind]
    INS_VEL_W   = read_check(N,h5_data,h5_file, "ins_vw")[ind]
    INS_VEL_V   = read_check(N,h5_data,h5_file, "ins_vu")[ind]

    PITCHRT     = read_check(N,h5_data,h5_file, "pitch_rate")[ind]
    ROLLRT      = read_check(N,h5_data,h5_file, "roll_rate")[ind]
    YAWRT       = read_check(N,h5_data,h5_file, "yaw_rate")[ind]
    LONG_ACC    = read_check(N,h5_data,h5_file, "lgtl_acc")[ind]
    LAT_ACC     = read_check(N,h5_data,h5_file, "ltrl_acc")[ind]
    NORM_ACC    = read_check(N,h5_data,h5_file, "nrml_acc")[ind]
    TRUE_AS     = read_check(N,h5_data,h5_file, "tas")[ind]
    PITOT_P     = read_check(N,h5_data,h5_file, "pitot_p")[ind]
    STATIC_P    = read_check(N,h5_data,h5_file, "static_p")[ind]
    TOT_P       = read_check(N,h5_data,h5_file, "total_p")[ind]

    CUR_COM1    = read_check(N,h5_data,h5_file, "cur_com_1")[ind]
    CUR_ACHi    = read_check(N,h5_data,h5_file, "cur_ac_hi")[ind]
    CUR_ACLo    = read_check(N,h5_data,h5_file, "cur_ac_lo")[ind]
    CUR_TANK    = read_check(N,h5_data,h5_file, "cur_tank")[ind]
    CUR_FLAP    = read_check(N,h5_data,h5_file, "cur_flap")[ind]
    CUR_STRB    = read_check(N,h5_data,h5_file, "cur_strb")[ind]
    CUR_SRVO_O  = read_check(N,h5_data,h5_file, "cur_srvo_o")[ind]
    CUR_SRVO_M  = read_check(N,h5_data,h5_file, "cur_srvo_m")[ind]
    CUR_SRVO_I  = read_check(N,h5_data,h5_file, "cur_srvo_i")[ind]
    CUR_IHTR    = read_check(N,h5_data,h5_file, "cur_heat")[ind]
    CUR_ACPWR   = read_check(N,h5_data,h5_file, "cur_acpwr")[ind]
    CUR_OUTPWR  = read_check(N,h5_data,h5_file, "cur_outpwr")[ind]
    CUR_BAT1    = read_check(N,h5_data,h5_file, "cur_bat_1")[ind]
    CUR_BAT2    = read_check(N,h5_data,h5_file, "cur_bat_2")[ind]

    V_ACPWR     = read_check(N,h5_data,h5_file, "vol_acpwr")[ind]
    V_OUTPWR    = read_check(N,h5_data,h5_file, "vol_outpwr")[ind]
    V_BAT1      = read_check(N,h5_data,h5_file, "vol_bat_1")[ind]
    V_BAT2      = read_check(N,h5_data,h5_file, "vol_bat_2")[ind]
    V_RESp      = read_check(N,h5_data,h5_file, "vol_res_p")[ind]
    V_RESn      = read_check(N,h5_data,h5_file, "vol_res_n")[ind]
    V_BACKp     = read_check(N,h5_data,h5_file, "vol_back_p")[ind]
    V_BACKn     = read_check(N,h5_data,h5_file, "vol_back_n")[ind]
    V_GYRO1     = read_check(N,h5_data,h5_file, "vol_gyro_1")[ind]
    V_GYRO2     = read_check(N,h5_data,h5_file, "vol_gyro_2")[ind]
    V_ACCp      = read_check(N,h5_data,h5_file, "vol_acc_p")[ind]
    V_ACCn      = read_check(N,h5_data,h5_file, "vol_acc_n")[ind]
    V_BLOCK     = read_check(N,h5_data,h5_file, "vol_block")[ind]
    V_BACK      = read_check(N,h5_data,h5_file, "vol_back")[ind]
    V_SERVO     = read_check(N,h5_data,h5_file, "vol_srvo")[ind]
    V_CABT      = read_check(N,h5_data,h5_file, "vol_cabt")[ind]
    V_FAN       = read_check(N,h5_data,h5_file, "vol_fan")[ind]

    return (XYZ(N         , DT        , LINE      , FLT       ,
                TIME      , UTM_X     , UTM_Y     , UTM_Z     ,
                MSL_Z     , LAT       , LONG      , BARO      ,
                RADAR     , TOPO      , DEM       , DRAPE     ,
                PITCH     , ROLL      , AZIMUTH   , DIURNAL   ,
                COMPMAG1  , LAGMAG1   , DCMAG1    , IGRFMAG1  ,
                UNCOMPMAG1, UNCOMPMAG2, UNCOMPMAG3, UNCOMPMAG4,
                UNCOMPMAG5, FLUXA_X   , FLUXA_Y   , FLUXA_Z   ,
                FLUXA_TOT , FLUXB_X   , FLUXB_Y   , FLUXB_Z   ,
                FLUXB_TOT , FLUXC_X   , FLUXC_Y   , FLUXC_Z   ,
                FLUXC_TOT , FLUXD_X   , FLUXD_Y   , FLUXD_Z   ,
                FLUXD_TOT , OGS_MAG   , OGS_HGT   , INS_ACC_X ,
                INS_ACC_Y , INS_ACC_Z , INS_WANDER, INS_LAT   ,
                INS_LON   , INS_HGT   , INS_VEL_N , INS_VEL_W ,
                INS_VEL_V , PITCHRT   , ROLLRT    , YAWRT     ,
                LONG_ACC  , LAT_ACC   , NORM_ACC  , TRUE_AS   ,
                PITOT_P   , STATIC_P  , TOT_P     , CUR_COM1  ,
                CUR_ACHi  , CUR_ACLo  , CUR_TANK  , CUR_FLAP  ,
                CUR_STRB  , CUR_SRVO_O, CUR_SRVO_M, CUR_SRVO_I,
                CUR_IHTR  , CUR_ACPWR , CUR_OUTPWR, CUR_BAT1  ,
                CUR_BAT2  , V_ACPWR   , V_OUTPWR  , V_BAT1    ,
                V_BAT2    , V_RESp    , V_RESn    , V_BACKp   ,
                V_BACKn   , V_GYRO1   , V_GYRO2   , V_ACCp    ,
                V_ACCn    , V_BLOCK   , V_BACK    , V_SERVO   ,
                V_CABT    , V_FAN     )
            )

end # function get_flight_data

"""
    read_check(N, data, file, field)

Check for NaNs or missing data (returned as NaNs) in saved xyz HDF5 data.
Prints out warning for any field that contains NaNs.

**Arguments:**
- `N`: number of samples in `field`
- `data`: opened xyz HDF5 data
- `file`: path/name of saved xyz HDF5 file
- `field`: data field to read

**Returns:**
- `val`: data returned for `field`
"""
function read_check(N, data, file, field)
    if field in keys(data)
        val = h5read(file,field)
        if isnan(sum(val))
            print("")
            println("WARNING: ",field," contains NaNs")
        end
    else
        val = zeros(N)*NaN
        print("")
        println("WARNING: ",field," contains NaNs")
    end
    return (val)
end # function read_check
