function get_flight_data(h5_file::String)
#   h5_file     location/name of flight HDF5 file

    println("")
    println("> Reading in file: ", h5_file)

    h5_data     = h5open(h5_file,"r")

    N           = h5read(h5_file, "N")
    DT          = h5read(h5_file, "dt")

    LINE        = readcheck(N,h5_data,h5_file, "tie_line")
    FLT         = readcheck(N,h5_data,h5_file, "flight")
    TIME        = readcheck(N,h5_data,h5_file, "tt")
    UTM_X       = readcheck(N,h5_data,h5_file, "utmX")
    UTM_Y       = readcheck(N,h5_data,h5_file, "utmY")
    UTM_Z       = readcheck(N,h5_data,h5_file, "utmZ")
    MSL_Z       = readcheck(N,h5_data,h5_file, "alt")
    LAT         = readcheck(N,h5_data,h5_file, "lat")
    LONG        = readcheck(N,h5_data,h5_file, "lon")

    BARO        = readcheck(N,h5_data,h5_file, "baro")
    RADAR       = readcheck(N,h5_data,h5_file, "radar")
    TOPO        = readcheck(N,h5_data,h5_file, "topo")
    DEM         = readcheck(N,h5_data,h5_file, "dem")
    DRAPE       = readcheck(N,h5_data,h5_file, "drape")

    PITCH       = readcheck(N,h5_data,h5_file, "ins_pitch")
    ROLL        = readcheck(N,h5_data,h5_file, "ins_roll")
    AZIMUTH     = readcheck(N,h5_data,h5_file, "ins_azim")
    DIURNAL     = readcheck(N,h5_data,h5_file, "diurnal")

    COMPMAG1    = readcheck(N,h5_data,h5_file, "mag_1_c")
    LAGMAG1     = readcheck(N,h5_data,h5_file, "mag_1_lag")
    DCMAG1      = readcheck(N,h5_data,h5_file, "mag_1_dc")
    IGRFMAG1    = readcheck(N,h5_data,h5_file, "mag_1_igrf")
    UNCOMPMAG1  = readcheck(N,h5_data,h5_file, "mag_1_uc")

    UNCOMPMAG2  = readcheck(N,h5_data,h5_file, "mag_2_uc")
    UNCOMPMAG3  = readcheck(N,h5_data,h5_file, "mag_3_uc")
    UNCOMPMAG4  = readcheck(N,h5_data,h5_file, "mag_4_uc")
    UNCOMPMAG5  = readcheck(N,h5_data,h5_file, "mag_5_uc")

    FLUXB_X     = readcheck(N,h5_data,h5_file, "flux_b_x")
    FLUXB_Y     = readcheck(N,h5_data,h5_file, "flux_b_y")
    FLUXB_Z     = readcheck(N,h5_data,h5_file, "flux_b_z")
    FLUXB_TOT   = readcheck(N,h5_data,h5_file, "flux_b_t")

    FLUXC_X     = readcheck(N,h5_data,h5_file, "flux_c_x")
    FLUXC_Y     = readcheck(N,h5_data,h5_file, "flux_c_y")
    FLUXC_Z     = readcheck(N,h5_data,h5_file, "flux_c_z")
    FLUXC_TOT   = readcheck(N,h5_data,h5_file, "flux_c_t")

    FLUXD_X     = readcheck(N,h5_data,h5_file, "flux_d_x")
    FLUXD_Y     = readcheck(N,h5_data,h5_file, "flux_d_y")
    FLUXD_Z     = readcheck(N,h5_data,h5_file, "flux_d_z")
    FLUXD_TOT   = readcheck(N,h5_data,h5_file, "flux_d_t")

    OGS_MAG     = readcheck(N,h5_data,h5_file, "ogs_mag")
    OGS_HGT     = readcheck(N,h5_data,h5_file, "ogs_alt")

    INS_ACC_X   = readcheck(N,h5_data,h5_file, "ins_acc_x")
    INS_ACC_Y   = readcheck(N,h5_data,h5_file, "ins_acc_y")
    INS_ACC_Z   = readcheck(N,h5_data,h5_file, "ins_acc_z")
    INS_WANDER  = readcheck(N,h5_data,h5_file, "ins_wander")
    INS_LAT     = readcheck(N,h5_data,h5_file, "ins_lat")
    INS_LON     = readcheck(N,h5_data,h5_file, "ins_lon")
    INS_HGT     = readcheck(N,h5_data,h5_file, "ins_alt")
    INS_VEL_N   = readcheck(N,h5_data,h5_file, "ins_vn")
    INS_VEL_W   = readcheck(N,h5_data,h5_file, "ins_vw")
    INS_VEL_V   = readcheck(N,h5_data,h5_file, "ins_vu")

    PITCHRT     = readcheck(N,h5_data,h5_file, "pitch_rt")
    ROLLRT      = readcheck(N,h5_data,h5_file, "roll_rt")
    YAWRT       = readcheck(N,h5_data,h5_file, "yaw_rt")
    LONG_ACC    = readcheck(N,h5_data,h5_file, "lon_acc")
    LAT_ACC     = readcheck(N,h5_data,h5_file, "lat_acc")
    NORM_ACC    = readcheck(N,h5_data,h5_file, "alt_acc")
    TRUE_AS     = readcheck(N,h5_data,h5_file, "true_as")
    PITOT_P     = readcheck(N,h5_data,h5_file, "pitot_p")
    STATIC_P    = readcheck(N,h5_data,h5_file, "static_p")
    TOT_P       = readcheck(N,h5_data,h5_file, "total_p")

    CUR_COM1    = readcheck(N,h5_data,h5_file, "cur_com_1")
    CUR_ACHi    = readcheck(N,h5_data,h5_file, "cur_ac_hi")
    CUR_ACLo    = readcheck(N,h5_data,h5_file, "cur_ac_lo")
    CUR_TANK    = readcheck(N,h5_data,h5_file, "cur_tank")
    CUR_FLAP    = readcheck(N,h5_data,h5_file, "cur_flap")
    CUR_STRB    = readcheck(N,h5_data,h5_file, "cur_strb")
    CUR_SRVO_O  = readcheck(N,h5_data,h5_file, "cur_srvo_o")
    CUR_SRVO_M  = readcheck(N,h5_data,h5_file, "cur_srvo_m")
    CUR_SRVO_I  = readcheck(N,h5_data,h5_file, "cur_srvo_i")
    CUR_IHTR    = readcheck(N,h5_data,h5_file, "cur_heat")
    CUR_ACPWR   = readcheck(N,h5_data,h5_file, "cur_acpwr")
    CUR_OUTPWR  = readcheck(N,h5_data,h5_file, "cur_outpwr")
    CUR_BAT1    = readcheck(N,h5_data,h5_file, "cur_bat_1")
    CUR_BAT2    = readcheck(N,h5_data,h5_file, "cur_bat_2")

    V_ACPWR     = readcheck(N,h5_data,h5_file, "vol_acpwr")
    V_OUTPWR    = readcheck(N,h5_data,h5_file, "vol_outpwr")
    V_BAT1      = readcheck(N,h5_data,h5_file, "vol_bat_1")
    V_BAT2      = readcheck(N,h5_data,h5_file, "vol_bat_2")
    V_RESp      = readcheck(N,h5_data,h5_file, "vol_res_p")
    V_RESn      = readcheck(N,h5_data,h5_file, "vol_res_n")
    V_BACKp     = readcheck(N,h5_data,h5_file, "vol_back_p")
    V_BACKn     = readcheck(N,h5_data,h5_file, "vol_back_n")
    V_GYRO1     = readcheck(N,h5_data,h5_file, "vol_gyro_1")
    V_GYRO2     = readcheck(N,h5_data,h5_file, "vol_gyro_2")
    V_ACCp      = readcheck(N,h5_data,h5_file, "vol_acc_p")
    V_ACCn      = readcheck(N,h5_data,h5_file, "vol_acc_n")
    V_BLOCK     = readcheck(N,h5_data,h5_file, "vol_block")
    V_BACK      = readcheck(N,h5_data,h5_file, "vol_back")
    V_SERVO     = readcheck(N,h5_data,h5_file, "vol_servo")
    V_CABT      = readcheck(N,h5_data,h5_file, "vol_cabt")
    V_FAN       = readcheck(N,h5_data,h5_file, "vol_fan")

    return (XYZ(N         , DT        , LINE      , FLT       ,
                TIME      , UTM_X     , UTM_Y     , UTM_Z     ,
                MSL_Z     , LAT       , LONG      , BARO      ,
                RADAR     , TOPO      , DEM       , DRAPE     ,
                PITCH     , ROLL      , AZIMUTH   , DIURNAL   ,
                COMPMAG1  , LAGMAG1   , DCMAG1    , IGRFMAG1  ,
                UNCOMPMAG1, UNCOMPMAG2, UNCOMPMAG3, UNCOMPMAG4,
                UNCOMPMAG5, FLUXB_X   , FLUXB_Y   , FLUXB_Z   ,
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

function readcheck(N,data,file,field)
    if field in names(data)
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
end # function readcheck
