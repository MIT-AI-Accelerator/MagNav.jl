function get_flight_data(h5_file::String)
#   h5_file    location/name of saved HDF5 flight data file

    println("")
    println("> Reading in file: ", h5_file)

    h5_data   = h5open(h5_file,"r")

    N          = h5read(h5_file, "N")
    dt         = h5read(h5_file, "dt")

    tie_line   = readcheck(N,h5_data,h5_file, "tie_line")
    flight     = readcheck(N,h5_data,h5_file, "flight")
    tt         = readcheck(N,h5_data,h5_file, "tt")
    utmX       = readcheck(N,h5_data,h5_file, "utmX")
    utmY       = readcheck(N,h5_data,h5_file, "utmY")
    utmZ       = readcheck(N,h5_data,h5_file, "utmZ")
    alt        = readcheck(N,h5_data,h5_file, "alt")
    lat        = readcheck(N,h5_data,h5_file, "lat")
    lon        = readcheck(N,h5_data,h5_file, "lon")

    # Added for June/July 2020 flights
    baro       = readcheck(N,h5_data,h5_file, "baro")
    radar      = readcheck(N,h5_data,h5_file, "radar")
    topo       = readcheck(N,h5_data,h5_file, "topo")
    dem        = readcheck(N,h5_data,h5_file, "dem")
    drape      = readcheck(N,h5_data,h5_file, "drape")

    ins_pitch  = readcheck(N,h5_data,h5_file, "ins_pitch")
    ins_roll   = readcheck(N,h5_data,h5_file, "ins_roll")
    ins_azim   = readcheck(N,h5_data,h5_file, "ins_azim")
    diurnal    = readcheck(N,h5_data,h5_file, "diurnal")

    mag_1_c    = readcheck(N,h5_data,h5_file, "mag_1_c")
    mag_1_lag  = readcheck(N,h5_data,h5_file, "mag_1_lag")
    mag_1_dc   = readcheck(N,h5_data,h5_file, "mag_1_dc")
    mag_1_igrf = readcheck(N,h5_data,h5_file, "mag_1_igrf")
    mag_1_uc   = readcheck(N,h5_data,h5_file, "mag_1_uc")

    mag_2_uc   = readcheck(N,h5_data,h5_file, "mag_2_uc")
    mag_3_uc   = readcheck(N,h5_data,h5_file, "mag_3_uc")
    mag_4_uc   = readcheck(N,h5_data,h5_file, "mag_4_uc")
    mag_5_uc   = readcheck(N,h5_data,h5_file, "mag_5_uc")

    flux_b_x   = readcheck(N,h5_data,h5_file, "flux_b_x")
    flux_b_y   = readcheck(N,h5_data,h5_file, "flux_b_y")
    flux_b_z   = readcheck(N,h5_data,h5_file, "flux_b_z")
    flux_b_t   = readcheck(N,h5_data,h5_file, "flux_b_t")

    flux_c_x   = readcheck(N,h5_data,h5_file, "flux_c_x")
    flux_c_y   = readcheck(N,h5_data,h5_file, "flux_c_y")
    flux_c_z   = readcheck(N,h5_data,h5_file, "flux_c_z")
    flux_c_t   = readcheck(N,h5_data,h5_file, "flux_c_t")

    flux_d_x   = readcheck(N,h5_data,h5_file, "flux_d_x")
    flux_d_y   = readcheck(N,h5_data,h5_file, "flux_d_y")
    flux_d_z   = readcheck(N,h5_data,h5_file, "flux_d_z")
    flux_d_t   = readcheck(N,h5_data,h5_file, "flux_d_t")

    ogs_mag    = readcheck(N,h5_data,h5_file, "ogs_mag")
    ogs_alt    = readcheck(N,h5_data,h5_file, "ogs_alt")

    ins_acc_x  = readcheck(N,h5_data,h5_file, "ins_acc_x")
    ins_acc_y  = readcheck(N,h5_data,h5_file, "ins_acc_y")
    ins_acc_z  = readcheck(N,h5_data,h5_file, "ins_acc_z")
    ins_wander = readcheck(N,h5_data,h5_file, "ins_wander")
    ins_lat    = readcheck(N,h5_data,h5_file, "ins_lat")
    ins_lon    = readcheck(N,h5_data,h5_file, "ins_lon")
    ins_alt    = readcheck(N,h5_data,h5_file, "ins_alt")
    ins_vn     = readcheck(N,h5_data,h5_file, "ins_vn")
    ins_vw     = readcheck(N,h5_data,h5_file, "ins_vw")
    ins_vu     = readcheck(N,h5_data,h5_file, "ins_vu")

    pitch_rt   = readcheck(N,h5_data,h5_file, "pitch_rt")
    roll_rt    = readcheck(N,h5_data,h5_file, "roll_rt")
    yaw_rt     = readcheck(N,h5_data,h5_file, "yaw_rt")
    lon_acc    = readcheck(N,h5_data,h5_file, "lon_acc")
    lat_acc    = readcheck(N,h5_data,h5_file, "lat_acc")
    alt_acc    = readcheck(N,h5_data,h5_file, "alt_acc")
    true_as    = readcheck(N,h5_data,h5_file, "true_as")
    pitot_p    = readcheck(N,h5_data,h5_file, "pitot_p")
    static_p   = readcheck(N,h5_data,h5_file, "static_p")
    total_p    = readcheck(N,h5_data,h5_file, "total_p")

    cur_com_1  = readcheck(N,h5_data,h5_file, "cur_com_1")
    cur_ac_hi  = readcheck(N,h5_data,h5_file, "cur_ac_hi")
    cur_ac_lo  = readcheck(N,h5_data,h5_file, "cur_ac_lo")
    cur_tank   = readcheck(N,h5_data,h5_file, "cur_tank")
    cur_flap   = readcheck(N,h5_data,h5_file, "cur_flap")
    cur_strb   = readcheck(N,h5_data,h5_file, "cur_strb")
    cur_srvo_o = readcheck(N,h5_data,h5_file, "cur_srvo_o")
    cur_srvo_m = readcheck(N,h5_data,h5_file, "cur_srvo_m")
    cur_srvo_i = readcheck(N,h5_data,h5_file, "cur_srvo_i")
    cur_heat   = readcheck(N,h5_data,h5_file, "cur_heat")
    cur_acpwr  = readcheck(N,h5_data,h5_file, "cur_acpwr")
    cur_outpwr = readcheck(N,h5_data,h5_file, "cur_outpwr")
    cur_bat_1  = readcheck(N,h5_data,h5_file, "cur_bat_1")
    cur_bat_2  = readcheck(N,h5_data,h5_file, "cur_bat_2")

    vol_acpwr  = readcheck(N,h5_data,h5_file, "vol_acpwr")
    vol_outpwr = readcheck(N,h5_data,h5_file, "vol_outpwr")
    vol_bat_1  = readcheck(N,h5_data,h5_file, "vol_bat_1")
    vol_bat_2  = readcheck(N,h5_data,h5_file, "vol_bat_2")
    vol_res_p  = readcheck(N,h5_data,h5_file, "vol_res_p")
    vol_res_n  = readcheck(N,h5_data,h5_file, "vol_res_n")
    vol_back_p = readcheck(N,h5_data,h5_file, "vol_back_p")
    vol_back_n = readcheck(N,h5_data,h5_file, "vol_back_n")
    vol_gyro_1 = readcheck(N,h5_data,h5_file, "vol_gyro_1")
    vol_gyro_2 = readcheck(N,h5_data,h5_file, "vol_gyro_2")
    vol_acc_p  = readcheck(N,h5_data,h5_file, "vol_acc_p")
    vol_acc_n  = readcheck(N,h5_data,h5_file, "vol_acc_n")
    vol_block  = readcheck(N,h5_data,h5_file, "vol_block")
    vol_back   = readcheck(N,h5_data,h5_file, "vol_back")
    vol_servo  = readcheck(N,h5_data,h5_file, "vol_servo")
    vol_cabt   = readcheck(N,h5_data,h5_file, "vol_cabt")
    vol_fan    = readcheck(N,h5_data,h5_file, "vol_fan")

    return (xyz(N         , dt        , tie_line  , flight    ,
                tt        , utmX      , utmY      , utmZ      ,
                alt       , lat       , lon       , baro      ,
                radar     , topo      , dem       , drape     ,
                ins_pitch , ins_roll  , ins_azim  , diurnal   ,
                mag_1_c   , mag_1_lag , mag_1_dc  , mag_1_igrf,
                mag_1_uc  , mag_2_uc  , mag_3_uc  , mag_4_uc  ,
                mag_5_uc  , flux_b_x  , flux_b_y  , flux_b_z  ,
                flux_b_t  , flux_c_x  , flux_c_y  , flux_c_z  ,
                flux_c_t  , flux_d_x  , flux_d_y  , flux_d_z  ,
                flux_d_t  , ogs_mag   , ogs_alt   , ins_acc_x ,
                ins_acc_y , ins_acc_z , ins_wander, ins_lat   ,
                ins_lon   , ins_alt   , ins_vn    , ins_vw    ,
                ins_vu    , pitch_rt  , roll_rt   , yaw_rt    ,
                lon_acc   , lat_acc   , alt_acc   , true_as   ,
                pitot_p   , static_p  , total_p   , cur_com_1 ,
                cur_ac_hi , cur_ac_lo , cur_tank  , cur_flap  ,
                cur_strb  , cur_srvo_o, cur_srvo_m, cur_srvo_i,
                cur_heat  , cur_acpwr , cur_outpwr, cur_bat_1 ,
                cur_bat_2 , vol_acpwr , vol_outpwr, vol_bat_1 ,
                vol_bat_2 , vol_res_p , vol_res_n , vol_back_p,
                vol_back_n, vol_gyro_1, vol_gyro_2, vol_acc_p ,
                vol_acc_n , vol_block , vol_back  , vol_servo ,
                vol_cabt  , vol_fan   )
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
