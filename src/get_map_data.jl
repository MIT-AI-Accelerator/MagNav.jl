function get_map_data(h5_file::String)
#   h5_file     location/name of magnetic anomaly map HDF5 file

    map_xx   = h5open(h5_file,"r") do file
        vec(read(file,"xx"))
    end

    map_yy   = h5open(h5_file,"r") do file
        vec(read(file,"yy"))
    end

    map_alt  = h5open(h5_file,"r") do file
        vec(read(file,"alt"))[1]
    end

    dlat = abs(map_yy[end]-map_yy[1])/(length(map_yy)-1)
    dlon = abs(map_xx[end]-map_xx[1])/(length(map_xx)-1)
    dn   = delta_north(deg2rad(dlat),deg2rad(mean(map_yy)))
    de   = delta_east( deg2rad(dlon),deg2rad(mean(map_yy)))

    map_data = h5open(h5_file,"r")

    if exists(map_data,"mapX") # vector magnetometer
        map_mapX = read(map_data,"mapX")
        map_mapY = read(map_data,"mapY")
        map_mapZ = read(map_data,"mapY")
        return mapV(map_mapX,map_mapY,map_mapZ,map_xx,map_yy,map_alt,dn,de)
    elseif exists(map_data,"map" ) # scalar magnetometer
        map_map  = read(map_data,"map" )
        return mapS(map_map,map_xx,map_yy,map_alt,dn,de)
    end

    close(map_data)

end # function get_map_data
