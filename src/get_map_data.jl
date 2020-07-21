function get_map_data(map_file::String)
#   map_file    location/name of magnetic anomaly map HDF5 or MAT file

    if occursin(".h5",map_file) # pull data from HDF5 file

        map_xx   = h5open(map_file,"r") do file
            vec(read(file,"xx"))
        end

        map_yy   = h5open(map_file,"r") do file
            vec(read(file,"yy"))
        end

        map_alt  = h5open(map_file,"r") do file
            vec(read(file,"alt"))[1]
        end

        dlat = abs(map_yy[end]-map_yy[1])/(length(map_yy)-1)
        dlon = abs(map_xx[end]-map_xx[1])/(length(map_xx)-1)
        dn   = delta_north(deg2rad(dlat),deg2rad(mean(map_yy)))
        de   = delta_east( deg2rad(dlon),deg2rad(mean(map_yy)))

        map_data = h5open(map_file,"r")
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

    elseif occursin(".mat",map_file) # pull data from MAT file

        map_data = matopen(map_file,"r") do file
          read(file,"map_data")
        end

        map_xx  = vec(map_data["xx"])
        map_yy  = vec(map_data["yy"])
        map_alt = map_data["alt"]

        dlat = abs(map_yy[end]-map_yy[1])/(length(map_yy)-1)
        dlon = abs(map_xx[end]-map_xx[1])/(length(map_xx)-1)
        dn   = delta_north(deg2rad(dlat),deg2rad(mean(map_yy)))
        de   = delta_east( deg2rad(dlon),deg2rad(mean(map_yy)))

        if haskey(map_data,"mapX") # vector magnetometer
            map_mapX = map_data["mapX"]
            map_mapY = map_data["mapY"]
            map_mapZ = map_data["mapZ"]
            return mapV(map_mapX,map_mapY,map_mapZ,map_xx,map_yy,map_alt,dn,de)
        elseif haskey(map_data,"map") # scalar magnetometer
            map_map  = map_data["map"]
            return mapS(map_map,map_xx,map_yy,map_alt,dn,de)
        end

    end

end # function get_map_data
