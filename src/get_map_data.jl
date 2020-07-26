function get_map_data(h5_file::String)
#   h5_file     location/name of magnetic anomaly map HDF5 file

    map_xx   = h5open(h5_file,"r") do file
        vec(read(file,"xx"))
    end

    map_yy   = h5open(h5_file,"r") do file
        vec(read(file,"yy"))
    end

    map_alt  = h5open(h5_file,"r") do file
        read(file,"alt")
    end

    map_map  = h5open(h5_file,"r") do file
        read(file,"map")
    end

    dn = abs(map_yy[end]-map_yy[1])/(length(map_yy)-1)
    de = abs(map_xx[end]-map_xx[1])/(length(map_xx)-1)

    return MapS(map_map,map_xx,map_yy,map_alt,dn,de)

end # function get_map_data
