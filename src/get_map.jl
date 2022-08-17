"""
    get_map(map_file::String; map_units::Symbol=:deg)

Get map data from saved file. Map files are typically saved with `:deg` units.

**Arguments:**
- `map_file`:  path/name of magnetic anomaly map HDF5 or MAT file
- `map_units`: (optional) map xx/yy units used in HDF5 file {`:deg`,`:rad`}

**Returns:**
- `mapS` or `mapV`: `MapS` scalar or `MapV` vector magnetic anomaly map struct
"""
function get_map(map_file::String; map_units::Symbol=:deg)

    map_vec = false

    if occursin(".h5",map_file) # get data from HDF5 file

            map_data = h5open(map_file,"r") # read-only

            map_xx   = read_check(map_data,:xx)
            map_yy   = read_check(map_data,:yy)
            map_alt  = read_check(map_data,:alt)

            if haskey(map_data,"mapX") # vector magnetic anomaly map
                map_vec  = true
                map_mapX = read_check(map_data,:mapX)
                map_mapY = read_check(map_data,:mapY)
                map_mapZ = read_check(map_data,:mapZ)
            elseif haskey(map_data,"map") # scalar magnetic anomaly map
                map_map  = read_check(map_data,:map)
            end

            close(map_data)

    elseif occursin(".mat",map_file) # get data from MAT file

        map_data = matopen(map_file,"r") do file
            read(file,"map_data")
        end

        map_xx  = map_data["xx"]
        map_yy  = map_data["yy"]
        map_alt = map_data["alt"]

        if haskey(map_data,"mapX") # vector magnetic anomaly map
            map_vec  = true
            map_mapX = map_data["mapX"]
            map_mapY = map_data["mapY"]
            map_mapZ = map_data["mapZ"]
        elseif haskey(map_data,"map") # scalar magnetic anomaly map
            map_map  = map_data["map"]
        end

    else
        error("$map_file map file is incorrect or invalid")
    end

    map_xx = vec(map_xx)
    map_yy = vec(map_yy)

    if map_units == :deg
        map_xx = deg2rad.(map_xx)
        map_yy = deg2rad.(map_yy)
    elseif map_units != :rad
        @info("$map_units map xx/yy units not defined")
    end

    if length(map_alt) == 1 # not drape map
        map_d   = false
        map_alt = map_alt[1]
    else # drape map
        map_d   = true
    end

    if map_vec
        if ((length(map_yy),length(map_xx)) == size(map_mapX)) & 
           ((length(map_yy),length(map_xx)) == size(map_mapY)) & 
           ((length(map_yy),length(map_xx)) == size(map_mapZ))
            if map_d
                return MapVd(map_mapX,map_mapY,map_mapZ,map_xx,map_yy,map_alt)
            else
                return MapV( map_mapX,map_mapY,map_mapZ,map_xx,map_yy,map_alt)
            end
        else
            error("map dimensions are inconsistent")
        end
    else
        if (length(map_yy),length(map_xx)) == size(map_map)
            if map_d
                return MapSd(map_map,map_xx,map_yy,map_alt)
            else
                return MapS( map_map,map_xx,map_yy,map_alt)
            end
        else
            error("map dimensions are inconsistent")
        end
    end

end # function get_map

"""
    get_map(map_name::Symbol, df_map::DataFrame; map_units::Symbol=:deg)

Get map data from saved file via DataFrame lookup. 
Map files are typically saved with `:deg` units.

**Arguments:**
- `map_name`:  name of magnetic anomaly map
- `df_map`:    lookup table (DataFrame) of map files
- `map_units`: (optional) map xx/yy units used in HDF5 file {`:deg`,`:rad`}

**Returns:**
- `mapS` or `mapV`: `MapS` scalar or `MapV` vector magnetic anomaly map struct
"""
function get_map(map_name::Symbol, df_map::DataFrame; map_units::Symbol=:deg)
    get_map(df_map.map_h5[df_map.map_name .== map_name][1]; map_units=map_units)
end # function get_map

"""
    save_map(map_map, map_xx, map_yy, map_alt, map_h5::String="map.h5";
             map_units::Symbol=:deg)

Save map data to HDF5 file. Map files are typically saved with `:deg` units.

**Arguments:**
- `map_map`:  `ny` x `nx` 2D gridded map data
- `map_xx`:   `nx` x-direction (longitude) map coordinates [rad] or [m]
- `map_yy`:   `ny` y-direction (latitude)  map coordinates [rad] or [m]
- `map_alt`:   map altitude or `ny` x `nx` 2D gridded altitude map data [m]
- `map_h5`:    map path/name to save
- `map_units`: (optional) map xx/yy units to use in HDF5 file {`:deg`,`:rad`,`:utm`}

**Returns:**
- `nothing`: HDF5 file `map_h5` is created
"""
function save_map(map_map, map_xx, map_yy, map_alt, map_h5::String="map.h5";
                  map_units::Symbol=:deg)

    if map_units == :deg
        map_xx = rad2deg.(map_xx)
        map_yy = rad2deg.(map_yy)
    end

    if map_units in [:deg,:rad,:utm]
        @info("saving map with $map_units map xx/yy units")
    else
        @info("$map_units map xx/yy units not defined")
    end

    h5open(map_h5,"w") do file # read-write, destroy existing contents
        write(file,"xx" ,map_xx)
        write(file,"yy" ,map_yy)
        write(file,"alt",map_alt)
        write(file,"map",map_map)
    end

end # function save_map

"""
    save_map(mapS::Union{MapS,MapSd}, map_h5::String="map.h5";
             map_units::Symbol=:deg)

Save map data to HDF5 file. Map files are typically saved with `:deg` units.

**Arguments:**
- `mapS`:      `MapS` or `MapSd` scalar magnetic anomaly map struct
- `map_h5`:    map path/name to save
- `map_units`: (optional) map xx/yy units to use in HDF5 file {`:deg`,`:rad`,`:utm`}

**Returns:**
- `nothing`: HDF5 file `map_h5` is created
"""
function save_map(mapS::Union{MapS,MapSd}, map_h5::String="map.h5";
                  map_units::Symbol=:deg)
    save_map(mapS.map,mapS.xx,mapS.yy,mapS.alt,map_h5;map_units=map_units)
end # function save_map
