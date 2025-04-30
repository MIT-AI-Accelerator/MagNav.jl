"""
    get_map(map_file::String   = namad,
            map_field::Symbol  = :map_data;
            map_info::String   = splitpath(map_file)[end],
            map_units::Symbol  = :rad,
            file_units::Symbol = :deg,
            flip_map::Bool     = false)

Get map data from saved HDF5 or MAT file or folder containing CSV files.
Maps are typically saved in `:deg` units, while `:rad` is used internally.

**Arguments:**
- `map_file`:   path/name of map data HDF5 or MAT file (`.h5` or `.mat` extension required) or folder containing CSV files
- `map_field`:  (optional) struct field within MAT file to use, not relevant for CSV or HDF5 file
- `map_info`:   (optional) map information, only used if not in `map_file`
- `map_units`:  (optional) map xx/yy units to use in `map_map` {`:rad`,`:deg`}
- `file_units`: (optional) map xx/yy units used in `map_file` {`:rad`,`:deg`}
- `flip_map`:   (optional) if true, vertically flip data from map file (possibly useful for CSV map)

**Returns:**
- `map_map`: `Map` magnetic anomaly map struct
"""
function get_map(map_file::String   = namad,
                 map_field::Symbol  = :map_data;
                 map_info::String   = splitpath(map_file)[end],
                 map_units::Symbol  = :rad,
                 file_units::Symbol = :deg,
                 flip_map::Bool     = false)

    @assert any(occursin.([".h5",".mat"],map_file)) | isdir(map_file) "$map_file map data file must have .h5 or .mat extension or be a folder containing .csv files"

    map_vec = false

    if occursin(".h5",map_file) # get data from HDF5 file

            map_data = h5open(map_file,"r") # read-only

            if haskey(map_data,"mapX") # vector map
                map_vec  = true
                map_mapX = read_check(map_data,:mapX)
                map_mapY = read_check(map_data,:mapY)
                map_mapZ = read_check(map_data,:mapZ)
                map_mask = map_params(map_mapX)[2]
            elseif haskey(map_data,"map") # scalar map
                map_map  = read_check(map_data,:map)
                map_mask = map_params(map_map)[2]
            end

            map_xx  = read_check(map_data,:xx)
            map_yy  = read_check(map_data,:yy)
            map_alt = read_check(map_data,:alt)

            # these fields might not be included
            map_info  = read_check(map_data,:info,map_info)
            map_mask_ = read_check(map_data,:mask,1,true)
            map_mask  = any(isnan.(map_mask_)) ? map_mask : convert.(Bool,map_mask_)

            close(map_data)

    elseif occursin(".mat",map_file) # get data from MAT file

        map_data = matopen(map_file,"r") do file
            read(file,"$map_field")
        end

        if haskey(map_data,"mapX") # vector map
            map_vec  = true
            map_mapX = map_data["mapX"]
            map_mapY = map_data["mapY"]
            map_mapZ = map_data["mapZ"]
            map_mask = map_params(map_mapX)[2]
        elseif haskey(map_data,"map") # scalar map
            map_map  = map_data["map"]
            map_mask = map_params(map_map)[2]
        end

        map_xx  = map_data["xx"]
        map_yy  = map_data["yy"]
        map_alt = map_data["alt"]

        # these fields might not be included
        map_info = haskey(map_data,"info") ? d["info"] : map_info
        map_mask = haskey(map_data,"mask") ? d["mask"] : map_mask

    else # get data from CSV files

        mapX_csv = joinpath(map_file,"mapX.csv")
        mapY_csv = joinpath(map_file,"mapY.csv")
        mapZ_csv = joinpath(map_file,"mapZ.csv")
        map_csv  = joinpath(map_file,"map.csv")
        xx_csv   = joinpath(map_file,"xx.csv")
        yy_csv   = joinpath(map_file,"yy.csv")
        alt_csv  = joinpath(map_file,"alt.csv")
        info_csv = joinpath(map_file,"info.csv")
        mask_csv = joinpath(map_file,"mask.csv")

        if isfile(mapX_csv) # vector map
            map_vec  = true
            map_mapX = readdlm(mapX_csv,',')
            map_mapY = readdlm(mapY_csv,',')
            map_mapZ = readdlm(mapZ_csv,',')
            map_mask = map_params(map_mapX)[2]
        elseif isfile(map_csv) # scalar map
            map_map  = readdlm(map_csv,',')
            map_mask = map_params(map_map)[2]
        end

        map_xx  = readdlm(xx_csv ,',')
        map_yy  = readdlm(yy_csv ,',')
        map_alt = readdlm(alt_csv,',')

        # these fields might not be included
        map_info = isfile(info_csv) ? readdlm(info_csv,',') : map_info
        map_mask = isfile(mask_csv) ? readdlm(mask_csv,',') : map_mask

    end

    map_xx = vec(map_xx)
    map_yy = vec(map_yy)

    if (file_units == :deg) & (map_units == :rad)
        map_xx .= deg2rad.(map_xx)
        map_yy .= deg2rad.(map_yy)
    elseif (file_units == :rad) & (map_units == :deg)
        map_xx .= rad2deg.(map_xx)
        map_yy .= rad2deg.(map_yy)
    elseif file_units != map_units
        error("[$file_units] map file xx/yy units ≠ [$map_units] map xx/yy units")
    elseif map_units ∉ [:rad,:deg]
        @info("[$map_units] map xx/yy units not defined")
    end

    map_alt = convert.(eltype(map_xx),map_alt)
    (ny,nx) = length.((map_yy,map_xx))

    map_flip(map_map, flip_map::Bool) = flip_map ? map_map[end:-1:1,:] : map_map

    if map_vec
        if (ny,nx) == size(map_mapX) == size(map_mapY) == size(map_mapZ) == size(map_mask)
            map_alt   = map_alt[1]
            map_mapX .= map_flip(map_mapX,flip_map)
            map_mapY .= map_flip(map_mapY,flip_map)
            map_mapZ .= map_flip(map_mapZ,flip_map)
            map_mask .= map_flip(map_mask,flip_map)
            return MapV(map_info,map_mapX,map_mapY,map_mapZ,
                        map_xx,map_yy,map_alt,map_mask)
        else
            error("map dimensions are inconsistent")
        end
    else
        if (ny,nx) == size(map_map[:,:,1]) == size(map_mask[:,:,1])
            if size(map_alt) == size(map_map) # drape map
                map_alt  .= map_flip(map_alt ,flip_map)
                map_map  .= map_flip(map_map ,flip_map)
                map_mask .= map_flip(map_mask,flip_map)
                return MapSd( map_info,map_map,map_xx,map_yy,map_alt,map_mask)
            elseif length(map_alt) > 1 # 3D map
                @assert length(map_alt) == size(map_map,3) "number of map altitude levels is inconsistent"
                map_map  .= map_flip(map_map ,flip_map)
                map_mask .= map_flip(map_mask,flip_map)
                return MapS3D(map_info,map_map,map_xx,map_yy,map_alt,map_mask)
            else
                map_alt   = map_alt[1]
                map_map  .= map_flip(map_map ,flip_map)
                map_mask .= map_flip(map_mask,flip_map)
                return MapS(  map_info,map_map,map_xx,map_yy,map_alt,map_mask)
            end
        else
            error("map dimensions are inconsistent")
        end
    end

end # function get_map

"""
    get_map(map_name::Symbol, df_map::DataFrame,
            map_field::Symbol  = :map_data;
            map_info::String   = "\$map_name",
            map_units::Symbol  = :rad,
            file_units::Symbol = :deg,
            flip_map::Bool     = false)

Get map data from saved HDF5 or MAT file or folder containing CSV files via
DataFrame lookup. Maps are typically saved in `:deg` units, while `:rad` is
used internally.

**Arguments:**
- `map_name`: name of magnetic anomaly map
- `df_map`:   lookup table (DataFrame) of map data HDF5 and/or MAT files and/or folder containing CSV files
|**Field**|**Type**|**Description**
|:--|:--|:--
`map_name`|`Symbol`| name of magnetic anomaly map
`map_file`|`String`| path/name of map data HDF5 or MAT file (`.h5` or `.mat` extension required) or folder containing CSV files
- `map_info`:   (optional) map information, only used if not in `map_file`
- `map_field`:  (optional) struct field within MAT file to use, not relevant for CSV or HDF5 file
- `map_units`:  (optional) map xx/yy units to use in `map_map` {`:rad`,`:deg`}
- `file_units`: (optional) map xx/yy units used in files within `df_map` {`:rad`,`:deg`}
- `flip_map`:   (optional) if true, vertically flip data from map file (possibly useful for CSV map)

**Returns:**
- `map_map`: `Map` magnetic anomaly map struct
"""
function get_map(map_name::Symbol, df_map::DataFrame,
                 map_field::Symbol  = :map_data;
                 map_info::String   = "$map_name",
                 map_units::Symbol  = :rad,
                 file_units::Symbol = :deg,
                 flip_map::Bool     = false)
    ind = findfirst(Symbol.(df_map.map_name) .== map_name)
    get_map(String(df_map.map_file[ind]),map_field;
            map_info   = map_info,
            map_units  = map_units,
            file_units = file_units,
            flip_map   = flip_map)
end # function get_map

"""
    save_map(map_map, map_xx, map_yy, map_alt, map_h5::String = "map_data.h5";
             map_info::String   = "Map",
             map_mask::BitArray = falses(1,1),
             map_border::Matrix = zeros(eltype(map_alt),1,1),
             map_units::Symbol  = :rad,
             file_units::Symbol = :deg)

Save map data to HDF5 file. Maps are typically saved in `:deg` units, while
`:rad` is used internally.

**Arguments:**
- `map_map`:   `ny` x `nx` (x `nz`) 2D or 3D gridded map data
- `map_xx`:    `nx` map x-direction (longitude) coordinates [rad] or [deg]
- `map_yy`:    `ny` map y-direction (latitude)  coordinates [rad] or [deg]
- `map_alt`:    map altitude(s) or `ny` x `nx` 2D gridded altitude map data [m]
- `map_h5`:     (optional) path/name of map data HDF5 file to save (`.h5` extension optional)
- `map_info`:   (optional) map information
- `map_mask`:   (optional) `ny` x `nx` (x `nz`) mask for valid (not filled-in) map data
- `map_border`: (optional) [xx yy] border for valid (not filled-in) map data [rad] or [deg]
- `map_units`:  (optional) map xx/yy units used in `map_xx` & `map_yy` {`:rad`,`:deg`}
- `file_units`: (optional) map xx/yy units to use in `map_h5` {`:rad`,`:deg`}

**Returns:**
- `nothing`: `map_h5` is created
"""
function save_map(map_map, map_xx, map_yy, map_alt, map_h5::String = "map_data.h5";
                  map_info::String   = "Map",
                  map_mask::BitArray = falses(1,1),
                  map_border::Matrix = zeros(eltype(map_alt),1,1),
                  map_units::Symbol  = :rad,
                  file_units::Symbol = :deg)

    map_h5 = add_extension(map_h5,".h5")

    map_xx = vec(map_xx)
    map_yy = vec(map_yy)

    if (map_units == :rad) & (file_units == :deg)
        map_xx     = rad2deg.(map_xx)
        map_yy     = rad2deg.(map_yy)
        map_border = rad2deg.(map_border)
    elseif (map_units == :deg) & (file_units == :rad)
        map_xx     = deg2rad.(map_xx)
        map_yy     = deg2rad.(map_yy)
        map_border = deg2rad.(map_border)
    elseif map_units != file_units
        error("[$map_units] map xx/yy units ≠ [$file_units] map file xx/yy units")
    elseif map_units ∉ [:rad,:deg]
        @info("[$map_units] map xx/yy units not defined")
    end

    map_alt  = convert.(eltype(map_xx),map_alt)
    map_mask = Int8.(map_mask)

    @info("saving map with [$file_units] map xx/yy units")

    h5open(map_h5,"w") do file # read-write, destroy existing contents
        write(file,"info",map_info)
        if length(map_map) == 3 # vector map
            write(file,"mapX",map_map[1])
            write(file,"mapY",map_map[2])
            write(file,"mapZ",map_map[3])
        else # scalar map
            write(file,"map",map_map)
        end
        write(file,"xx" ,map_xx)
        write(file,"yy" ,map_yy)
        write(file,"alt",map_alt)
        sum(map_mask  ) != 0 && write(file,"mask"  ,map_mask)
        sum(map_border) != 0 && write(file,"border",map_border)
    end

    return (nothing)
end # function save_map

"""
    save_map(map_map::Map, map_h5::String = "map_data.h5";
             map_border::Matrix = zeros(eltype(map_map.alt),1,1),
             map_units::Symbol  = :rad,
             file_units::Symbol = :deg)

Save map data to HDF5 file. Maps are typically saved in `:deg` units, while
`:rad` is used internally.

**Arguments:**
- `map_map`:    `Map` magnetic anomaly map struct
- `map_h5`:     (optional) path/name of map data HDF5 file to save (`.h5` extension optional)
- `map_border`: (optional) [xx yy] border for valid (not filled-in) map data [rad] or [deg]
- `map_units`:  (optional) map xx/yy units used in `map_map` {`:rad`,`:deg`}
- `file_units`: (optional) map xx/yy units to use in `map_h5` {`:rad`,`:deg`}

**Returns:**
- `nothing`: `map_h5` is created
"""
function save_map(map_map::Map, map_h5::String = "map_data.h5";
                  map_border::Matrix = zeros(eltype(map_map.alt),1,1),
                  map_units::Symbol  = :rad,
                  file_units::Symbol = :deg)
    if map_map isa MapV # vector map
        save_map((map_map.mapX,map_map.mapY,map_map.mapZ),
                 map_map.xx,map_map.yy,map_map.alt,map_h5;
                 map_info   = map_map.info,
                 map_mask   = map_map.mask,
                 map_border = map_border,
                 map_units  = map_units,
                 file_units = file_units)
    else # scalar map
        save_map(map_map.map,
                 map_map.xx,map_map.yy,map_map.alt,map_h5;
                 map_info   = map_map.info,
                 map_mask   = map_map.mask,
                 map_border = map_border,
                 map_units  = map_units,
                 file_units = file_units)
    end
    return (nothing)
end # function save_map

"""
    get_comp_params(comp_params_bson::String, silent::Bool = false)

Get aeromagnetic compensation parameters from saved BSON file.

**Arguments:**
- `comp_params_bson`: path/name of aeromagnetic compensation parameters BSON file (`.bson` extension optional)
- `silent`:           (optional) if true, no print outs

**Returns:**
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct, either:
    - `NNCompParams`:  neural network-based aeromagnetic compensation parameters struct
    - `LinCompParams`: linear aeromagnetic compensation parameters struct
"""
function get_comp_params(comp_params_bson::String, silent::Bool = false)

    comp_params_bson = add_extension(comp_params_bson,".bson")

    # load in fields of comp_params
    d = load(comp_params_bson)
    model_type  = :model_type in keys(d) ? d[:model_type] : nothing
    comp_params = nothing

    # get default field values (in case not in saved comp_params)
    if model_type in [:m1,:m2a,:m2b,:m2c,:m2d,:m3tl,:m3s,:m3v,:m3sc,:m3vc,:m3w,:m3tf]
        comp_params_default = NNCompParams()
        silent || @info("loading individual model $model_type NN compensation parameters")
    elseif model_type in [:TL,:mod_TL,:map_TL,:elasticnet,:plsr]
        comp_params_default = LinCompParams()
        silent || @info("loading individual model $model_type linear compensation parameters")
    else
        try
            comp_params = d[:comp_params]
            comp_params_default = nothing
            silent || @info("loading full compensation parameters struct")
        catch _
            error("$comp_params_bson compensation parameters BSON file is invalid")
        end
    end

    # use default value if field is not in saved comp_params
    for field in fieldnames(typeof(comp_params_default))
        if !(field in keys(d))
            @info("using default for $field field")
            d[field] = getfield(comp_params_default,field)
        end
    end

    if comp_params_default isa NNCompParams
        @unpack version, features_setup, features_no_norm, model_type, y_type,
        use_mag, use_vec, data_norms, model, terms, terms_A, sub_diurnal,
        sub_igrf, bpf_mag, reorient_vec, norm_type_A, norm_type_x, norm_type_y,
        TL_coef, η_adam, epoch_adam, epoch_lbfgs, hidden, activation, loss,
        batchsize, frac_train, α_sgl, λ_sgl, k_pca,
        drop_fi, drop_fi_bson, drop_fi_csv, perm_fi, perm_fi_csv = d
        comp_params = NNCompParams(version          = VersionNumber(version),
                                   features_setup   = features_setup,
                                   features_no_norm = features_no_norm,
                                   model_type       = model_type,
                                   y_type           = y_type,
                                   use_mag          = use_mag,
                                   use_vec          = use_vec,
                                   data_norms       = data_norms,
                                   model            = model,
                                   terms            = terms,
                                   terms_A          = terms_A,
                                   sub_diurnal      = sub_diurnal,
                                   sub_igrf         = sub_igrf,
                                   bpf_mag          = bpf_mag,
                                   reorient_vec     = reorient_vec,
                                   norm_type_A      = norm_type_A,
                                   norm_type_x      = norm_type_x,
                                   norm_type_y      = norm_type_y,
                                   TL_coef          = TL_coef,
                                   η_adam           = η_adam,
                                   epoch_adam       = epoch_adam,
                                   epoch_lbfgs      = epoch_lbfgs,
                                   hidden           = hidden,
                                   activation       = activation,
                                   loss             = loss,
                                   batchsize        = batchsize,
                                   frac_train       = frac_train,
                                   α_sgl            = α_sgl,
                                   λ_sgl            = λ_sgl,
                                   k_pca            = k_pca,
                                   drop_fi          = drop_fi,
                                   drop_fi_bson     = drop_fi_bson,
                                   drop_fi_csv      = drop_fi_csv,
                                   perm_fi          = perm_fi,
                                   perm_fi_csv      = perm_fi_csv)
    elseif comp_params_default isa LinCompParams
        @unpack version, features_setup, features_no_norm, model_type, y_type,
        use_mag, use_vec, data_norms, model, terms, terms_A, sub_diurnal,
        sub_igrf, bpf_mag, reorient_vec, norm_type_A, norm_type_x, norm_type_y,
        k_plsr, λ_TL = d
        comp_params = LinCompParams(version          = VersionNumber(version),
                                    features_setup   = features_setup,
                                    features_no_norm = features_no_norm,
                                    model_type       = model_type,
                                    y_type           = y_type,
                                    use_mag          = use_mag,
                                    use_vec          = use_vec,
                                    data_norms       = data_norms,
                                    model            = model,
                                    terms            = terms,
                                    terms_A          = terms_A,
                                    sub_diurnal      = sub_diurnal,
                                    sub_igrf         = sub_igrf,
                                    bpf_mag          = bpf_mag,
                                    reorient_vec     = reorient_vec,
                                    norm_type_A      = norm_type_A,
                                    norm_type_x      = norm_type_x,
                                    norm_type_y      = norm_type_y,
                                    k_plsr           = k_plsr,
                                    λ_TL             = λ_TL)
    end

    return (comp_params)
end # function get_comp_params

"""
    save_comp_params(comp_params::CompParams,
                     comp_params_bson::String = "comp_params.bson")

Save aeromagnetic compensation parameters to BSON file.

**Arguments:**
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct, either:
    - `NNCompParams`:  neural network-based aeromagnetic compensation parameters struct
    - `LinCompParams`: linear aeromagnetic compensation parameters struct
- `comp_params_bson`: (optional) path/name of aeromagnetic compensation parameters BSON file to save (`.bson` extension optional)

**Returns:**
- `nothing`: `comp_params_bson` is created
"""
function save_comp_params(comp_params::CompParams,
                          comp_params_bson::String = "comp_params.bson")
    comp_params_bson = add_extension(comp_params_bson,".bson")
    d = Dict{Symbol,Any}()
    # push!(d,:comp_params => comp_params) # do NOT do this, version issue
    for field in fieldnames(typeof(comp_params))
        push!(d,field => getfield(comp_params,field))
    end
    bson(comp_params_bson,d)
    return (nothing)
end # function save_comp_params
