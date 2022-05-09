"""
    map_interpolate(map_map, map_xx, map_yy, type::Symbol = :cubic)

Create map grid interpolation, equivalent of griddedInterpolant in MATLAB. 
Uses the Interpolations package rather than Dierckx or GridInterpolations, as 
Interpolations was found to be fastest for MagNav use cases.

**Arguments:**
- `map_map`: `ny` x `nx` 2D gridded map data
- `map_xx`:  `nx` x-direction map coordinates
- `map_yy`:  `ny` y-direction map coordinates
- `type`:    (optional) type of interpolation {:linear,:quad,:cubic}

**Returns:**
- `itp_map`: map grid interpolation
"""
function map_interpolate(map_map, map_xx, map_yy, type::Symbol = :cubic)

    if type == :linear
        spline_type = BSpline(Linear())
    elseif type == :quad
        spline_type = BSpline(Quadratic(Line(OnGrid())))
    elseif type == :cubic
        spline_type = BSpline(Cubic(Line(OnGrid())))
    else
        error("$type interpolation type not defined")
    end

    xx = LinRange(minimum(map_xx),maximum(map_xx),length(map_xx))
    yy = LinRange(minimum(map_yy),maximum(map_yy),length(map_yy))

    return scale(interpolate(map_map',spline_type),xx,yy)
end # function map_interpolate

"""
    map_interpolate(mapS::MapS, type::Symbol = :cubic; vert::Bool = false)

Create map grid interpolation, equivalent of griddedInterpolant in MATLAB.
Optionally return vertical derivative grid interpolation, which is calculated
using finite differences between the map and a slightly upward continued map.

**Arguments:**
- `mapS`: `MapS` scalar magnetic anomaly map struct
- `type`: (optional) type of interpolation {:linear,:quad,:cubic}
- `vert`: (optional) if true, also return vertical derivative grid interpolation

**Returns:**
- `itp_map`: map grid interpolation
"""
function map_interpolate(mapS::MapS, type::Symbol = :cubic; vert::Bool = false)

    if vert
        return (map_interpolate(mapS.map,mapS.xx,mapS.yy,type),
                map_interpolate(upward_fft(mapS,mapS.alt+1).map - mapS.map,
                                           mapS.xx,mapS.yy,type))
    else
        map_interpolate(mapS.map,mapS.xx,mapS.yy,type)
    end
end # function map_interpolate

"""
    map_interpolate(mapV::MapV, dim::Symbol = :X, type::Symbol = :cubic)

Create map grid interpolation, equivalent of griddedInterpolant in MATLAB.

**Arguments:**
- `mapV`: `MapV` vector magnetic anomaly map struct
- `dim`:  map dimension to interpolate {`:X`,`:Y`,`:Z`}
- `type`: (optional) type of interpolation {:linear,:quad,:cubic}

**Returns:**
- `itp_map`: map grid interpolation
"""
function map_interpolate(mapV::MapV, dim::Symbol = :X, type::Symbol = :cubic)

    if dim == :X
        map_interpolate(mapV.mapX,mapV.xx,mapV.yy,type)
    elseif dim == :Y
        map_interpolate(mapV.mapY,mapV.xx,mapV.yy,type)
    elseif dim == :Z
        map_interpolate(mapV.mapZ,mapV.xx,mapV.yy,type)
    else
        error("$dim dim not defined")
    end

end # function map_interpolate

map_itp = map_interpolate

"""
    map_get_gxf(gxf_file::String)

Use ArchGDAL to read in map data from .gxf file.

**Arguments:**
- `gxf_file`: path/name of .gxf file with map data

**Returns:**
- `map_map`: `ny` x `nx` 2D gridded map data
- `map_xx`:  `nx` x-direction map coordinates
- `map_yy`:  `ny` y-direction map coordinates
"""
function map_get_gxf(gxf_file::String)

    # configure dataset to be read as Float64
    ArchGDAL.setconfigoption("GXF_DATATYPE","Float64")

    # read GXF (raster-type) dataset from gxf_file
    ArchGDAL.read(gxf_file) do dataset

        # read map data into array
        # rows reversed to match getgrd2 in MATLAB
        map_map = reverse(ArchGDAL.read(dataset,1)',dims=1)

        # read size of map
        nx = ArchGDAL.width(dataset)
        ny = ArchGDAL.height(dataset)

        # read geometry transformation properties
        gt = ArchGDAL.getgeotransform(dataset)

        # create x and y coordinate arrays
        # map_yy reversed to match getgrd2 in MATLAB
        # both offset by half step size to match getgrd2 in MATLAB
        map_xx = [LinRange(gt[1],gt[1]+gt[2]*(nx-1),nx);] .+ gt[2]/2
        map_yy = [LinRange(gt[4]+gt[6]*(ny-1),gt[4],ny);] .+ gt[6]/2

        # dummy value used where no data exists
        dum = minimum(map_map)
        replace!(map_map, dum=>0)
        replace!(map_map, NaN=>0) # just in case

        # map_map differs from  getgrd2 in MATLAB by ~1e-7
        return (map_map, map_xx, map_yy)

    end

end # function map_get_gxf

"""
    map_params(map_map::Matrix, map_xx::Vector, map_yy::Vector)

Internal helper function to get basic map parameters.

**Arguments:**
- `map_map`: `ny` x `nx` 2D gridded map data
- `map_xx`:  `nx` x-direction map coordinates
- `map_yy`:  `ny` y-direction map coordinates

**Returns:**
- `ind0`: map indices with zeros
- `ind1`: map indices without zeros
- `nx`: x-direction map dimension
- `ny`: y-direction map dimension
"""
function map_params(map_map::Matrix, map_xx::Vector, map_yy::Vector)

    replace!(map_map, NaN=>0) # just in case

    # map indices with (ind0) and without (ind1) zeros
    ind0 = map_map .== 0
    ind1 = map_map .!= 0

    # map size
    (ny,nx) = size(map_map)
    nx == length(map_xx) || error("xx map dimensions are inconsistent")
    ny == length(map_yy) || error("yy map dimensions are inconsistent")

    return (ind0, ind1, nx, ny)
end # function map_params

"""
    map_params(map_map::Map)

Internal helper function to get basic map parameters.

**Arguments:**
- `map_map`: `Map` magnetic anomaly map struct

**Returns:**
- `ind0`: map indices with zeros
- `ind1`: map indices without zeros
- `nx`: x-direction map dimension
- `ny`: y-direction map dimension
"""
function map_params(map_map::Map)
    if typeof(map_map) <: Union{MapS,MapSd} # scalar map
        map_params(deepcopy(map_map.map),map_map.xx,map_map.yy)
    else # vector Map
        map_params(deepcopy(map_map.mapX),map_map.xx,map_map.yy)
    end
end # function map_params

"""
    map_lla_lim(map_xx::Vector, map_yy::Vector, alt,
                xx_1::Int=1 ,xx_nx::Int=length(map_xx),
                yy_1::Int=1, yy_ny::Int=length(map_yy);
                zone_utm::Int  = 18,
                is_north::Bool = true)

Internal helper function to get lat/lon limits at 4 corners of UTMZ map.

**Arguments:**
- `map_xx`:   `nx` x-direction map coordinates [m]
- `map_yy`:   `ny` y-direction map coordinates [m]
- `alt`:      nominal map altitude, median used for 2D altitude map [m]
- `xx_1`:     first index of `map_xx` to consider
- `xx_nx`:    last  index of `map_xx` to consider
- `yy_1`:     first index of `map_yy` to consider
- `yy_ny`:    last  index of `map_yy` to consider
- `zone_utm`: (optional) UTM zone
- `is_north`: (optional) if true, map is in northern hemisphere

**Returns:**
- `lons`: longitudes for each of 4 corners of UTMZ map [deg]
- `lats`: latitudes  for each of 4 corners of UTMZ map [deg]
"""
function map_lla_lim(map_xx::Vector, map_yy::Vector, alt,
                     xx_1, xx_nx, yy_1, yy_ny;
                     zone_utm::Int  = 18,
                     is_north::Bool = true)

    utm2lla = LLAfromUTMZ(WGS84)

    # 4 corners of UTMZ map
    utms = [UTMZ(x,y,alt,zone_utm,is_north) for x in map_xx[[xx_1,xx_nx]]
                                            for y in map_yy[[yy_1,yy_ny]]]

    # sorted longitudes at 4 corners of UTMZ map
    # left/right edges are straight, so only corners needed
    lons = sort([utm2lla(utm).lon for utm in utms])

    # lower/upper parallels of UTMZ map
    utms_1  = UTMZ.(map_xx[xx_1:xx_nx],map_yy[yy_1 ],alt,zone_utm,is_north)
    utms_ny = UTMZ.(map_xx[xx_1:xx_nx],map_yy[yy_ny],alt,zone_utm,is_north)

    # sorted latitude limits for lower/upper parallels of UTMZ map
    lats = sort([extrema([utm2lla(utm).lat for utm in utms_1 ])...,
                 extrema([utm2lla(utm).lat for utm in utms_ny])...])

    return (lons, lats)
end # function map_lla_lim

"""
    map_trim(map_map::Matrix, map_xx::Vector, map_yy::Vector, alt;
             pad::Int          = 0,
             xx_lim::Tuple     = (-Inf,Inf),
             yy_lim::Tuple     = (-Inf,Inf),
             zone_utm::Int     = 18,
             is_north::Bool    = true,
             map_units::Symbol = :utm,
             silent::Bool      = false)

Trim map by removing large areas that are missing map data. Returns 
indices for the original map that will produce the appropriate trimmed map.

**Arguments:**
- `map_map`:   `ny` x `nx` 2D gridded map data
- `map_xx`:    `nx` x-direction map coordinates [m] or [rad] or [deg]
- `map_yy`:    `ny` y-direction map coordinates [m] or [rad] or [deg]
- `alt`:       nominal map altitude, median used for 2D altitude map [m]
- `pad`:       (optional) minimum padding along map edges
- `xx_lim`:    (optional) x-direction map limits (xx_min,xx_max) [m] or [rad] or [deg]
- `yy_lim`:    (optional) y-direction map limits (yy_min,yy_max) [m] or [rad] or [deg]
- `zone_utm`:  (optional) UTM zone
- `is_north`:  (optional) if true, map is in northern hemisphere
- `map_units`: (optional) map xx/yy units {`:utm`,`:m`,`:rad`,`:deg`}
- `silent`:    (optional) if true, no print outs

**Returns:**
- `ind_xx`: `nx` trimmed x-direction map indices
- `ind_yy`: `ny` trimmed y-direction map indices
"""
function map_trim(map_map::Matrix, map_xx::Vector, map_yy::Vector, alt;
                  pad::Int          = 0,
                  xx_lim::Tuple     = (-Inf,Inf),
                  yy_lim::Tuple     = (-Inf,Inf),
                  zone_utm::Int     = 18,
                  is_north::Bool    = true,
                  map_units::Symbol = :utm,
                  silent::Bool      = false)

    (_,ind1,nx,ny) = map_params(map_map,map_xx,map_yy)

    length(alt) > 1 && (alt = median(alt[ind1])) # in case 2D altitude map provided
    alt < 0 && (alt = 300) # in case drape map altitude (-1) provided

    # xx limits of data-containing UTMZ map
    xx_sum  = vec(sum(map_map,dims=1))
    xx_1    = findfirst(xx_sum .!= 0) # 2491,  580
    xx_nx   = findlast(xx_sum  .!= 0) # 8588, 3743

    # yy limits of data-containing UTMZ map
    yy_sum = vec(sum(map_map,dims=2))
    yy_1   = findfirst(yy_sum .!= 0) #  494, 2604
    yy_ny  = findlast(yy_sum  .!= 0) # 5290, 6301

    # (optional) user-specified limits
    xx_lim = extrema(findall((map_xx .> minimum(xx_lim)) .& 
                             (map_xx .< maximum(xx_lim))))
    yy_lim = extrema(findall((map_yy .> minimum(yy_lim)) .& 
                             (map_yy .< maximum(yy_lim))))

    # smallest possible data-containing & user-specified map
    xx_1   = maximum([xx_1 ,xx_lim[1]])
    xx_nx  = minimum([xx_nx,xx_lim[2]])
    yy_1   = maximum([yy_1 ,yy_lim[1]])
    yy_ny  = minimum([yy_ny,yy_lim[2]])

    if map_units in [:utm,:m]

        # get xx/yy limits at 4 corners of data-containing UTMZ map for no data loss
        (lons,lats) = map_lla_lim(map_xx,map_yy,alt,xx_1,xx_nx,yy_1,yy_ny;
                                  zone_utm  = zone_utm,
                                  is_north  = is_north)

        lla2utm = UTMZfromLLA(WGS84)

        # use EXTERIOR 2 lons/lats as xx/yy limits
        llas = [LLA(lat,lon,alt) for lat in lats[[1,end]] for lon in lons[[1,end]]]

        # xx/yy limits at 4 corners of UTMZ map for no data loss
        # due to earth's curvature, xx/yy limits are further out
        xxs  = sort([lla2utm(lla).x for lla in llas])
        yys  = sort([lla2utm(lla).y for lla in llas])

    elseif map_units in [:rad,:deg]

        # directly use data-containing xx/yy limits at 4 corners
        xxs = sort(map_xx[[xx_1,xx_1,xx_nx,xx_nx]])
        yys = sort(map_yy[[yy_1,yy_1,yy_ny,yy_ny]])

    else
        error("$map_units map xx/yy units not defined")

    end

    # minimum padding (per edge) to prevent data loss during utm2lla
    pad_xx_1 = xxs[1] < map_xx[1] ? xx_1-1 : xx_1 - findlast(map_xx .< xxs[1])
    pad_yy_1 = yys[1] < map_yy[1] ? yy_1-1 : yy_1 - findlast(map_yy .< yys[1])

    if xxs[end] > map_xx[end]
        pad_xx_nx = nx - xx_nx
    else
        pad_xx_nx = findfirst(map_xx .> xxs[end]) - xx_nx
    end

    if yys[end] > map_yy[end]
        pad_yy_ny = ny - yy_ny
    else
        pad_yy_ny = findfirst(map_yy .> yys[end]) - yy_ny
    end

    silent || @info("max map padding: $([xx_1-1,nx-xx_nx,yy_1-1,ny-yy_ny])")
    silent || @info("min map padding: $([pad_xx_1,pad_xx_nx,pad_yy_1,pad_yy_ny])")

    # padding (per edge): < available padding & > utm2lla conversion limit
    pad_xx_1  = clamp(pad,pad_xx_1  ,xx_1-1)
    pad_yy_1  = clamp(pad,pad_yy_1  ,yy_1-1)
    pad_xx_nx = clamp(pad,pad_xx_nx,nx-xx_nx)
    pad_yy_ny = clamp(pad,pad_yy_ny,ny-yy_ny)

    silent || @info("set map padding: $([pad_xx_1,pad_xx_nx,pad_yy_1,pad_yy_ny])")

    # indices that will remove zero rows/columns with padding (per edge)
    ind_xx   = xx_1-pad_xx_1:xx_nx+pad_xx_nx
    ind_yy   = yy_1-pad_yy_1:yy_ny+pad_yy_ny

    return (ind_xx, ind_yy)
end # function map_trim

"""
    map_trim(map_map::Map;
             pad::Int       = 0,
             xx_lim::Tuple  = (-Inf,Inf),
             yy_lim::Tuple  = (-Inf,Inf),
             zone_utm::Int  = 18,
             is_north::Bool = true,
             silent::Bool   = false)

Trim map by removing large areas that are missing map data. Returns 
trimmed magnetic anomaly map struct.

**Arguments:**
- `map_map`:  `Map` magnetic anomaly map struct
- `pad`:      (optional) minimum padding along map edges
- `xx_lim`:   (optional) x-direction map limits (xx_min,xx_max) [rad]
- `yy_lim`:   (optional) y-direction map limits (yy_min,yy_max) [rad]
- `zone_utm`: (optional) UTM zone
- `is_north`: (optional) if true, map is in northern hemisphere
- `silent`:   (optional) if true, no print outs

**Returns:**
- `map_out`: trimmed `Map` magnetic anomaly map struct
"""
function map_trim(map_map::Map;
                  pad::Int       = 0,
                  xx_lim::Tuple  = (-Inf,Inf),
                  yy_lim::Tuple  = (-Inf,Inf),
                  zone_utm::Int  = 18,
                  is_north::Bool = true,
                  silent::Bool   = false)

    if typeof(map_map) <: Union{MapS,MapSd} # scalar map
        (ind_xx,ind_yy) = map_trim(map_map.map,map_map.xx,map_map.yy,map_map.alt;
                                   pad=pad,xx_lim=xx_lim,yy_lim=yy_lim,
                                   zone_utm=zone_utm,is_north=is_north,
                                   map_units=:rad,silent=silent)
        if typeof(map_map) <: MapS # 2D
            return MapS(map_map.map[ind_yy,ind_xx],map_map.xx[ind_xx],
                        map_map.yy[ind_yy],map_map.alt)
        else # 3D
            return MapSd(map_map.map[ind_yy,ind_xx],map_map.xx[ind_xx],
                         map_map.yy[ind_yy],map_map.alt[ind_yy,ind_xx])
        end
    else # vector map
        (ind_xx,ind_yy) = map_trim(map_map.mapX,map_map.xx,map_map.yy,map_map.alt;
                                   pad=pad,xx_lim=xx_lim,yy_lim=yy_lim,
                                   zone_utm=zone_utm,is_north=is_north,
                                   map_units=:rad,silent=silent)
        if typeof(map_map) <: MapV # 2D
            return MapV(map_map.mapX[ind_yy,ind_xx],map_map.mapY[ind_yy,ind_xx],
                        map_map.mapZ[ind_yy,ind_xx],map_map.xx[ind_xx],
                        map_map.yy[ind_yy],map_map.alt)
        else
            return MapVd(map_map.mapX[ind_yy,ind_xx],map_map.mapY[ind_yy,ind_xx],
                         map_map.mapZ[ind_yy,ind_xx],map_map.xx[ind_xx],
                         map_map.yy[ind_yy],map_map.alt[ind_yy,ind_xx])
        end
    end

end # function map_trim

"""
    map_trim(map_map::Map, path::Path;
             pad::Int       = 0,
             zone_utm::Int  = 18,
             is_north::Bool = true,
             silent::Bool   = true)

Trim map by removing large areas far away from `path`. Do not use prior to 
upward continuation, as it will result in edge effect errors. Returns trimmed 
magnetic anomaly map struct.

**Arguments:**
- `map_map`:  `Map` magnetic anomaly map struct
- `path`:     `Path` struct, i.e. `Traj` trajectory struct, `INS` inertial navigation system struct, or `FILTout` filter extracted output struct
- `pad`:      (optional) minimum padding along map edges
- `zone_utm`: (optional) UTM zone
- `is_north`: (optional) if true, map is in northern hemisphere
- `silent`:   (optional) if true, no print outs

**Returns:**
- `map_out`: trimmed `Map` magnetic anomaly map struct
"""
function map_trim(map_map::Map, path::Path;
                  pad::Int       = 0,
                  zone_utm::Int  = 18,
                  is_north::Bool = true,
                  silent::Bool   = true)
    map_trim(map_map;
             pad      = pad,
             xx_lim   = extrema(path.lon),
             yy_lim   = extrema(path.lat),
             zone_utm = zone_utm,
             is_north = is_north,
             silent   = silent)
end # function map_trim

"""
    map_correct_igrf!(map_map::Matrix, map_alt,
                      map_xx::Vector, map_yy::Vector;
                      sub_igrf_date  = 2013+293/365, # 20-Oct-2013
                      add_igrf_date  = -1,
                      zone_utm::Int  = 18,
                      is_north::Bool = true)

Correct the International Geomagnetic Reference Field (IGRF), i.e. core field, 
of a map by subtracting and/or adding the IGRF on specified date(s).

**Arguments:**
- `map_map`:       `ny` x `nx` 2D gridded map data
- `map_alt`:       `ny` x `nx` 2D gridded altitude map data [m]
- `map_xx`:        `nx` x-direction map coordinates
- `map_yy`:        `ny` y-direction map coordinates
- `sub_igrf_date`: (optional) date of IGRF core field to subtract [yr], -1 to ignore
- `add_igrf_date`: (optional) date of IGRF core field to add [yr], -1 to ignore
- `zone_utm`:      (optional) UTM zone
- `is_north`:      (optional) if true, map is in northern hemisphere

**Returns:**
- `map_map`: IGRF-corrected 2D gridded map data (mutated)
"""
function map_correct_igrf!(map_map::Matrix, map_alt,
                           map_xx::Vector, map_yy::Vector;
                           sub_igrf_date  = 2013+293/365,
                           add_igrf_date  = -1,
                           zone_utm::Int  = 18,
                           is_north::Bool = true)

    (_,ind1,nx,ny) = map_params(map_map,map_xx,map_yy)

    all(map_alt .< 0) && (map_alt = 300) # in case drape map altitude provided (uses alt = -1)
    length(map_alt) == 1 && (map_alt = map_alt*one.(map_map)) # in case scalar altitude provided

    sub_igrf = sub_igrf_date > 0 ? true : false
    add_igrf = add_igrf_date > 0 ? true : false

    if sub_igrf | add_igrf

        utm2lla = LLAfromUTMZ(WGS84)

        @info("starting igrf")

        for i = 1:nx # time consumer
            for j = 1:ny
                if ind1[j,i]
                    lla = utm2lla(UTMZ(map_xx[i],map_yy[j],map_alt[j,i],
                                       zone_utm,is_north))
                    if sub_igrf
                        map_map[j,i] -= norm(igrfd(sub_igrf_date,lla.alt,lla.lat,
                                                   lla.lon,Val(:geodetic)))
                    end
                    if add_igrf
                        map_map[j,i] += norm(igrfd(add_igrf_date,lla.alt,lla.lat,
                                                   lla.lon,Val(:geodetic)))
                    end
                end
            end
        end
    end

end # function map_correct_igrf!

"""
    map_correct_igrf!(mapS::Union{MapS,MapSd};
                      sub_igrf_date  = 2013+293/365, # 20-Oct-2013
                      add_igrf_date  = -1,
                      zone_utm::Int  = 18,
                      is_north::Bool = true)

Correct the International Geomagnetic Reference Field (IGRF), i.e. core field, 
of a map by subtracting and/or adding the IGRF on specified date(s).

**Arguments:**
- `mapS`:          `MapS` or `MapSd` scalar magnetic anomaly map struct  
- `sub_igrf_date`: (optional) date of IGRF core field to subtract [yr], -1 to ignore
- `add_igrf_date`: (optional) date of IGRF core field to add [yr], -1 to ignore
- `zone_utm`:      (optional) UTM zone
- `is_north`:      (optional) if true, map is in northern hemisphere

**Returns:**
- `mapS`: IGRF-corrected `MapS` or `MapSd` scalar magnetic anomaly map struct (mutated)
"""
function map_correct_igrf!(mapS::Union{MapS,MapSd};
                           sub_igrf_date  = 2013+293/365,
                           add_igrf_date  = -1,
                           zone_utm::Int  = 18,
                           is_north::Bool = true)
    map_correct_igrf!(mapS.map,mapS.alt,mapS.xx,mapS.yy;
                      sub_igrf_date = sub_igrf_date,
                      add_igrf_date = add_igrf_date,
                      zone_utm      = zone_utm,
                      is_north      = is_north)
end # function map_correct_igrf!

"""
    map_fill!(map_map::Matrix, map_xx::Vector, map_yy::Vector; k::Int=3)

Fill map areas that are missing map data.

**Arguments:**
- `map_map`: `ny` x `nx` 2D gridded map data
- `map_xx`:  `nx` x-direction map coordinates
- `map_yy`:  `ny` y-direction map coordinates
- `k`:       (optional) number of nearest neighbors for knn

**Returns:**
- `map_map`: `ny` x `nx` filled 2D gridded map data (mutated)
"""
function map_fill!(map_map::Matrix, map_xx::Vector, map_yy::Vector; k::Int=3)

    (ind0,ind1,nx,ny) = map_params(map_map,map_xx,map_yy)

    data = vcat(vec(repeat(map_xx', ny)[ind1])',
                vec(repeat(map_yy,1,nx)[ind1])')
    pts  = vcat(vec(repeat(map_xx', ny)[ind0])',
                vec(repeat(map_yy,1,nx)[ind0])')
    vals = vec(map_map[ind1])
    tree = KDTree(data)
    inds = knn(tree,pts,k,true)[1]

    j = 1
    for i in eachindex(map_map)
        if ind0[i]
            @inbounds map_map[i] = mean(vals[inds[j]])
            j += 1
        end
    end

end # function map_fill!

"""
    map_fill!(mapS::Union{MapS,MapSd}; k::Int=3)

Fill map areas that are missing map data.

**Arguments:**
- `mapS`: `MapS` or `MapSd` scalar magnetic anomaly map struct
- `k`: (optional) number of nearest neighbors for knn

**Returns:**
- `mapS`: filled `MapS` or `MapSd` scalar magnetic anomaly map struct (mutated)
"""
function map_fill!(mapS::Union{MapS,MapSd}; k::Int=3)
    map_fill!(mapS.map,mapS.xx,mapS.yy;k=k)
end # function map_fill!

"""
    map_chessboard!(map_map::Matrix, map_alt::Matrix, map_xx::Vector, map_yy::Vector, alt;
                    down_cont::Bool = true,
                    dz              = 5,
                    down_max        = 150,
                    α               = 200)

The "chessboard method", which upward (and possibly downward) continues a map 
to multiple altitudes to create a 3D map, then vertically interpolates at each 
horizontal grid point.

Reference: Cordell, Phillips, & Godson, U.S. Geological Survey Potential-Field 
Software Version 2.0, 1992.

**Arguments:**
- `map_map`:  `ny` x `nx` 2D gridded target (e.g. magnetic) map data
- `map_alt`:  `ny` x `nx` 2D gridded altitude map data [m]
- `map_xx`:   `nx` x-direction map coordinates [m]
- `map_yy`:   `ny` y-direction map coordinates [m]
- `alt`:      final map altitude after upward continuation [m], -1 for drape map
- `down_cont`:(optional) if true, downward continue if needed, only used if `up_cont = true`
- `dz`:       (optional) upward continuation step size [m]
- `down_max`: (optional) maximum downward continuation distance [m]
- `α`:        (optional) regularization parameter for downward continuation

**Returns:**
- `map_map`: upward continued 2D gridded map data (mutated)
"""
function map_chessboard!(map_map::Matrix, map_alt::Matrix, map_xx::Vector, map_yy::Vector, alt;
                         down_cont::Bool = true,
                         dz              = 5,
                         down_max        = 150,
                         α               = 200)

    (_,ind1,nx,ny) = map_params(map_map,map_xx,map_yy)

    # map sample intervals
    dx = abs(map_xx[end]-map_xx[1]) / (nx-1)
    dy = abs(map_yy[end]-map_yy[1]) / (ny-1)

    alt_min = floor(minimum(map_alt[ind1]))
    alt_max = ceil( maximum(map_alt[ind1]))
    alt_dif_down = clamp(alt_max - alt, 0, down_max)
    alt_dif_up   = clamp(alt - alt_min, 0, 500)
    alt_lev_down = 0:dz:alt_dif_down+dz # downward continuation levels 
    alt_lev_up   = 0:dz:alt_dif_up+dz   # upward   continuation levels

    if down_cont
        alt_lev = -alt_lev_down[end]:dz:alt_lev_up[end]
        k0 = length(alt_lev_down)
    else
        alt_lev = alt_lev_up
        k0 = 1
    end

    nz = length(alt_lev)
    map_d = zeros(ny,nx,nz)

    @info("starting upward and/or downward continuation with $nz levels")

    for k = 1:nz # time consumer
        if k == k0
            map_d[:,:,k] = deepcopy(map_map)
        else
            @inbounds map_d[:,:,k] = upward_fft(map_map,dx,dy,alt_lev[k];
                                                 expand=true,α=α)
        end
    end

    @info("starting chessboard interpolation")

    # interpolate vertical direction at each grid point
    for i = 1:nx
        for j = 1:ny
            if map_alt[j,i] + alt_lev[1] > alt # desired map below data
                map_map[j,i] = map_d[j,i,1] # take lowest (closest) value
            elseif map_alt[j,i] + alt_lev[end] < alt # desired map above data
                map_map[j,i] = map_d[j,i,end] # take highest (closest) value
            elseif ind1[j,i] # altitude data is available
                itp = interpolate(map_d[j,i,:],BSpline(Linear()))
                map_map[j,i] = scale(itp,map_alt[j,i].+alt_lev)(alt)
            end
        end
    end

end # function map_chessboard!

"""
    map_chessboard(mapSd::MapSd, alt;
                   down_cont::Bool = true,
                   dz              = 5,
                   down_max        = 150,
                   α               = 200)

The "chessboard method", which upward (and possibly downward) continues a map 
to multiple altitudes to create a 3D map, then vertically interpolates at each 
horizontal grid point.

Reference: Cordell, Phillips, & Godson, U.S. Geological Survey Potential-Field 
Software Version 2.0, 1992.

**Arguments:**
- `MapSd`:   `MapSd` scalar magnetic anomaly map struct with UTMZ map grid
- `alt`:      final map altitude after upward continuation [m], -1 for drape map
- `down_cont`:(optional) if true, downward continue if needed
- `dz`:       (optional) upward continuation step size [m]
- `down_max`: (optional) maximum downward continuation distance [m]
- `α`:        (optional) regularization parameter for downward continuation

**Returns:**
- `map_map`: `MapS` scalar magnetic anomaly map struct with UTMZ map grid
"""
function map_chessboard(mapSd::MapSd, alt;
                        down_cont::Bool = true,
                        dz              = 5,
                        down_max        = 150,
                        α               = 200)
    map_chessboard!(mapSd.map,mapSd.alt,mapSd.xx,mapSd.yy,alt;
                    down_cont = down_cont,
                    dz        = dz,
                    down_max  = down_max,
                    α         = α)
    return MapS(mapSd.map, mapSd.xx, mapSd.yy, alt)
end # function map_chessboard

"""
    map_utm2lla!(map_map::Matrix, map_xx::Vector, map_yy::Vector, alt;
                 zone_utm::Int  = 18,
                 is_north::Bool = true,
                 save_h5::Bool  = false,
                 map_h5::String = "map.h5")

Convert map grid from UTMZ to LLA.

**Arguments:**
- `map_map`:  `ny` x `nx` 2D gridded map data with UTMZ map grid
- `map_xx`:   `nx` x-direction map coordinates [m]
- `map_yy`:   `ny` y-direction map coordinates [m]
- `alt`:      nominal map altitude, median used for 2D altitude map [m]
- `zone_utm`: (optional) UTM zone
- `is_north`: (optional) if true, map is in northern hemisphere
- `save_h5`:  (optional) if true, save HDF5 file `map_h5`
- `map_h5`:   (optional) path/name of HDF5 file to save with map data

**Returns:**
- `mapS`: `MapS` or `MapSd` scalar magnetic anomaly map struct with LLA map grid
"""
function map_utm2lla!(map_map::Matrix, map_xx::Vector, map_yy::Vector, alt;
                      zone_utm::Int  = 18,
                      is_north::Bool = true,
                      save_h5::Bool  = false,
                      map_h5::String = "map.h5")

    (_,ind1,nx,ny) = map_params(map_map,map_xx,map_yy)

    map_alt = deepcopy(alt)
    map_d   = length(alt) > 1 ? true : false
    map_d   && (alt = median(alt[ind1])) # in case 2D altitude map provided
    alt < 0 && (alt = 300) # in case drape map altitude (-1) provided

    # interpolation for original (UTMZ) map
    itp_map = map_itp(map_map,map_xx,map_yy,:linear)
    map_d && (itp_alt = map_itp(map_alt,map_xx,map_yy,:linear))

    # get xx/yy limits at 4 corners of data-containing UTMZ map for no data loss
    (lons,lats) = map_lla_lim(map_xx,map_yy,alt,1,nx,1,ny;
                              zone_utm  = zone_utm,
                              is_north  = is_north)

    # use interior 2 lons/lats as xx/yy limits for new (LLA) map (stay in range)
    δ = 1e-10 # ad hoc to solve rounding related error
    map_xx .= [LinRange(lons[2]+δ,lons[3]-δ,nx);]
    map_yy .= [LinRange(lats[2]+δ,lats[3]-δ,ny);]

    # interpolate original (UTMZ) map with grid for new (LLA) map
    lla2utm = UTMZfromLLA(WGS84)
    for i = 1:nx
        for j = 1:ny
            utm_temp = lla2utm(LLA(map_yy[j],map_xx[i],alt))
            @inbounds map_map[j,i] = itp_map(utm_temp.x,utm_temp.y)
            map_d && (@inbounds map_alt[j,i] = itp_alt(utm_temp.x,utm_temp.y))
        end
    end

    # convert to [rad]
    map_xx .= deg2rad.(map_xx)
    map_yy .= deg2rad.(map_yy)

    save_h5 && save_map(map_map,map_xx,map_yy,map_alt,map_h5;map_units=:deg)

    map_d || return MapS( map_map, map_xx, map_yy, convert(eltype(map_map),map_alt))
    map_d && return MapSd(map_map, map_xx, map_yy, map_alt)
end # function map_utm2lla!

"""
    map_utm2lla!(mapS::Union{MapS,MapSd};
                 zone_utm::Int  = 18,
                 is_north::Bool = true,
                 save_h5::Bool  = false
                 map_h5::String = "map.h5")

Convert map grid from UTMZ to LLA.

**Arguments:**
- `mapS`:     `MapS` or `MapSd` scalar magnetic anomaly map struct with UTMZ map grid
- `zone_utm`: (optional) UTM zone
- `is_north`: (optional) if true, map is in northern hemisphere
- `save_h5`:  (optional) if true, save HDF5 file `map_h5`
- `map_h5`:   (optional) path/name of HDF5 file to save with map data

**Returns:**
- `mapS`: `MapS` or `MapSd` scalar magnetic anomaly map struct with LLA map grid (mutated)
"""
function map_utm2lla!(mapS::Union{MapS,MapSd};
                      zone_utm::Int  = 18,
                      is_north::Bool = true,
                      save_h5::Bool  = false,
                      map_h5::String = "map.h5")
    map_utm2lla!(mapS.map,mapS.xx,mapS.yy,mapS.alt;
                 zone_utm  = zone_utm,
                 is_north  = is_north,
                 save_h5   = save_h5,
                 map_h5    = map_h5)
end # function map_utm2lla!

"""
    map_gxf2h5(map_gxf::String, alt_gxf::String, alt;
               pad::Int        = 0,
               sub_igrf_date   = 2013+293/365, # 20-Oct-2013
               add_igrf_date   = -1,
               zone_utm::Int   = 18,
               is_north::Bool  = true,
               fill_map::Bool  = true,
               up_cont::Bool   = true,
               down_cont::Bool = true,
               get_lla::Bool   = true,
               dz              = 5,
               down_max        = 150,
               α               = 200,
               save_h5::Bool   = false,
               map_h5::String  = "map.h5")

Convert map file from .gxf to HDF5. The order of operations is: 
- original => 
- trim away large areas that are missing data => 
- subtract and/or add igrf => 
- fill remaining areas that are missing data => 
- upward continue to `alt` => 
- convert grid from UTMZ to LLA
This can be memory intensive, largely depending on the map size and `dz`. If 
`up_cont` = true, a `MapS` struct (2D map) is returned. If `up_cont` = false, 
a `MapSd` struct is returned, while has an included altitude map.

**Arguments:**
- `map_gxf`:       path/name of .gxf file with target (e.g. magnetic) map data
- `alt_gxf`:       path/name of .gxf file with altitude map data
- `alt`:           final map altitude after upward continuation [m], -1 for drape map
- `pad`:           (optional) minimum padding along map edges
- `sub_igrf_date`: (optional) date of IGRF core field to subtract [yr], -1 to ignore
- `add_igrf_date`: (optional) date of IGRF core field to add [yr], -1 to ignore
- `zone_utm`:      (optional) UTM zone
- `is_north`:      (optional) if true, map is in northern hemisphere
- `fill_map`:      (optional) if true, fill areas that are missing map data
- `up_cont`:       (optional) if true, upward continue to `alt`
- `down_cont`:     (optional) if true, downward continue if needed, only used if `up_cont = true`
- `get_lla`:       (optional) if true, convert map grid from UTMZ to LLA
- `dz`:            (optional) upward continuation step size [m]
- `down_max`:      (optional) maximum downward continuation distance [m]
- `α`:             (optional) regularization parameter for downward continuation
- `save_h5`:       (optional) if true, save HDF5 file `map_h5`
- `map_h5`:        (optional) path/name of HDF5 file to save with map data

**Returns:**
- `mapS`: `MapS` or `MapSd` scalar magnetic anomaly map struct
"""
function map_gxf2h5(map_gxf::String, alt_gxf::String, alt;
                    pad::Int        = 0,
                    sub_igrf_date   = 2013+293/365,
                    add_igrf_date   = -1,
                    zone_utm::Int   = 18,
                    is_north::Bool  = true,
                    fill_map::Bool  = true,
                    up_cont::Bool   = true,
                    down_cont::Bool = true,
                    get_lla::Bool   = true,
                    dz              = 5,
                    down_max        = 150,
                    α               = 200,
                    save_h5::Bool   = false,
                    map_h5::String  = "map.h5")

    @info("starting .gxf read")

    # get raw target (e.g. magnetic) and altitude data
    (map_map,map_xx ,map_yy ) = map_get_gxf(map_gxf)
    (map_alt,map_xx_,map_yy_) = map_get_gxf(alt_gxf)

    # make sure grids match
    (map_xx ≈ map_xx_) & (map_yy ≈ map_yy_) || error("grids do not match")

    @info("starting trim")

    # trim away large areas that are missing map data
    (ind_xx,ind_yy) = map_trim(map_map,map_xx,map_yy,alt;
                               pad=pad,zone_utm=zone_utm,is_north=is_north,
                               map_units=:utm)
    map_xx  = map_xx[ind_xx]
    map_yy  = map_yy[ind_yy]
    map_map = map_map[ind_yy,ind_xx]
    map_alt = map_alt[ind_yy,ind_xx]

    # subtract and/or add IGRF
    map_correct_igrf!(map_map,map_alt,map_xx,map_yy;
                      sub_igrf_date = sub_igrf_date,
                      add_igrf_date = add_igrf_date,
                      zone_utm      = zone_utm,
                      is_north      = is_north)

    if fill_map # fill areas that are missing map data
        @info("starting fill")
        map_fill!(map_map,map_xx,map_yy)
        map_fill!(map_alt,map_xx,map_yy)
    end

    if up_cont # upward continue to alt
        map_chessboard!(map_map,map_alt,map_xx,map_yy,alt;
                        down_cont = down_cont,
                        dz        = dz,
                        down_max  = down_max,
                        α         = α)
    end

    if get_lla # convert map grid from UTMZ to LLA
        @info("starting utm2lla")
        if up_cont # return MapS
            return map_utm2lla!(map_map,map_xx,map_yy,alt;
                                zone_utm = zone_utm,
                                is_north = is_north,
                                save_h5  = save_h5,
                                map_h5   = map_h5)
        else # return MapSd
            return map_utm2lla!(map_map,map_xx,map_yy,map_alt;
                                zone_utm = zone_utm,
                                is_north = is_north,
                                save_h5  = save_h5,
                                map_h5   = map_h5)
        end
    else
        if up_cont # return MapS
            save_h5 && save_map(map_map,map_xx,map_yy,alt,map_h5;map_units=:utm)
            return MapS(map_map, map_xx, map_yy, convert(eltype(map_map),alt))
        else # return MapSd
            save_h5 && save_map(map_map,map_xx,map_yy,map_alt,map_h5;map_units=:utm)
            return MapSd(map_map, map_xx, map_yy, map_alt)
        end
    end

end # map_gxf2h5

"""
    map_gxf2h5(map_gxf::String, alt;
               fill_map::Bool = true,
               get_lla::Bool  = true,
               zone_utm::Int  = 18,
               is_north::Bool = true,
               save_h5::Bool  = false,
               map_h5::String = "map.h5")

Convert map file from .gxf to HDF5. The order of operations is: 
- original => 
- fill areas that are missing data => 
- convert grid from UTMZ to LLA 
Specifically meant for SMALL and LEVEL maps ONLY.

**Arguments:**
- `map_gxf`:  path/name of .gxf file with target (e.g. magnetic) map data
- `alt`    :  map altitude [m]
- `fill_map`: (optional) if true, fill areas that are missing map data
- `get_lla`:  (optional) if true, convert map grid from UTMZ to LLA
- `zone_utm`: (optional) UTM zone
- `is_north`: (optional) if true, map is in northern hemisphere
- `save_h5`:  (optional) if true, save HDF5 file `map_h5`
- `map_h5`:   (optional) path/name of HDF5 file to save with map data

**Returns:**
- `mapS`: `MapS` scalar magnetic anomaly map struct
"""
function map_gxf2h5(map_gxf::String, alt;
                    fill_map::Bool = true,
                    get_lla::Bool  = true,
                    zone_utm::Int  = 18,
                    is_north::Bool = true,
                    save_h5::Bool  = false,
                    map_h5::String = "map.h5")

    (map_map,map_xx,map_yy) = map_get_gxf(map_gxf) # get raw map data

    fill_map && map_fill!(map_map,map_xx,map_yy) # fill in missing map data

    if get_lla
        return map_utm2lla!(map_map,map_xx,map_yy,alt;zone_utm=zone_utm,
                            is_north=is_north,save_h5=save_h5,map_h5=map_h5)
    else
        save_h5 && save_map(map_map,map_xx,map_yy,alt,map_h5;map_units=:utm)
        return MapS(map_map,map_xx,map_yy,convert(eltype(map_map),alt))
    end

end # function map_gxf2h5

"""
    plot_map!(p1, map_map::Matrix,
              map_xx::Vector     = [],
              map_yy::Vector     = [];
              clims::Tuple       = (-500,500),
              dpi::Int           = 200,
              margin::Int        = 2,
              legend::Bool       = true,
              axis::Bool         = true,
              fewer_pts::Bool    = true,
              map_color::Symbol  = :usgs,
              bg_color::Symbol   = :white,
              map_units::Symbol  = :rad,
              plot_units::Symbol = :deg,
              b_e                = gr())

Plot map on an existing plot.

**Arguments:**
- `p1`: existing plot
- `map_map`:    `ny` x `nx` 2D gridded map data
- `map_xx`:     `nx` x-direction (longitude) map coordinates [rad] or [deg]
- `map_yy`:     `ny` y-direction (latitude)  map coordinates [rad] or [deg]
- `clims`:      (optional) color scale limits
- `dpi`:        (optional) dots per inch (image resolution)
- `margin`:     (optional) margin around plot [mm]
- `legend`:     (optional) if true, show legend
- `axis`:       (optional) if true, show axes
- `fewer_pts`:  (optional) if true, reduce number of data points plotted
- `map_color`:  (optional) filled contour color scheme {`:usgs`,`:gray`,`:gray1`,`:gray2`}
- `bg_color`:   (optional) background color
- `map_units`:  (optional) map  xx/yy units {`:rad`,`:deg`}
- `plot_units`: (optional) plot xx/yy units {`:rad`,`:deg`,`:utm`,`:m`}
- `b_e`:        (optional) plotting backend

**Returns:**
- `nothing`: map is plotted on `p1`
"""
function plot_map!(p1, map_map::Matrix,
                   map_xx::Vector     = [],
                   map_yy::Vector     = [];
                   clims::Tuple       = (-500,500),
                   dpi::Int           = 200,
                   margin::Int        = 2,
                   legend::Bool       = true,
                   axis::Bool         = true,
                   fewer_pts::Bool    = true,
                   map_color::Symbol  = :usgs,
                   bg_color::Symbol   = :white,
                   map_units::Symbol  = :rad,
                   plot_units::Symbol = :deg,
                   b_e                = gr())

    (ny,nx) = size(map_map)

    # avoid changing map struct data
    map_map = deepcopy(map_map)
    map_xx  = length(map_xx) < nx ? [1:nx;] : deepcopy(map_xx)
    map_yy  = length(map_yy) < ny ? [1:ny;] : deepcopy(map_yy)

    if map_units == :rad
        if plot_units == :deg
            map_xx = rad2deg.(map_xx)
            map_yy = rad2deg.(map_yy)
        elseif plot_units in [:utm,:m] # inaccuracy scales with map size
            mid_xx = floor(Int,nx/2)
            mid_yy = floor(Int,ny/2)
            map_xx = dlon2de.(map_xx .- map_xx[mid_xx], map_yy[mid_yy])
            map_yy = dlat2dn.(map_yy .- map_yy[mid_yy], map_yy[mid_yy])
        end
    elseif map_units == :deg
        if plot_units == :rad
            map_xx = deg2rad.(map_xx)
            map_yy = deg2rad.(map_yy)
        elseif plot_units in [:utm,:m] # inaccuracy scales with map size
            mid_xx = floor(Int,nx/2)
            mid_yy = floor(Int,ny/2)
            map_xx = dlon2de.(deg2rad.(map_xx .- map_xx[mid_xx]),
                              deg2rad( map_yy[mid_yy]))
            map_yy = dlat2dn.(deg2rad.(map_yy .- map_yy[mid_yy]),
                              deg2rad( map_yy[mid_yy]))
        end
    end

    if map_units in [:rad,:deg]
        xlab = (map_xx[end] == nx | !axis) ? "" : "longitude [deg]"
        ylab = (map_yy[end] == ny | !axis) ? "" : "latitude [deg]"
    elseif map_units in [:utm,:m]
        xlab = (map_xx[end] == nx | !axis) ? "" : "easting [m]"
        ylab = (map_yy[end] == ny | !axis) ? "" : "northing [m]"
    else
        error("map_units $map_units  not defined")
    end

    # map indices with zeros (ind0)
    (ind0,_,nx,ny) = map_params(map_map,map_xx,map_yy)

    # select color scale
    c = map_cs(map_color)

    # adjust color scale and set contour limits based on map data
    clims == (0,0) && ((c,clims) = map_clims(c,map_map))

    # values outside contour limits set to contour limits (plotly workaround)
    map_map .= clamp.(map_map,clims[1],clims[2])

    # set data points without actual data to NaN (not plotted)
    map_map[ind0] .= NaN

    xind = 1:nx
    yind = 1:ny

    if fewer_pts
        nx < dpi*6 || (xind = 1:round(Int,nx/(dpi*6)):nx)
        ny < dpi*6 || (yind = 1:round(Int,ny/(dpi*6)):ny)
    end

    b_e # backend
    contourf!(p1,map_xx[xind],map_yy[yind],map_map[yind,xind],dpi=dpi,lw=0,
              c=c,bg=bg_color,clims=clims,margin=margin*mm,legend=legend,
              axis=axis,xticks=axis,yticks=axis,xlab=xlab,ylab=ylab,lab=false);

end # function plot_map!

"""
    plot_map!(p1, mapS::Union{MapS,MapSd};
              clims::Tuple       = (-500,500),
              dpi::Int           = 200,
              margin::Int        = 2,
              legend::Bool       = true,
              axis::Bool         = true,
              fewer_pts::Bool    = true,
              map_color::Symbol  = :usgs,
              bg_color::Symbol   = :white,
              map_units::Symbol  = :rad,
              plot_units::Symbol = :deg
              b_e                = gr())

Plot map on an existing plot.

**Arguments:**
- `p1`:         existing plot
- `mapS`:       `MapS` or `MapSd` scalar magnetic anomaly map struct
- `clims`:      (optional) color scale limits
- `dpi`:        (optional) dots per inch (image resolution)
- `margin`:     (optional) margin around plot [mm]
- `legend`:     (optional) if true, show legend
- `axis`:       (optional) if true, show axes
- `fewer_pts`:  (optional) if true, reduce number of data points plotted
- `map_color`:  (optional) filled contour color scheme {`:usgs`,`:gray`,`:gray1`,`:gray2`}
- `bg_color`:   (optional) background color
- `map_units`:  (optional) map  xx/yy units {`:rad`,`:deg`}
- `plot_units`: (optional) plot xx/yy units {`:rad`,`:deg`,`:utm`,`:m`}
- `b_e`:        (optional) plotting backend

**Returns:**
- `nothing`: map is plotted on `p1`
"""
function plot_map!(p1, mapS::Union{MapS,MapSd};
                   clims::Tuple       = (-500,500),
                   dpi::Int           = 200,
                   margin::Int        = 2,
                   legend::Bool       = true,
                   axis::Bool         = true,
                   fewer_pts::Bool    = true,
                   map_color::Symbol  = :usgs,
                   bg_color::Symbol   = :white,
                   map_units::Symbol  = :rad,
                   plot_units::Symbol = :deg,
                   b_e                = gr())
    plot_map!(p1,
              mapS.map,mapS.xx,mapS.yy;
              clims      = clims,
              dpi        = dpi,
              margin     = margin,
              legend     = legend,
              axis       = axis,
              fewer_pts  = fewer_pts,
              map_color  = map_color,
              bg_color   = bg_color,
              map_units  = map_units,
              plot_units = plot_units,
              b_e        = b_e)
end # function plot_map!

"""
    plot_map(map_map::Matrix,
             map_xx::Vector     = [],
             map_yy::Vector     = [];
             clims::Tuple       = (0,0),
             dpi::Int           = 200,
             margin::Int        = 2,
             legend::Bool       = true,
             axis::Bool         = true,
             fewer_pts::Bool    = true,
             map_color::Symbol  = :usgs,
             bg_color::Symbol   = :white,
             map_units::Symbol  = :rad,
             plot_units::Symbol = :deg,
             b_e                = gr())

Plot map.

**Arguments:**
- `map_map`:    `ny` x `nx` 2D gridded map data
- `map_xx`:     `nx` x-direction (longitude) map coordinates [rad] or [deg]
- `map_yy`:     `ny` y-direction (latitude)  map coordinates [rad] or [deg]
- `clims`:      (optional) color scale limits
- `dpi`:        (optional) dots per inch (image resolution)
- `margin`:     (optional) margin around plot [mm]
- `legend`:     (optional) if true, show legend
- `axis`:       (optional) if true, show axes
- `fewer_pts`:  (optional) if true, reduce number of data points plotted
- `map_color`:  (optional) filled contour color scheme {`:usgs`,`:gray`,`:gray1`,`:gray2`}
- `bg_color`:   (optional) background color
- `map_units`:  (optional) map  xx/yy units {`:rad`,`:deg`}
- `plot_units`: (optional) plot xx/yy units {`:rad`,`:deg`,`:utm`,`:m`}
- `b_e`:        (optional) plotting backend

**Returns:**
- `p1`: plot with map
"""
function plot_map(map_map::Matrix,
                  map_xx::Vector     = [],
                  map_yy::Vector     = [];
                  clims::Tuple       = (0,0),
                  dpi::Int           = 200,
                  margin::Int        = 2,
                  legend::Bool       = true,
                  axis::Bool         = true,
                  fewer_pts::Bool    = true,
                  map_color::Symbol  = :usgs,
                  bg_color::Symbol   = :white,
                  map_units::Symbol  = :rad,
                  plot_units::Symbol = :deg,
                  b_e                = gr())
    b_e # backend
    p1 = plot(legend=legend,lab=false)
    plot_map!(p1,map_map,map_xx,map_yy;
              clims      = clims,
              dpi        = dpi,
              margin     = margin,
              legend     = legend,
              axis       = axis,
              fewer_pts  = fewer_pts,
              map_color  = map_color,
              bg_color   = bg_color,
              map_units  = map_units,
              plot_units = plot_units,
              b_e        = b_e)
end # function plot_map

"""
    plot_map(mapS::Union{MapS,MapSd};
             clims::Tuple       = (0,0),
             dpi::Int           = 200,
             margin::Int        = 2,
             legend::Bool       = true,
             axis::Bool         = true,
             fewer_pts::Bool    = true,
             map_color::Symbol  = :usgs,
             bg_color::Symbol   = :white,
             map_units::Symbol  = :rad,
             plot_units::Symbol = :deg,
             b_e                = gr())

Plot map.

**Arguments:**
- `mapS`:       `MapS` or `MapSd` scalar magnetic anomaly map struct
- `clims`:      (optional) color scale limits
- `dpi`:        (optional) dots per inch (image resolution)
- `margin`:     (optional) margin around plot [mm]
- `legend`:     (optional) if true, show legend
- `axis`:       (optional) if true, show axes
- `fewer_pts`:  (optional) if true, reduce number of data points plotted
- `map_color`:  (optional) filled contour color scheme {`:usgs`,`:gray`,`:gray1`,`:gray2`}
- `bg_color`:   (optional) background color
- `map_units`:  (optional) map  xx/yy units {`:rad`,`:deg`}
- `plot_units`: (optional) plot xx/yy units {`:rad`,`:deg`,`:utm`,`:m`}
- `b_e`:        (optional) plotting backend

**Returns:**
- `p1`: plot with map
"""
function plot_map(mapS::Union{MapS,MapSd};
                  clims::Tuple       = (0,0),
                  dpi::Int           = 200,
                  margin::Int        = 2,
                  legend::Bool       = true,
                  axis::Bool         = true,
                  fewer_pts::Bool    = true,
                  map_color::Symbol  = :usgs,
                  bg_color::Symbol   = :white,
                  map_units::Symbol  = :rad,
                  plot_units::Symbol = :deg,
                  b_e                = gr())
    b_e # backend
    p1 = plot(legend=legend,lab=false)
    plot_map!(p1,mapS.map,mapS.xx,mapS.yy;
              clims      = clims,
              dpi        = dpi,
              margin     = margin,
              legend     = legend,
              axis       = axis,
              fewer_pts  = fewer_pts,
              map_color  = map_color,
              bg_color   = bg_color,
              map_units  = map_units,
              plot_units = plot_units,
              b_e        = b_e)
end # function plot_map

"""
    map_cs(map_color::Symbol=:usgs)

Select map color scale. Default is from the USGS: 
https://mrdata.usgs.gov/magnetic/namag.png

**Arguments:**
- `map_color`: (optional) filled contour color scheme {`:usgs`,`:gray`,`:gray1`,`:gray2`}

**Returns:**
- `c`: color scale
"""
function map_cs(map_color::Symbol=:usgs)

    if map_color == :usgs # standard for geological maps
        f = readdlm(usgs,',')
        c = cgrad([RGB(f[i,:]...) for i = 1:size(f,1)])
    elseif map_color == :gray # light gray
        c = cgrad(:gist_gray)[61:90]
    elseif map_color == :gray1 # light gray (lower end)
        c = cgrad(:gist_gray)[61:81]
    elseif map_color == :gray2 # light gray (upper end)
        c = cgrad(:gist_gray)[71:90]
    else # :viridis, :plasma, :inferno, :magma
        c = cgrad(map_color)
    end

    return (c)
end # function map_cs

"""
    map_clims(c, map_map)

Adjust color scale for histogram equalization (maximum contrast) and set 
contour limits based on map data.

**Arguments:**
- `c`: original color scale
- `map_map`: `ny` x `nx` 2D gridded map data

**Returns:**
- `c`: new color scale
- `clims`: contour limits
"""
function map_clims(c, map_map)

    lc    = length(c) # length of original color scale
    ind1  = abs.(map_map) .>= 1e-3 # map indices without (approximately) zeros
    indc  = round.(Int,LinRange(0.5,lc-0.5,lc)/lc*sum(ind1)) # bin indices
    bcen  = sort(map_map[ind1])[indc] # bin centers
    bwid  = fdm(bcen) # bin widths
    nc    = round.(Int,bwid/minimum(bwid)) # times to repeat each color
    c     = cgrad([c[i] for i = 1:lc for j = 1:nc[i]]) # new color scale
    clims = (bcen[1] - bwid[1]/2, bcen[end] + bwid[end]/2) # contour limits

    return (c, clims)
end # function map_clims

"""
    plot_path!(p1, lat, lon;
               lab = "",
               fewer_pts::Bool    = true,
               show_plot::Bool    = true,
               zoom_plot::Bool    = false,
               path_color::Symbol = :ignore)

Plot flight path on an existing plot.

**Arguments:**
- `p1`:         existing plot
- `lat`:        latitude  [rad]
- `lon`:        longitude [rad]
- `lab`:        (optional) data (legend) label
- `fewer_pts`:  (optional) if true, reduce number of data points plotted
- `show_plot`:  (optional) if true, show plot
- `zoom_plot`:  (optional) if true, zoom plot onto flight path
- `path_color`: (optional) path color {`:ignore`,`:black`,`:gray`,`:red`,`:orange`,`:yellow`,`:green`,`:cyan`,`:blue`,`:purple`}

**Returns:**
- `nothing`: flight path is plotted on `p1`
"""
function plot_path!(p1, lat, lon;
                    lab = "",
                    fewer_pts::Bool    = true,
                    show_plot::Bool    = true,
                    zoom_plot::Bool    = false,
                    path_color::Symbol = :ignore)

    lon = rad2deg.(deepcopy(lon))
    lat = rad2deg.(deepcopy(lat))

    N    = length(lat)
    Nmax = 5000
    if fewer_pts & (N > Nmax)
        lat = lat[1:round(Int,N/Nmax):N]
        lon = lon[1:round(Int,N/Nmax):N]
    end

    if path_color in [:black,:gray,:red,:orange,:yellow,:green,:cyan,:blue,:purple]
        p1 = plot!(p1,lon,lat,lab=lab,legend=true,lc=path_color)
    else
        p1 = plot!(p1,lon,lat,lab=lab,legend=true)
    end

    if zoom_plot
        xlim = extrema(lon) .+ [-0.2,0.2]*(extrema(lon)[2] - extrema(lon)[1])
        ylim = extrema(lat) .+ [-0.2,0.2]*(extrema(lat)[2] - extrema(lat)[1])
        p1 = plot!(p1,xlim=xlim,ylim=ylim)
    end

    show_plot && display(p1)

    return (p1)
end # function plot_path!

"""
    plot_path!(p1, path::Path, ind=trues(length(path.lat));
               lab = "",
               fewer_pts::Bool = true,
               show_plot::Bool = true,
               zoom_plot::Bool = false,
               path_color::Symbol = :ignore)

Plot flight path on an existing plot.

**Arguments:**
- `p1`: existing plot
- `path`: `Path` struct, i.e. `Traj` trajectory struct, `INS` inertial navigation system struct, or `FILTout` filter extracted output struct
- `ind`:        (optional) selected data indices
- `lab`:        (optional) data (legend) label
- `fewer_pts`:  (optional) if true, reduce number of data points plotted
- `show_plot`:  (optional) if true, show plot
- `zoom_plot`:  (optional) if true, zoom plot onto flight path
- `path_color`: (optional) path color {`:ignore`,`:black`,`:gray`,`:red`,`:orange`,`:yellow`,`:green`,`:cyan`,`:blue`,`:purple`}

**Returns:**
- `nothing`: flight path is plotted on `p1`
"""
function plot_path!(p1, path::Path, ind=trues(length(path.lat));
                    lab = "",
                    fewer_pts::Bool    = true,
                    show_plot::Bool    = true,
                    zoom_plot::Bool    = false,
                    path_color::Symbol = :ignore)
    plot_path!(p1,path.lat[ind],path.lon[ind];
               lab        = lab,
               fewer_pts  = fewer_pts,
               show_plot  = show_plot,
               zoom_plot  = zoom_plot,
               path_color = path_color)
end # function plot_path!

"""
    plot_path(lat, lon;
              lab = "",
              dpi::Int           = 200,
              margin::Int        = 2,
              fewer_pts::Bool    = false,
              show_plot::Bool    = true,
              zoom_plot::Bool    = true,
              path_color::Symbol = :ignore)

Plot flight path.

**Arguments:**
- `lat`:        latitude  [rad]
- `lon`:        longitude [rad]
- `lab`:        (optional) data (legend) label
- `dpi`:        (optional) dots per inch (image resolution)
- `margin`:     (optional) margin around plot [mm]
- `fewer_pts`:  (optional) if true, reduce number of data points plotted
- `show_plot`:  (optional) if true, show plot
- `zoom_plot`:  (optional) if true, zoom plot onto flight path
- `path_color`: (optional) path color {`:ignore`,`:black`,`:gray`,`:red`,`:orange`,`:yellow`,`:green`,`:cyan`,`:blue`,`:purple`}

**Returns:**
- `p1`: plot with flight path
"""
function plot_path(lat, lon;
                   lab = "",
                   dpi::Int           = 200,
                   margin::Int        = 2,
                   fewer_pts::Bool    = false,
                   show_plot::Bool    = true,
                   zoom_plot::Bool    = true,
                   path_color::Symbol = :ignore)
    p1 = plot(xlab="latitude [deg]",ylab="latitude [deg]",
              dpi=dpi,margin=margin*mm)
    plot_path!(p1,lat,lon;
               lab        = lab,
               fewer_pts  = fewer_pts,
               show_plot  = show_plot,
               zoom_plot  = zoom_plot,
               path_color = path_color)
end # function plot_path

"""
    plot_path(path::Path, ind=trues(length(path.lat));
              lab = "",
              dpi::Int        = 200,
              margin::Int     = 2,
              fewer_pts::Bool = false,
              show_plot::Bool = true,
              zoom_plot::Bool = true,
              path_color::Symbol = :ignore)

Plot flight path.

**Arguments:**
- `path`: `Path` struct, i.e. `Traj` trajectory struct, `INS` inertial navigation system struct, or `FILTout` filter extracted output struct
- `ind`:        (optional) selected data indices
- `lab`:        (optional) data (legend) label
- `dpi`:        (optional) dots per inch (image resolution)
- `margin`:     (optional) margin around plot [mm]
- `fewer_pts`:  (optional) if true, reduce number of data points plotted
- `show_plot`:  (optional) if true, show plot
- `zoom_plot`:  (optional) if true, zoom plot onto flight path
- `path_color`: (optional) path color {`:ignore`,`:black`,`:gray`,`:red`,`:orange`,`:yellow`,`:green`,`:cyan`,`:blue`,`:purple`}

**Returns:**
- `p1`: plot with flight path
"""
function plot_path(path::Path, ind=trues(length(path.lat));
                   lab = "",
                   dpi::Int        = 200,
                   margin::Int     = 2,
                   fewer_pts::Bool = false,
                   show_plot::Bool = true,
                   zoom_plot::Bool = true,
                   path_color::Symbol = :ignore)
    plot_path(path.lat[ind],path.lon[ind];
              lab        = lab,
              dpi        = dpi,
              margin     = margin,
              fewer_pts  = fewer_pts,
              show_plot  = show_plot,
              zoom_plot  = zoom_plot,
              path_color = path_color)
end # function plot_path

"""
    plot_events!(p1, t, lab=nothing; ylim=ylims(p1), t_units::Symbol=:sec,
                 legend::Symbol=:outertopright)

Plot in-flight event on existing plot.

**Arguments:**
- `p1`:      existing plot
- `t`:       time of in-flight event
- `lab`:     (optional) in-flight event (legend) label
- `ylim`:    (optional) 2-element y limits for in-flight event
- `t_units`: (optional) time units used for plotting {`:sec`,`:min`}
- `legend`:  (optional) position of legend

**Returns:**
- `p1`: in-flight events are plotted on `p1`
"""
function plot_events!(p1, t, lab=nothing; ylim=ylims(p1), t_units::Symbol=:sec,
                      legend::Symbol=:outertopright)
    ylim = ylim .+ 0.05 .* (1,-1) .* (ylim[2]-ylim[1]) # not quite to edge
    t_units == :min && (t = t/60)
    plot!(p1,[t,t],[ylim[1],ylim[2]],lab=lab,c=:red,ls=:dash,legend=legend)
end # function plot_events

"""
    plot_events!(p1, flight::Symbol, df_event::DataFrame;
                 show_lab::Bool=true, ylim=ylims(p1),
                 t0=0, t_units::Symbol=:min, legend::Symbol=:outertopright)

Plot in-flight event(s) on existing plot.

**Arguments:**
- `p1`:       existing plot
- `flight`:   name of flight data
- `df_event`: lookup table (DataFrame) of in-flight events
- `show_lab`: (optional) if true, show in-flight event (legend) label(s)
- `ylim`:     (optional) 2-element y limits for in-flight event(s)
- `t0`:       (optional) time offset
- `t_units`:  (optional) time units used for plotting {`:sec`,`:min`}
- `legend`:   (optional) position of legend

**Returns:**
- `p1`: in-flight events are plotted on `p1`
"""
function plot_events!(p1, flight::Symbol, df_event::DataFrame;
                      show_lab::Bool=true, ylim=ylims(p1),
                      t0=0, t_units::Symbol=:min, legend::Symbol=:outertopright)
    xlim = xlims(p1)
    t_units == :min && (xlim = xlims(p1).*60)
    df   = df_event[(df_event[:,:flight]  .== flight)  .& 
                    (df_event[:,:t] .- t0 .>  xlim[1]) .& 
                    (df_event[:,:t] .- t0 .<  xlim[2]),:]
    for i = 1:size(df,1)
        lab = show_lab ? df[i,:event] : nothing
        plot_events!(p1,df[i,:t]-t0,lab;ylim=ylim,t_units=t_units,legend=legend)
    end
end # function plot_events

"""
    map_check(map_map::Map, lat, lon)

Check if latitude and longitude points are on given map.

**Arguments:**
- `map_map`: `Map` magnetic anomaly map struct
- `lat`:     latitude  [rad]
- `lon`:     longitude [rad]

**Returns:**
- `bool`: if true, all `lat` and `lon` points are on `map_map`
"""
function map_check(map_map::Map, lat, lon)
    itp_map = map_itp(map_map,:linear)
    N   = length(lat)
    val = trues(N)
    for i = 1:N
        minimum(map_map.xx) < lon[i] < maximum(map_map.xx) || (val[i] = false)
        minimum(map_map.yy) < lat[i] < maximum(map_map.yy) || (val[i] = false)
        val[i] == true && (itp_map(lon[i],lat[i]) != 0     || (val[i] = false))
    end
    return (all(val))
end # function map_check

"""
    map_check(map_map::Map, path::Path, ind=trues(path.N))

Check if latitude and longitude points are on given map.

**Arguments:**
- `map_map`: `Map` magnetic anomaly map struct
- `path`:    `Path` struct, i.e. `Traj` trajectory struct, `INS` inertial navigation system struct, or `FILTout` filter extracted output struct
- `ind`:     (optional) selected data indices

**Returns:**
- `bool`: if true, all `path[ind]` points are on `map_map`
"""
function map_check(map_map::Map, path::Path, ind=trues(path.N))
    map_check(map_map,path.lat[ind],path.lon[ind])
end # function map_check

"""
    map_map::Vector{Map}, path::Path, ind=trues(path.N))

Check if latitude and longitude points are on given map(s).

**Arguments:**
- `map_map`: `Map` magnetic anomaly map struct(s)
- `path`:    `Path` struct, i.e. `Traj` trajectory struct, `INS` inertial navigation system struct, or `FILTout` filter extracted output struct
- `ind`:     (optional) selected data indices

**Returns:**
- `bool(s)`: if true, all `path[ind]` points are on `map_map`
"""
function map_check(map_map::Vector{Map}, path::Path, ind=trues(path.N))
    [map_check(map_map[i],path,ind) for i = 1:length(map_map)]
end # function map_check
