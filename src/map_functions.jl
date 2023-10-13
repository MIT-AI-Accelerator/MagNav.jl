"""
    map_interpolate(map_map::AbstractArray{T},
                    map_xx::AbstractVector{T},
                    map_yy::AbstractVector{T},
                    type::Symbol = :cubic,
                    map_alt::AbstractVector = []) where T

Create map interpolation function, equivalent of griddedInterpolant in MATLAB.

**Arguments:**
- `map_map`: `ny` x `nx` (x `nz`) 2D or 3D gridded map data
- `map_xx`:  `nx` map x-direction (longitude) coordinates
- `map_yy`:  `ny` map y-direction (latitude)  coordinates
- `type`:    (optional) type of interpolation {:linear,:quad,:cubic}
- `map_alt`: (optional) map altitude levels

**Returns:**
- `itp_map`: map interpolation function
"""
function map_interpolate(map_map::AbstractArray{T},
                         map_xx::AbstractVector{T},
                         map_yy::AbstractVector{T},
                         type::Symbol = :cubic,
                         map_alt::AbstractVector = []) where T

    # uses Interpolations package rather than Dierckx or GridInterpolations,
    # as Interpolations was found to be fastest for MagNav use cases.

    if type == :linear
        spline_type = BSpline(Linear())
    elseif type == :quad
        spline_type = BSpline(Quadratic(Line(OnGrid())))
    elseif type == :cubic
        spline_type = BSpline(Cubic(Line(OnGrid())))
    else
        error("$type interpolation type not defined")
    end

    xx = LinRange(extrema(map_xx)...,length(map_xx))
    yy = LinRange(extrema(map_yy)...,length(map_yy))

    if map_alt == []
        itp_map = scale(interpolate(map_map',spline_type),xx,yy)
    else
        zz = LinRange(extrema(map_alt)...,length(map_alt))
        itp_map = scale(interpolate(permutedims(map_map,(2,1,3)),spline_type),xx,yy,zz)
    end

    return map_itp_function(itp_map)
end # function map_interpolate

"""
    map_itp_function(itp_map::ScaledInterpolation{T1}) where T1

Create map interpolation function from map ScaledInterpolation.

**Arguments:**
- `itp_map`: map ScaledInterpolation

**Returns:**
- `itp_map`: map interpolation function
"""
function map_itp_function(itp_map::ScaledInterpolation{T1}) where T1
    if length(size(itp_map)) == 2
        function itp_map_2D(lon::T1,lat::T1,alt::T1=lat) where T1
            itp_map(lon,lat)
        end
        return (itp_map_2D)
    elseif length(size(itp_map)) == 3
        function itp_map_3D(lon::T1,lat::T1,alt::T1) where T1
            itp_map(lon,lat,alt)
        end
        return (itp_map_3D)
    end
end # function map_itp_function

"""
    map_interpolate(mapS::Union{MapS,MapSd,MapS3D}, type::Symbol = :cubic;
                    return_vert_deriv::Bool = false)

Create map interpolation function, equivalent of griddedInterpolant in MATLAB.
Optionally return vertical derivative grid interpolation, which is calculated
using finite differences between the map and a slightly upward continued map.

**Arguments:**
- `mapS`:              `MapS`, `MapSd`, or `MapS3D` scalar magnetic anomaly map struct
- `type`:              (optional) type of interpolation {:linear,:quad,:cubic}
- `return_vert_deriv`: (optional) if true, also return `der_map`

**Returns:**
- `itp_map`: map interpolation function
- `der_map`: if `return_vert_deriv = true`, vertical derivative grid interpolation
"""
function map_interpolate(mapS::Union{MapS,MapSd,MapS3D}, type::Symbol = :cubic;
                         return_vert_deriv::Bool = false)

    if return_vert_deriv
        if typeof(mapS) <: Union{MapS,MapSd}
            map_map = upward_fft(mapS,mapS.alt+1).map - mapS.map
            return (map_itp(mapS.map,mapS.xx,mapS.yy,type),
                    map_itp( map_map,mapS.xx,mapS.yy,type))
        elseif typeof(mapS) <: Union{MapS3D}
            map_map = zero.(mapS.map)
            for i in eachindex(mapS.alt)
                mapS_ = MapS(mapS.map[:,:,i],mapS.xx,mapS.yy,mapS.alt[i])
                map_map[:,:,i] = upward_fft(mapS_,mapS_.alt+1).map - mapS.map[:,:,i]
            end
            return (map_itp(mapS.map,mapS.xx,mapS.yy,type,mapS.alt),
                    map_itp( map_map,mapS.xx,mapS.yy,type,mapS.alt))
        end
    else
        if typeof(mapS) <: Union{MapS,MapSd}
            return map_itp(mapS.map,mapS.xx,mapS.yy,type)
        elseif typeof(mapS) <: Union{MapS3D}
            return map_itp(mapS.map,mapS.xx,mapS.yy,type,mapS.alt)
        end
    end
end # function map_interpolate

"""
    map_interpolate(mapV::MapV, dim::Symbol = :X, type::Symbol = :cubic)

Create map interpolation function, equivalent of griddedInterpolant in MATLAB.

**Arguments:**
- `mapV`: `MapV` vector magnetic anomaly map struct
- `dim`:  map dimension to interpolate {`:X`,`:Y`,`:Z`}
- `type`: (optional) type of interpolation {:linear,:quad,:cubic}

**Returns:**
- `itp_map`: map interpolation function
"""
function map_interpolate(mapV::MapV, dim::Symbol = :X, type::Symbol = :cubic)

    if dim == :X
        map_itp(mapV.mapX,mapV.xx,mapV.yy,type)
    elseif dim == :Y
        map_itp(mapV.mapY,mapV.xx,mapV.yy,type)
    elseif dim == :Z
        map_itp(mapV.mapZ,mapV.xx,mapV.yy,type)
    else
        error("$dim dim not defined")
    end

end # function map_interpolate

map_itp = map_interpolate

"""
    (mapS3D::MapS3D)(alt::Real=mapS3D.alt[1])

Get scalar magnetic anomaly map at specific altitude.

**Arguments:**
- `mapS3D`: `MapS3D` 3D (multi-level) scalar magnetic anomaly map struct
- `alt`:    (optional) map altitude [m]

**Returns:**
- `mapS`: `MapS` scalar magnetic anomaly map struct at `alt`
"""
function (mapS3D::MapS3D)(alt::Real=mapS3D.alt[1])
    alt_lev = LinRange(extrema(mapS3D.alt)...,length(mapS3D.alt))
    map_3D  = mapS3D.map

    if alt < alt_lev[1] # desired map below data
        @info("extracting map from lowest map altitude level, $(alt_lev[1])")
        map_map = map_3D[:,:,1] # take lowest (closest) value
    elseif alt_lev[end] < alt # desired map above data
        @info("extracting map from highest map altitude level, $(alt_lev[end])")
        map_map = map_3D[:,:,end] # take highest (closest) value
    else
        map_map = zero.(map_3D[:,:,1])
        (ny,nx) = size(map_map)
        for i = 1:nx
            for j = 1:ny
                itp = interpolate(map_3D[j,i,:],BSpline(Linear()))
                map_map[j,i] = scale(itp,alt_lev)(alt)
            end
        end
    end

    return MapS(map_map, mapS3D.xx, mapS3D.yy, convert(eltype(map_map), alt))
end # function MapS3D

"""
    map_get_gxf(map_gxf::String)

Use ArchGDAL to read in map data from GXF file.

**Arguments:**
- `map_gxf`: path/name of map GXF file (`.gxf` extension optional)

**Returns:**
- `map_map`: `ny` x `nx` 2D gridded map data
- `map_xx`:  `nx` map x-direction (longitude) coordinates
- `map_yy`:  `ny` map y-direction (latitude)  coordinates
"""
function map_get_gxf(map_gxf::String)

    map_gxf = add_extension(map_gxf,".gxf")

    # configure dataset to be read as Float64
    ArchGDAL.setconfigoption("GXF_DATATYPE","Float64")

    # read GXF (raster-type) dataset from map_gxf
    ArchGDAL.read(map_gxf) do dataset

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
    map_params(map_map::Matrix,
               map_xx::Vector = collect(axes(map_map,2)),
               map_yy::Vector = collect(axes(map_map,1)))

Internal helper function to get basic map parameters.

**Arguments:**
- `map_map`: `ny` x `nx` 2D gridded map data
- `map_xx`:  (optional) `nx` map x-direction (longitude) coordinates
- `map_yy`:  (optional) `ny` map y-direction (latitude)  coordinates

**Returns:**
- `ind0`: map indices with zeros
- `ind1`: map indices without zeros
- `nx`:   x-direction map dimension
- `ny`:   y-direction map dimension
"""
function map_params(map_map::Matrix,
                    map_xx::Vector = collect(axes(map_map,2)),
                    map_yy::Vector = collect(axes(map_map,1)))

    map_map = deepcopy(map_map)
    replace!(map_map, NaN=>0) # just in case

    # map indices with (ind0) and without (ind1) zeros
    ind0 = map_map .== 0
    ind1 = map_map .!= 0

    # map size
    (ny,nx) = size(map_map)
    @assert nx == length(map_xx) "xx map dimensions are inconsistent"
    @assert ny == length(map_yy) "yy map dimensions are inconsistent"

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
- `nx`:   x-direction map dimension
- `ny`:   y-direction map dimension
"""
function map_params(map_map::Map)
    if typeof(map_map) <: MapV # vector map
        map_params(map_map.mapX,map_map.xx,map_map.yy)
    else # scalar map
        map_params(map_map.map[:,:,1],map_map.xx,map_map.yy)
    end
end # function map_params

"""
    map_lla_lim(map_xx::Vector, map_yy::Vector;
                xx_1::Int      = 1,
                xx_nx::Int     = length(map_xx),
                yy_1::Int      = 1,
                yy_ny::Int     = length(map_yy),
                zone_utm::Int  = 18,
                is_north::Bool = true)

Internal helper function to get lat/lon limits at 4 corners of `UTM` map.

**Arguments:**
- `map_xx`:   `nx` map x-direction (longitude) coordinates [m]
- `map_yy`:   `ny` map y-direction (latitude)  coordinates [m]
- `xx_1`:     (optional) first index of `map_xx` to consider
- `xx_nx`:    (optional) last  index of `map_xx` to consider
- `yy_1`:     (optional) first index of `map_yy` to consider
- `yy_ny`:    (optional) last  index of `map_yy` to consider
- `zone_utm`: (optional) UTM zone
- `is_north`: (optional) if true, map is in northern hemisphere

**Returns:**
- `lons`: longitudes for each of 4 corners of `UTM` map [deg]
- `lats`: latitudes  for each of 4 corners of `UTM` map [deg]
"""
function map_lla_lim(map_xx::Vector, map_yy::Vector;
                     xx_1::Int      = 1,
                     xx_nx::Int     = length(map_xx),
                     yy_1::Int      = 1,
                     yy_ny::Int     = length(map_yy),
                     zone_utm::Int  = 18,
                     is_north::Bool = true)

    # 4 corners of UTM map
    utm2lla = LLAfromUTM(zone_utm,is_north,WGS84)
    x       = map_xx[[xx_1,xx_1,xx_nx,xx_nx]]
    y       = map_yy[[yy_1,yy_ny,yy_1,yy_ny]]
    llas    = utm2lla.(UTM.(x,y))

    # sorted longitudes at 4 corners of UTM map
    # left/right edges are straight, so only corners needed
    lons = sort([lla.lon for lla in llas])

    # lower/upper parallels of UTM map
    x       = map_xx[xx_1:xx_nx]
    llas_1  = utm2lla.(UTM.(x,map_yy[yy_1 ]))
    llas_ny = utm2lla.(UTM.(x,map_yy[yy_ny]))

    # sorted latitude limits for lower/upper parallels of UTM map
    lats = sort([extrema([lla.lat for lla in llas_1 ])...,
                 extrema([lla.lat for lla in llas_ny])...])

    return (lons, lats)
end # function map_lla_lim

"""
    map_trim(map_map::Matrix,
             map_xx::Vector    = collect(axes(map_map,2)),
             map_yy::Vector    = collect(axes(map_map,1)),
             pad::Int          = 0,
             xx_lim::Tuple     = (-Inf,Inf),
             yy_lim::Tuple     = (-Inf,Inf),
             zone_utm::Int     = 18,
             is_north::Bool    = true,
             map_units::Symbol = :rad,
             silent::Bool      = true)

Trim map by removing large areas that are missing map data. Returns
indices for the original map that will produce the appropriate trimmed map.

**Arguments:**
- `map_map`:   `ny` x `nx` 2D gridded map data
- `map_xx`:    (optional) `nx` map x-direction (longitude) coordinates [rad] or [deg] or [m]
- `map_yy`:    (optional) `ny` map y-direction (latitude)  coordinates [rad] or [deg] or [m]
- `pad`:       (optional) minimum padding (grid cells) along map edges
- `xx_lim`:    (optional) x-direction map limits `(xx_min,xx_max)` [rad] or [deg] or [m]
- `yy_lim`:    (optional) y-direction map limits `(yy_min,yy_max)` [rad] or [deg] or [m]
- `zone_utm`:  (optional) UTM zone, only used if `map_units = :utm`
- `is_north`:  (optional) if true, map is in northern hemisphere, only used if `map_units = :utm`
- `map_units`: (optional) map xx/yy units {`:rad`,`:deg`,`:utm`}
- `silent`:    (optional) if true, no print outs

**Returns:**
- `ind_xx`: `nx` trimmed x-direction map indices
- `ind_yy`: `ny` trimmed y-direction map indices
"""
function map_trim(map_map::Matrix,
                  map_xx::Vector    = collect(axes(map_map,2)),
                  map_yy::Vector    = collect(axes(map_map,1));
                  pad::Int          = 0,
                  xx_lim::Tuple     = (-Inf,Inf),
                  yy_lim::Tuple     = (-Inf,Inf),
                  zone_utm::Int     = 18,
                  is_north::Bool    = true,
                  map_units::Symbol = :rad,
                  silent::Bool      = true)

    (ny,nx) = size(map_map)

    # xx limits of data-containing map
    xx_sum  = vec(sum(map_map,dims=1))
    xx_1    = findfirst(xx_sum .!= 0)
    xx_nx   = findlast(xx_sum  .!= 0)

    # yy limits of data-containing map
    yy_sum = vec(sum(map_map,dims=2))
    yy_1   = findfirst(yy_sum .!= 0)
    yy_ny  = findlast(yy_sum  .!= 0)

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

    if map_units == :utm

        # get xx/yy limits at 4 corners of data-containing UTM map for no data loss
        (lons,lats) = map_lla_lim(map_xx,map_yy;
                                  xx_1     = xx_1,
                                  xx_nx    = xx_nx,
                                  yy_1     = yy_1,
                                  yy_ny    = yy_ny,
                                  zone_utm = zone_utm,
                                  is_north = is_north)

        # use EXTERIOR 2 lons/lats as xx/yy limits
        lla2utm = UTMfromLLA(zone_utm,is_north,WGS84)
        utms    = lla2utm.(LLA.(lats[[1,1,end,end]],lons[[1,end,1,end]]))

        # xx/yy limits at 4 corners of UTM map for no data loss
        # due to Earth's curvature, xx/yy limits are further out
        xxs = sort([utm.x for utm in utms])
        yys = sort([utm.y for utm in utms])

    elseif map_units in [:rad,:deg]

        # directly use data-containing xx/yy limits at 4 corners
        xxs = sort(map_xx[[xx_1,xx_1,xx_nx,xx_nx]])
        yys = sort(map_yy[[yy_1,yy_1,yy_ny,yy_ny]])

    else
        error("[$map_units] map xx/yy units not defined")

    end

    # minimum padding (per edge) to prevent data loss during utm2lla
    pad_xx_1 = xxs[1] <= map_xx[1] ? xx_1-1 : xx_1 - findlast(map_xx .< xxs[1])
    pad_yy_1 = yys[1] <= map_yy[1] ? yy_1-1 : yy_1 - findlast(map_yy .< yys[1])

    if xxs[end] >= map_xx[end]
        pad_xx_nx = nx - xx_nx
    else
        pad_xx_nx = findfirst(map_xx .> xxs[end]) - xx_nx
    end

    if yys[end] >= map_yy[end]
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
             pad::Int          = 0,
             xx_lim::Tuple     = (-Inf,Inf),
             yy_lim::Tuple     = (-Inf,Inf),
             zone_utm::Int     = 18,
             is_north::Bool    = true,
             map_units::Symbol = :rad,
             silent::Bool      = true)

Trim map by removing large areas that are missing map data. Returns
trimmed magnetic anomaly map struct.

**Arguments:**
- `map_map`:   `Map` magnetic anomaly map struct
- `pad`:       (optional) minimum padding (grid cells) along map edges
- `xx_lim`:    (optional) x-direction map limits `(xx_min,xx_max)` [rad] or [deg] or [m]
- `yy_lim`:    (optional) y-direction map limits `(yy_min,yy_max)` [rad] or [deg] or [m]
- `zone_utm`:  (optional) UTM zone, only used if `map_units = :utm`
- `is_north`:  (optional) if true, map is in northern hemisphere, only used if `map_units = :utm`
- `map_units`: (optional) map xx/yy units {`:rad`,`:deg`,`:utm`}
- `silent`:    (optional) if true, no print outs

**Returns:**
- `map_map`: `Map` magnetic anomaly map struct, trimmed
"""
function map_trim(map_map::Map;
                  pad::Int          = 0,
                  xx_lim::Tuple     = (-Inf,Inf),
                  yy_lim::Tuple     = (-Inf,Inf),
                  zone_utm::Int     = 18,
                  is_north::Bool    = true,
                  map_units::Symbol = :rad,
                  silent::Bool      = true)

    if typeof(map_map) <: Union{MapS,MapSd,MapS3D} # scalar map
        typeof(map_map) <: MapS3D && @info("3D map provided, using map at lowest altitude")
        (ind_xx,ind_yy) = map_trim(map_map.map[:,:,1],map_map.xx,map_map.yy;
                                   pad=pad,xx_lim=xx_lim,yy_lim=yy_lim,
                                   zone_utm=zone_utm,is_north=is_north,
                                   map_units=map_units,silent=silent)
        if typeof(map_map) <: MapS
            return MapS(  map_map.map[ind_yy,ind_xx],map_map.xx[ind_xx],
                          map_map.yy[ind_yy],map_map.alt)
        elseif typeof(map_map) <: MapSd # drape map
            return MapSd( map_map.map[ind_yy,ind_xx],map_map.xx[ind_xx],
                          map_map.yy[ind_yy],map_map.alt[ind_yy,ind_xx])
        elseif typeof(map_map) <: MapS3D # 3D map
            return MapS3D(map_map.map[ind_yy,ind_xx,:],map_map.xx[ind_xx],
                          map_map.yy[ind_yy],map_map.alt)
        end
    elseif typeof(map_map) <: MapV # vector map
        (ind_xx,ind_yy) = map_trim(map_map.mapX,map_map.xx,map_map.yy;
                                   pad=pad,xx_lim=xx_lim,yy_lim=yy_lim,
                                   zone_utm=zone_utm,is_north=is_north,
                                   map_units=map_units,silent=silent)
        return MapV(map_map.mapX[ind_yy,ind_xx],map_map.mapY[ind_yy,ind_xx],
                    map_map.mapZ[ind_yy,ind_xx],map_map.xx[ind_xx],
                    map_map.yy[ind_yy],map_map.alt)
    end

end # function map_trim

"""
    map_trim(map_map::Map, path::Path;
             pad::Int          = 0,
             zone_utm::Int     = 18,
             is_north::Bool    = true,
             map_units::Symbol = :rad,
             silent::Bool      = true)

Trim map by removing large areas far away from `path`. Do not use prior to
upward continuation, as it will result in edge effect errors. Returns trimmed
magnetic anomaly map struct.

**Arguments:**
- `map_map`:   `Map` magnetic anomaly map struct
- `path`:      `Path` struct, i.e., `Traj` trajectory struct, `INS` inertial navigation system struct, or `FILTout` filter extracted output struct
- `pad`:       (optional) minimum padding (grid cells) along map edges
- `zone_utm`:  (optional) UTM zone, only used if `map_units = :utm`
- `is_north`:  (optional) if true, map is in northern hemisphere, only used if `map_units = :utm`
- `map_units`: (optional) map xx/yy units {`:rad`,`:deg`,`:utm`}
- `silent`:    (optional) if true, no print outs

**Returns:**
- `map_map`: `Map` magnetic anomaly map struct, trimmed
"""
function map_trim(map_map::Map, path::Path;
                  pad::Int          = 0,
                  zone_utm::Int     = 18,
                  is_north::Bool    = true,
                  map_units::Symbol = :rad,
                  silent::Bool      = true)
    map_trim(map_map;
             pad       = pad,
             xx_lim    = extrema(path.lon),
             yy_lim    = extrema(path.lat),
             zone_utm  = zone_utm,
             is_north  = is_north,
             map_units = map_units,
             silent    = silent)
end # function map_trim

"""
    map_correct_igrf!(map_map::Matrix, map_alt,
                      map_xx::Vector, map_yy::Vector;
                      sub_igrf_date::Real = get_years(2013,293), # 20-Oct-2013
                      add_igrf_date::Real = -1,
                      zone_utm::Int       = 18,
                      is_north::Bool      = true,
                      map_units::Symbol   = :rad)

Correct the International Geomagnetic Reference Field (IGRF), i.e., core field,
of a map by subtracting and/or adding the IGRF on specified date(s).

**Arguments:**
- `map_map`:       `ny` x `nx` 2D gridded map data
- `map_alt`:       `ny` x `nx` 2D gridded altitude map data, single altitude value may be provided [m]
- `map_xx`:        `nx` map x-direction (longitude) coordinates [rad] or [deg] or [m]
- `map_yy`:        `ny` map y-direction (latitude)  coordinates [rad] or [deg] or [m]
- `sub_igrf_date`: (optional) date of IGRF core field to subtract [yr], -1 to ignore
- `add_igrf_date`: (optional) date of IGRF core field to add [yr], -1 to ignore
- `zone_utm`:      (optional) UTM zone, only used if `map_units = :utm`
- `is_north`:      (optional) if true, map is in northern hemisphere, only used if `map_units = :utm`
- `map_units`:     (optional) map xx/yy units {`:rad`,`:deg`,`:utm`}

**Returns:**
- `nothing`: `map_map` is mutated with IGRF corrected map data
"""
function map_correct_igrf!(map_map::Matrix, map_alt,
                           map_xx::Vector, map_yy::Vector;
                           sub_igrf_date::Real = get_years(2013,293),
                           add_igrf_date::Real = -1,
                           zone_utm::Int       = 18,
                           is_north::Bool      = true,
                           map_units::Symbol   = :rad)

    (_,ind1,nx,ny) = map_params(map_map,map_xx,map_yy)

    all(map_alt .< 0)    && (map_alt = 300) # in case drape map altitude provided (uses alt = -1)
    length(map_alt) == 1 && (map_alt = map_alt*one.(map_map)) # in case single altitude provided
    map_alt = float(map_alt)

    sub_igrf = sub_igrf_date > 0 ? true : false
    add_igrf = add_igrf_date > 0 ? true : false

    if sub_igrf | add_igrf

        utm2lla = LLAfromUTM(zone_utm,is_north,WGS84)

        @info("starting igrf")

        for i = 1:nx # time consumer
            for j = 1:ny
                if ind1[j,i]

                    if map_units == :utm
                        lla = utm2lla(UTM(map_xx[i],map_yy[j],map_alt[j,i]))
                    elseif map_units == :rad
                        lla = LLA(rad2deg(map_yy[j]),rad2deg(map_xx[i]),map_alt[j,i])
                    elseif map_units == :deg
                        lla = LLA(map_yy[j],map_xx[i],map_alt[j,i])
                    else
                        error("[$map_units] map xx/yy units not defined")
                    end

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
    map_correct_igrf!(mapS::Union{MapS,MapSd,MapS3D};
                      sub_igrf_date::Real = get_years(2013,293), # 20-Oct-2013
                      add_igrf_date::Real = -1,
                      zone_utm::Int       = 18,
                      is_north::Bool      = true,
                      map_units::Symbol   = :rad)

Correct the International Geomagnetic Reference Field (IGRF), i.e., core field,
of a map by subtracting and/or adding the IGRF on specified date(s).

**Arguments:**
- `mapS`:          `MapS`, `MapSd`, or `MapS3D` scalar magnetic anomaly map struct
- `sub_igrf_date`: (optional) date of IGRF core field to subtract [yr], -1 to ignore
- `add_igrf_date`: (optional) date of IGRF core field to add [yr], -1 to ignore
- `zone_utm`:      (optional) UTM zone, only used if `map_units = :utm`
- `is_north`:      (optional) if true, map is in northern hemisphere, only used if `map_units = :utm`
- `map_units`:     (optional) map xx/yy units {`:rad`,`:deg`,`:utm`}

**Returns:**
- `nothing`: `map` field within `mapS` is mutated with IGRF corrected map data
"""
function map_correct_igrf!(mapS::Union{MapS,MapSd,MapS3D};
                           sub_igrf_date::Real = get_years(2013,293),
                           add_igrf_date::Real = -1,
                           zone_utm::Int       = 18,
                           is_north::Bool      = true,
                           map_units::Symbol   = :rad)
    if typeof(mapS) <: Union{MapS,MapSd}
        map_correct_igrf!(mapS.map,mapS.alt,mapS.xx,mapS.yy;
                          sub_igrf_date = sub_igrf_date,
                          add_igrf_date = add_igrf_date,
                          zone_utm      = zone_utm,
                          is_north      = is_north,
                          map_units     = map_units)
    elseif typeof(mapS) <: MapS3D
        for i in eachindex(mapS.alt)
            mapS.map[:,:,i] = map_correct_igrf(mapS.map[:,:,i],mapS.alt[i],
                                               mapS.xx,mapS.yy;
                                               sub_igrf_date = sub_igrf_date,
                                               add_igrf_date = add_igrf_date,
                                               zone_utm      = zone_utm,
                                               is_north      = is_north,
                                               map_units     = map_units)
        end
    end
end # function map_correct_igrf!

"""
    map_correct_igrf(map_map::Matrix, map_alt,
                     map_xx::Vector, map_yy::Vector;
                     sub_igrf_date::Real = get_years(2013,293), # 20-Oct-2013
                     add_igrf_date::Real = -1,
                     zone_utm::Int       = 18,
                     is_north::Bool      = true,
                     map_units::Symbol   = :rad)

Correct the International Geomagnetic Reference Field (IGRF), i.e., core field,
of a map by subtracting and/or adding the IGRF on specified date(s).

**Arguments:**
- `map_map`:       `ny` x `nx` 2D gridded map data
- `map_alt`:       `ny` x `nx` 2D gridded altitude map data [m]
- `map_xx`:        `nx` map x-direction (longitude) coordinates [rad] or [deg] or [m]
- `map_yy`:        `ny` map y-direction (latitude)  coordinates [rad] or [deg] or [m]
- `sub_igrf_date`: (optional) date of IGRF core field to subtract [yr], -1 to ignore
- `add_igrf_date`: (optional) date of IGRF core field to add [yr], -1 to ignore
- `zone_utm`:      (optional) UTM zone, only used if `map_units = :utm`
- `is_north`:      (optional) if true, map is in northern hemisphere, only used if `map_units = :utm`
- `map_units`:     (optional) map xx/yy units {`:rad`,`:deg`,`:utm`}

**Returns:**
- `map_map`: `ny` x `nx` 2D gridded map data, IGRF corrected
"""
function map_correct_igrf(map_map::Matrix, map_alt,
                          map_xx::Vector, map_yy::Vector;
                          sub_igrf_date::Real = get_years(2013,293),
                          add_igrf_date::Real = -1,
                          zone_utm::Int       = 18,
                          is_north::Bool      = true,
                          map_units::Symbol   = :rad)
    map_map = deepcopy(map_map)
    map_correct_igrf!(map_map,map_alt,map_xx,map_yy;
                      sub_igrf_date = sub_igrf_date,
                      add_igrf_date = add_igrf_date,
                      zone_utm      = zone_utm,
                      is_north      = is_north,
                      map_units     = map_units)
    return (map_map)
end # function map_correct_igrf

"""
    map_correct_igrf(mapS::Union{MapS,MapSd,MapS3D};
                     sub_igrf_date::Real = get_years(2013,293), # 20-Oct-2013
                     add_igrf_date::Real = -1,
                     zone_utm::Int       = 18,
                     is_north::Bool      = true,
                     map_units::Symbol   = :rad)

Correct the International Geomagnetic Reference Field (IGRF), i.e., core field,
of a map by subtracting and/or adding the IGRF on specified date(s).

**Arguments:**
- `mapS`:          `MapS`, `MapSd`, or `MapS3D` scalar magnetic anomaly map struct
- `sub_igrf_date`: (optional) date of IGRF core field to subtract [yr], -1 to ignore
- `add_igrf_date`: (optional) date of IGRF core field to add [yr], -1 to ignore
- `zone_utm`:      (optional) UTM zone, only used if `map_units = :utm`
- `is_north`:      (optional) if true, map is in northern hemisphere, only used if `map_units = :utm`
- `map_units`:     (optional) map xx/yy units {`:rad`,`:deg`,`:utm`}

**Returns:**
- `mapS`: `MapS`, `MapSd`, or `MapS3D` scalar magnetic anomaly map struct, IGRF corrected
"""
function map_correct_igrf(mapS::Union{MapS,MapSd,MapS3D};
                          sub_igrf_date::Real = get_years(2013,293),
                          add_igrf_date::Real = -1,
                          zone_utm::Int       = 18,
                          is_north::Bool      = true,
                          map_units::Symbol   = :rad)
    mapS = deepcopy(mapS)
    map_correct_igrf!(mapS;
                      sub_igrf_date = sub_igrf_date,
                      add_igrf_date = add_igrf_date,
                      zone_utm      = zone_utm,
                      is_north      = is_north,
                      map_units     = map_units)
    return (mapS)
end # function map_correct_igrf

"""
    map_fill!(map_map::Matrix, map_xx::Vector, map_yy::Vector; k::Int=3)

Fill areas that are missing map data.

**Arguments:**
- `map_map`: `ny` x `nx` 2D gridded map data
- `map_xx`:  `nx` map x-direction (longitude) coordinates
- `map_yy`:  `ny` map y-direction (latitude)  coordinates
- `k`:       (optional) number of nearest neighbors for knn

**Returns:**
- `nothing`: `map_map` is mutated with filled map data
"""
function map_fill!(map_map::Matrix, map_xx::Vector, map_yy::Vector; k::Int=3)

    (ind0,ind1,nx,ny) = map_params(map_map,map_xx,map_yy)

    data = vcat(vec(repeat(map_xx',ny,1)[ind1])',
                vec(repeat(map_yy ,1,nx)[ind1])') # xx & yy at ind1 [2 x N1]
    pts  = vcat(vec(repeat(map_xx',ny,1)[ind0])',
                vec(repeat(map_yy ,1,nx)[ind0])') # xx & yy at ind0 [2 x N0]
    vals = vec(map_map[ind1]) # map data at ind1 [N1]
    tree = KDTree(data)
    inds = knn(tree,pts,k,true)[1]

    j = 0
    for i in eachindex(map_map)
        if ind0[i]
            j += 1
            @inbounds map_map[i] = mean(vals[inds[j]])
        end
    end

end # function map_fill!

"""
    map_fill!(mapS::Union{MapS,MapSd,MapS3D}; k::Int=3)

Fill areas that are missing map data.

**Arguments:**
- `mapS`: `MapS`, `MapSd`, or `MapS3D` scalar magnetic anomaly map struct
- `k`:    (optional) number of nearest neighbors for knn

**Returns:**
- `nothing`: `map` field within `mapS` is mutated with filled map data
"""
function map_fill!(mapS::Union{MapS,MapSd,MapS3D}; k::Int=3)
    if typeof(mapS) <: MapS
        map_fill!(mapS.map,mapS.xx,mapS.yy;k=k)
    elseif typeof(mapS) <: MapSd
        map_fill!(mapS.map,mapS.xx,mapS.yy;k=k)
        map_fill!(mapS.alt,mapS.xx,mapS.yy;k=k)
    elseif typeof(mapS) <: MapS3D
        for i in axes(mapS.map,3)
            mapS.map[:,:,i] = map_fill(mapS.map[:,:,i],mapS.xx,mapS.yy;k=k)
        end
    end
end # function map_fill!

"""
    map_fill(map_map::Matrix, map_xx::Vector, map_yy::Vector; k::Int=3)

Fill areas that are missing map data.

**Arguments:**
- `map_map`: `ny` x `nx` 2D gridded map data
- `map_xx`:  `nx` map x-direction (longitude) coordinates
- `map_yy`:  `ny` map y-direction (latitude)  coordinates
- `k`:       (optional) number of nearest neighbors for knn

**Returns:**
- `map_map`: `ny` x `nx` 2D gridded map data, filled
"""
function map_fill(map_map::Matrix, map_xx::Vector, map_yy::Vector; k::Int=3)
    map_map = deepcopy(map_map)
    map_fill!(map_map,map_xx,map_yy;k=k)
    return (map_map)
end # function map_fill

"""
    map_fill(mapS::Union{MapS,MapSd,MapS3D}; k::Int=3)

Fill areas that are missing map data.

**Arguments:**
- `mapS`: `MapS`, `MapSd`, or `MapS3D` scalar magnetic anomaly map struct
- `k`:    (optional) number of nearest neighbors for knn

**Returns:**
- `mapS`: `MapS`, `MapSd`, or `MapS3D` scalar magnetic anomaly map struct, filled
"""
function map_fill(mapS::Union{MapS,MapSd,MapS3D}; k::Int=3)
    mapS = deepcopy(mapS)
    map_fill!(mapS;k=k)
    return (mapS)
end # function map_fill

"""
    map_chessboard!(map_map::Matrix, map_alt::Matrix, map_xx::Vector,
                    map_yy::Vector, alt::Real;
                    down_cont::Bool = true,
                    dz              = 5,
                    down_max        = 150,
                    α               = 200)

The `chessboard method`, which upward (and possibly downward) continues a map
to multiple altitudes to create a 3D map, then vertically interpolates at each
horizontal grid point.

Reference: Cordell, Phillips, & Godson, U.S. Geological Survey Potential-Field
Software Version 2.0, 1992.

**Arguments:**
- `map_map`:   `ny` x `nx` 2D gridded target (e.g., magnetic) map data on [m] grid
- `map_alt`:   `ny` x `nx` 2D gridded altitude map data [m] on [m] grid
- `map_xx`:    `nx` map x-direction (longitude) coordinates [m]
- `map_yy`:    `ny` map y-direction (latitude)  coordinates [m]
- `alt`:       final map altitude after upward continuation [m]
- `down_cont`: (optional) if true, downward continue if needed, only used if `up_cont = true`
- `dz`:        (optional) upward continuation step size [m]
- `down_max`:  (optional) maximum downward continuation distance [m]
- `α`:         (optional) regularization parameter for downward continuation

**Returns:**
- `nothing`: `map_map` is mutated with upward continued map data
"""
function map_chessboard!(map_map::Matrix, map_alt::Matrix, map_xx::Vector,
                         map_yy::Vector, alt::Real;
                         down_cont::Bool = true,
                         dz              = 5,
                         down_max        = 150,
                         α               = 200)

    (ind0 ,ind1 ,nx ,ny ) = map_params(map_map,map_xx,map_yy)
    (ind0_,ind1_,nx_,ny_) = map_params(map_alt,map_xx,map_yy)

    @assert (nx,ny) == (nx_,ny_) "map dimensions are inconsistent"
    @assert sum(ind0 )/sum(ind0 +ind1 ) < 0.01 "target   map must be filled to use chessboard method"
    @assert sum(ind0_)/sum(ind0_+ind1_) < 0.01 "altitude map must be filled to use chessboard method"

    # map step sizes (spacings)
    dx = get_step(map_xx)
    dy = get_step(map_yy)

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
    map_3D = zeros(ny,nx,nz)

    @info("starting upward and/or downward continuation with $nz levels")

    for k = 1:nz # time consumer
        if k == k0
            map_3D[:,:,k] = deepcopy(map_map)
        else
            @inbounds map_3D[:,:,k] = upward_fft(map_map,dx,dy,alt_lev[k];
                                                 expand=true,α=α)
        end
    end

    @info("starting chessboard interpolation")

    # interpolate vertical direction at each grid point
    for i = 1:nx
        for j = 1:ny
            if alt < alt_lev[1] + map_alt[j,i] # desired map below data
                map_map[j,i] = map_3D[j,i,1] # take lowest (closest) value
            elseif alt_lev[end] + map_alt[j,i] < alt # desired map above data
                map_map[j,i] = map_3D[j,i,end] # take highest (closest) value
            elseif ind1[j,i] # altitude data is available
                itp = interpolate(map_3D[j,i,:],BSpline(Linear()))
                map_map[j,i] = scale(itp,map_alt[j,i].+alt_lev)(alt)
            end
        end
    end

end # function map_chessboard!

"""
    map_chessboard(mapSd::MapSd, alt::Real;
                   down_cont::Bool = true,
                   dz              = 5,
                   down_max        = 150,
                   α               = 200)

The `chessboard method`, which upward (and possibly downward) continues a map
to multiple altitudes to create a 3D map, then vertically interpolates at each
horizontal grid point.

Reference: Cordell, Phillips, & Godson, U.S. Geological Survey Potential-Field
Software Version 2.0, 1992.

**Arguments:**
- `mapSd`:     `MapSd` scalar magnetic anomaly map struct
- `alt`:       final map altitude after upward continuation [m]
- `down_cont`: (optional) if true, downward continue if needed
- `dz`:        (optional) upward continuation step size [m]
- `down_max`:  (optional) maximum downward continuation distance [m]
- `α`:         (optional) regularization parameter for downward continuation

**Returns:**
- `mapS`: `MapS` scalar magnetic anomaly map struct
"""
function map_chessboard(mapSd::MapSd, alt::Real;
                        down_cont::Bool = true,
                        dz              = 5,
                        down_max        = 150,
                        α               = 200)
    map_xx = zero(mapSd.xx)
    map_yy = zero(mapSd.yy)
    for i in eachindex(map_xx)[2:end]
        map_xx[i] = map_xx[i-1] + dlon2de(mapSd.xx[i] - mapSd.xx[i-1],
                                          mean(mapSd.yy[i-1:i]))
    end
    for i in eachindex(map_yy)[2:end]
        map_yy[i] = map_yy[i-1] + dlat2dn(mapSd.yy[i] - mapSd.yy[i-1],
                                          mean(mapSd.yy[i-1:i]))
    end
    map_chessboard!(mapSd.map,mapSd.alt,map_xx,map_yy,alt;
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
                 map_h5::String = "map_data.h5")

Convert map grid from `UTM` to `LLA`.

**Arguments:**
- `map_map`:  `ny` x `nx` 2D gridded map data on `UTM` grid
- `map_xx`:   `nx` map x-direction (longitude) coordinates [m]
- `map_yy`:   `ny` map y-direction (latitude)  coordinates [m]
- `alt`:      map altitude(s) or altitude map [m]
- `zone_utm`: (optional) UTM zone
- `is_north`: (optional) if true, map is in northern hemisphere
- `save_h5`:  (optional) if true, save map data to `map_h5`
- `map_h5`:   (optional) path/name of map data HDF5 file to save (`.h5` extension optional)

**Returns:**
- `nothing`: `map_map`, `map_xx`, & `map_yy` (& `alt`) are mutated with `LLA` gridded map data
"""
function map_utm2lla!(map_map::Matrix, map_xx::Vector, map_yy::Vector, alt;
                      zone_utm::Int  = 18,
                      is_north::Bool = true,
                      save_h5::Bool  = false,
                      map_h5::String = "map_data.h5")

    ind1 = convert.(eltype(map_map),map_params(map_map,map_xx,map_yy)[2])
    (ny,nx) = size(map_map)

    map_drp = (ny,nx) == size(alt) ? true : false

    # interpolation for original (UTM) map
    itp_ind1 = map_itp(ind1   ,map_xx,map_yy,:linear)
    itp_map  = map_itp(map_map,map_xx,map_yy,:linear)
    map_drp && (itp_alt = map_itp(alt,map_xx,map_yy,:linear))

    # get xx/yy limits at 4 corners of data-containing UTM map for no data loss
    (lons,lats) = map_lla_lim(map_xx,map_yy;
                              zone_utm = zone_utm,
                              is_north = is_north)

    # use interior 2 lons/lats as xx/yy limits for new (LLA) map (stay in range)
    δ = 1e-10 # ad hoc to solve rounding related error
    map_xx .= [LinRange(lons[2]+δ,lons[3]-δ,nx);]
    map_yy .= [LinRange(lats[2]+δ,lats[3]-δ,ny);]

    # interpolate original (UTM) map with grid for new (LLA) map
    lla2utm = UTMfromLLA(zone_utm,is_north,WGS84)
    for i = 1:nx
        for j = 1:ny
            utm = lla2utm(LLA(map_yy[j],map_xx[i]))
            if itp_ind1(utm.x,utm.y) ≈ 1
                @inbounds map_map[j,i] = itp_map(utm.x,utm.y)
                map_drp && (@inbounds alt[j,i] = itp_alt(utm.x,utm.y))
            else
                @inbounds map_map[j,i] = 0
                map_drp && (@inbounds alt[j,i] = 0)
            end
        end
    end

    # convert to [rad]
    map_xx .= deg2rad.(map_xx)
    map_yy .= deg2rad.(map_yy)

    save_h5 && save_map(map_map,map_xx,map_yy,alt,map_h5;
                        map_units=:rad,file_units=:deg)

end # function map_utm2lla!

"""
    map_utm2lla!(mapS::Union{MapS,MapSd,MapS3D};
                 zone_utm::Int  = 18,
                 is_north::Bool = true,
                 save_h5::Bool  = false
                 map_h5::String = "map_data.h5")

Convert map grid from `UTM` to `LLA`.

**Arguments:**
- `mapS`:     `MapS`, `MapSd`, or `MapS3D` scalar magnetic anomaly map struct on `UTM` grid
- `zone_utm`: (optional) UTM zone
- `is_north`: (optional) if true, map is in northern hemisphere
- `save_h5`:  (optional) if true, save `mapS` to `map_h5`
- `map_h5`:   (optional) path/name of map data HDF5 file to save (`.h5` extension optional)

**Returns:**
- `nothing`: `map`, `xx`, & `yy` (& `alt`) fields within `mapS` are mutated with `LLA` gridded map data
"""
function map_utm2lla!(mapS::Union{MapS,MapSd,MapS3D};
                      zone_utm::Int  = 18,
                      is_north::Bool = true,
                      save_h5::Bool  = false,
                      map_h5::String = "map_data.h5")
    if typeof(mapS) <: Union{MapS,MapSd}
        map_utm2lla!(mapS.map,mapS.xx,mapS.yy,mapS.alt;
                     zone_utm = zone_utm,
                     is_north = is_north,
                     save_h5  = save_h5,
                     map_h5   = map_h5)
    elseif typeof(mapS) <: MapS3D
        map_xx_ = deepcopy(mapS.xx)
        map_yy_ = deepcopy(mapS.yy)
        for i in eachindex(mapS.alt)
            (map_map,map_xx,map_yy) = map_utm2lla(mapS.map[:,:,i],
                                                  map_xx_,map_yy_,mapS.alt[i];
                                                  zone_utm = zone_utm,
                                                  is_north = is_north,
                                                  save_h5  = false)
            mapS.map[:,:,i] = map_map
            mapS.xx .= map_xx
            mapS.yy .= map_yy
        end
        save_h5 && save_map(mapS,map_h5;map_units=:rad,file_units=:deg)
    end
end # function map_utm2lla!

"""
    map_utm2lla(map_map::Matrix, map_xx::Vector, map_yy::Vector, alt;
                zone_utm::Int  = 18,
                is_north::Bool = true,
                save_h5::Bool  = false,
                map_h5::String = "map_data.h5")

Convert map grid from `UTM` to `LLA`.

**Arguments:**
- `map_map`:  `ny` x `nx` 2D gridded map data on `UTM` grid
- `map_xx`:   `nx` map x-direction (longitude) coordinates [m]
- `map_yy`:   `ny` map y-direction (latitude)  coordinates [m]
- `alt`:      map altitude(s) or altitude map [m]
- `zone_utm`: (optional) UTM zone
- `is_north`: (optional) if true, map is in northern hemisphere
- `save_h5`:  (optional) if true, save map data to `map_h5`
- `map_h5`:   (optional) path/name of map data HDF5 file to save (`.h5` extension optional)

**Returns:**
- `map_map`: `ny` x `nx` 2D gridded map data on `LLA` grid
- `map_xx`:  `nx` map x-direction (longitude) coordinates [rad]
- `map_yy`:  `ny` map y-direction (latitude)  coordinates [rad]
"""
function map_utm2lla(map_map::Matrix, map_xx::Vector, map_yy::Vector, alt;
                     zone_utm::Int  = 18,
                     is_north::Bool = true,
                     save_h5::Bool  = false,
                     map_h5::String = "map_data.h5")
    map_map = deepcopy(map_map)
    map_xx  = deepcopy(map_xx)
    map_yy  = deepcopy(map_yy)
    alt     = deepcopy(alt)
    map_utm2lla!(map_map,map_xx,map_yy,alt;
                 zone_utm = zone_utm,
                 is_north = is_north,
                 save_h5  = save_h5,
                 map_h5   = map_h5)
    return (map_map, map_xx, map_yy)
end # function map_utm2lla

"""
    map_utm2lla(mapS::Union{MapS,MapSd,MapS3D};
                zone_utm::Int  = 18,
                is_north::Bool = true,
                save_h5::Bool  = false
                map_h5::String = "map_data.h5")

Convert map grid from `UTM` to `LLA`.

**Arguments:**
- `mapS`:     `MapS`, `MapSd`, or `MapS3D` scalar magnetic anomaly map struct on `UTM` grid
- `zone_utm`: (optional) UTM zone
- `is_north`: (optional) if true, map is in northern hemisphere
- `save_h5`:  (optional) if true, save `mapS` to `map_h5`
- `map_h5`:   (optional) path/name of map data HDF5 file to save (`.h5` extension optional)

**Returns:**
- `mapS`: `MapS`, `MapSd`, or `MapS3D` scalar magnetic anomaly map struct on `LLA` grid
"""
function map_utm2lla(mapS::Union{MapS,MapSd,MapS3D};
                     zone_utm::Int  = 18,
                     is_north::Bool = true,
                     save_h5::Bool  = false,
                     map_h5::String = "map_data.h5")
    mapS = deepcopy(mapS)
    map_utm2lla!(mapS;
                 zone_utm = zone_utm,
                 is_north = is_north,
                 save_h5  = save_h5,
                 map_h5   = map_h5)
    return (mapS)
end # function map_utm2lla

"""
    map_gxf2h5(map_gxf::String, alt_gxf::String, alt::Real;
               pad::Int            = 0,
               sub_igrf_date::Real = get_years(2013,293),
               add_igrf_date::Real = -1,
               zone_utm::Int       = 18,
               is_north::Bool      = true,
               fill_map::Bool      = true,
               up_cont::Bool       = true,
               down_cont::Bool     = true,
               get_lla::Bool       = true,
               dz::Real            = 5,
               down_max::Real      = 150,
               α::Real             = 200,
               save_h5::Bool       = false,
               map_h5::String      = "map_data.h5")

Convert map data file (with assumed `UTM` grid) from GXF to HDF5.
The order of operations is:
- original map from `map_gxf` =>
- trim away large areas that are missing map data =>
- subtract and/or add IGRF to map data =>
- fill remaining areas that are missing map data =>
- upward/downward continue to `alt` =>
- convert map grid from `UTM` to `LLA`
This can be memory intensive, largely depending on the map size and `dz`. If
`up_cont = true`, a `MapS` struct is returned. If `up_cont = false`, a `MapSd`
struct is returned, which has an included altitude map.

**Arguments:**
- `map_gxf`:       path/name of target (e.g., magnetic) map GXF file (`.gxf` extension optional)
- `alt_gxf`:       path/name of altitude map GXF file (`.gxf` extension optional)
- `alt`:           final map altitude after upward continuation [m], -1 for drape map
- `pad`:           (optional) minimum padding (grid cells) along map edges
- `sub_igrf_date`: (optional) date of IGRF core field to subtract [yr], -1 to ignore
- `add_igrf_date`: (optional) date of IGRF core field to add [yr], -1 to ignore
- `zone_utm`:      (optional) UTM zone
- `is_north`:      (optional) if true, map is in northern hemisphere
- `fill_map`:      (optional) if true, fill areas that are missing map data
- `up_cont`:       (optional) if true, upward/downward continue to `alt`
- `down_cont`:     (optional) if true, downward continue if needed, only used if `up_cont = true`
- `get_lla`:       (optional) if true, convert map grid from `UTM` to `LLA`
- `dz`:            (optional) upward continuation step size [m]
- `down_max`:      (optional) maximum downward continuation distance [m]
- `α`:             (optional) regularization parameter for downward continuation
- `save_h5`:       (optional) if true, save `mapS` to `map_h5`
- `map_h5`:        (optional) path/name of map data HDF5 file to save (`.h5` extension optional)

**Returns:**
- `mapS`: `MapS` or `MapSd` scalar magnetic anomaly map struct
"""
function map_gxf2h5(map_gxf::String, alt_gxf::String, alt::Real;
                    pad::Int            = 0,
                    sub_igrf_date::Real = get_years(2013,293),
                    add_igrf_date::Real = -1,
                    zone_utm::Int       = 18,
                    is_north::Bool      = true,
                    fill_map::Bool      = true,
                    up_cont::Bool       = true,
                    down_cont::Bool     = true,
                    get_lla::Bool       = true,
                    dz::Real            = 5,
                    down_max::Real      = 150,
                    α::Real             = 200,
                    save_h5::Bool       = false,
                    map_h5::String      = "map_data.h5")

    @info("starting GXF read")

    # get raw map data
    (map_map,map_xx ,map_yy ) = map_get_gxf(map_gxf)
    (map_alt,map_xx_,map_yy_) = map_get_gxf(alt_gxf)

    # make sure grids match
    @assert (map_xx ≈ map_xx_) & (map_yy ≈ map_yy_) "grids do not match"

    @info("starting trim")

    # trim away large areas that are missing map data
    (ind_xx,ind_yy) = map_trim(map_map,map_xx,map_yy;
                               pad       = pad,
                               zone_utm  = zone_utm,
                               is_north  = is_north,
                               map_units = :utm,
                               silent    = false)
    map_xx  = map_xx[ind_xx]
    map_yy  = map_yy[ind_yy]
    map_map = map_map[ind_yy,ind_xx]
    map_alt = map_alt[ind_yy,ind_xx]

    # subtract and/or add IGRF to map data
    map_correct_igrf!(map_map,map_alt,map_xx,map_yy;
                      sub_igrf_date = sub_igrf_date,
                      add_igrf_date = add_igrf_date,
                      zone_utm      = zone_utm,
                      is_north      = is_north,
                      map_units     = :utm)

    if fill_map # fill remaining areas that are missing map data
        @info("starting fill")
        map_fill!(map_map,map_xx,map_yy)
        map_fill!(map_alt,map_xx,map_yy)
    end

    if up_cont # upward/downward continue to alt
        if alt < 0
            @info("not upward continuing to $alt < 0")
        else
            map_chessboard!(map_map,map_alt,map_xx,map_yy,alt;
                            down_cont = down_cont,
                            dz        = dz,
                            down_max  = down_max,
                            α         = α)
        end
    end

    alt_ = up_cont ? alt : map_alt

    if get_lla # convert map grid from UTM to LLA
        @info("starting utm2lla")
        map_utm2lla!(map_map,map_xx,map_yy,alt_;
                     zone_utm = zone_utm,
                     is_north = is_north,
                     save_h5  = save_h5,
                     map_h5   = map_h5)
    elseif save_h5
        save_map(map_map,map_xx,map_yy,alt_,map_h5;
                 map_units=:utm,file_units=:utm)
    end

    if up_cont
        return MapS( map_map, map_xx, map_yy, convert(eltype(map_map), alt))
    else
        return MapSd(map_map, map_xx, map_yy, map_alt)
    end
end # function map_gxf2h5

"""
    map_gxf2h5(map_gxf::String, alt::Real;
               fill_map::Bool = true,
               get_lla::Bool  = true,
               zone_utm::Int  = 18,
               is_north::Bool = true,
               save_h5::Bool  = false,
               map_h5::String = "map_data.h5")

Convert map data file (with assumed `UTM` grid) from GXF to HDF5.
The order of operations is:
- original map from `map_gxf` =>
- trim away large areas that are missing map data =>
- fill remaining areas that are missing map data =>
- convert map grid from `UTM` to `LLA`
Specifically meant for SMALL and LEVEL maps ONLY.

**Arguments:**
- `map_gxf`:  path/name of target (e.g., magnetic) map GXF file (`.gxf` extension optional)
- `alt`:      map altitude [m]
- `fill_map`: (optional) if true, fill areas that are missing map data
- `get_lla`:  (optional) if true, convert map grid from `UTM` to `LLA`
- `zone_utm`: (optional) UTM zone
- `is_north`: (optional) if true, map is in northern hemisphere
- `save_h5`:  (optional) if true, save `mapS` to `map_h5`
- `map_h5`:   (optional) path/name of map data HDF5 file to save (`.h5` extension optional)

**Returns:**
- `mapS`: `MapS` scalar magnetic anomaly map struct
"""
function map_gxf2h5(map_gxf::String, alt::Real;
                    fill_map::Bool = true,
                    get_lla::Bool  = true,
                    zone_utm::Int  = 18,
                    is_north::Bool = true,
                    save_h5::Bool  = false,
                    map_h5::String = "map_data.h5")

    (map_map,map_xx,map_yy) = map_get_gxf(map_gxf) # get raw map data

    # trim away large areas that are missing map data
    (ind_xx,ind_yy) = map_trim(map_map,map_xx,map_yy;
                               pad       = 0,
                               zone_utm  = zone_utm,
                               is_north  = is_north,
                               map_units = :utm,
                               silent    = true)
    map_xx  = map_xx[ind_xx]
    map_yy  = map_yy[ind_yy]
    map_map = map_map[ind_yy,ind_xx]

    # fill remaining areas that are missing map data
    fill_map && map_fill!(map_map,map_xx,map_yy)

    if get_lla # convert map grid from UTM to LLA
        map_utm2lla!(map_map,map_xx,map_yy,alt;
                     zone_utm = zone_utm,
                     is_north = is_north,
                     save_h5  = save_h5,
                     map_h5   = map_h5)
    elseif save_h5
        save_map(map_map,map_xx,map_yy,alt,map_h5;
                 map_units=:utm,file_units=:utm)
    end

    return MapS(map_map, map_xx, map_yy, convert(eltype(map_map), alt))
end # function map_gxf2h5

"""
    plot_map!(p1, map_map::Matrix,
              map_xx::Vector     = [],
              map_yy::Vector     = [];
              clims::Tuple       = (-500,500),
              dpi::Int           = 200,
              margin::Int        = 2,
              Nmax::Int          = 6*dpi,
              legend::Bool       = true,
              axis::Bool         = true,
              map_color::Symbol  = :usgs,
              bg_color::Symbol   = :white,
              map_units::Symbol  = :rad,
              plot_units::Symbol = :deg,
              b_e                = gr())

Plot map on an existing plot.

**Arguments:**
- `p1`:         existing plot
- `map_map`:    `ny` x `nx` 2D gridded map data
- `map_xx`:     `nx` map x-direction (longitude) coordinates [rad] or [deg]
- `map_yy`:     `ny` map y-direction (latitude)  coordinates [rad] or [deg]
- `clims`:      (optional) color scale limits
- `dpi`:        (optional) dots per inch (image resolution)
- `margin`:     (optional) margin around plot [mm]
- `Nmax`:       (optional) maximum number of data points plotted
- `legend`:     (optional) if true, show legend
- `axis`:       (optional) if true, show axes
- `map_color`:  (optional) filled contour color scheme {`:usgs`,`:gray`,`:gray1`,`:gray2`,`:plasma`,`:magma`}
- `bg_color`:   (optional) background color
- `map_units`:  (optional) map  xx/yy units {`:rad`,`:deg`}
- `plot_units`: (optional) plot xx/yy units {`:rad`,`:deg`,`:m`}
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
                   Nmax::Int          = 6*dpi,
                   legend::Bool       = true,
                   axis::Bool         = true,
                   map_color::Symbol  = :usgs,
                   bg_color::Symbol   = :white,
                   map_units::Symbol  = :rad,
                   plot_units::Symbol = :deg,
                   b_e                = gr())

    (ny,nx) = size(map_map)
    xx_mid  = ceil(Int,nx/2)
    yy_mid  = ceil(Int,ny/2)

    # avoid changing map struct data
    map_map = float.(map_map)
    map_xx  = length(map_xx) < nx ? float.([1:nx;]) : float.(map_xx)
    map_yy  = length(map_yy) < ny ? float.([1:ny;]) : float.(map_yy)

    if map_units == :rad
        if plot_units == :deg
            map_xx .= rad2deg.(map_xx)
            map_yy .= rad2deg.(map_yy)
        elseif plot_units == :m # longitude inaccuracy scales with map size
            lon_mid = map_xx[xx_mid]
            lat_mid = map_yy[yy_mid]
            map_xx .= dlon2de.(map_xx .- lon_mid, lat_mid)
            map_yy .= dlat2dn.(map_yy .- lat_mid, map_yy)
        end
    elseif map_units == :deg
        if plot_units == :rad
            map_xx .= deg2rad.(map_xx)
            map_yy .= deg2rad.(map_yy)
        elseif plot_units == :m # longitude inaccuracy scales with map size
            lon_mid = map_xx[xx_mid]
            lat_mid = map_yy[yy_mid]
            map_xx .= dlon2de.(deg2rad.(map_xx .- lon_mid), deg2rad.(lat_mid))
            map_yy .= dlat2dn.(deg2rad.(map_yy .- lat_mid), deg2rad.(map_yy))
        end
    else
        error("[$map_units] map xx/yy units not defined")
    end

    if plot_units == :rad
        xlab = ((map_xx[end] == nx) | !axis) ? "" : "longitude [rad]"
        ylab = ((map_yy[end] == ny) | !axis) ? "" : "latitude [rad]"
    elseif plot_units == :deg
        xlab = ((map_xx[end] == nx) | !axis) ? "" : "longitude [deg]"
        ylab = ((map_yy[end] == ny) | !axis) ? "" : "latitude [deg]"
    elseif plot_units == :m
        xlab = ((map_xx[end] == nx) | !axis) ? "" : "easting [m]"
        ylab = ((map_yy[end] == ny) | !axis) ? "" : "northing [m]"
    else
        error("$plot_units plot xx/yy units not defined")
    end

    # map indices with zeros (ind0)
    (ind0,_,nx,ny) = map_params(map_map,map_xx,map_yy)

    # select color scale
    c = map_cs(map_color)

    # adjust color scale and set contour limits based on map data
    if length(map_map) > length(c)
        clims == (0,0) && ((c,clims) = map_clims(c,map_map))
    else
        clims = extrema(map_map)
    end

    # values outside contour limits set to contour limits (plotly workaround)
    map_map .= clamp.(map_map,clims[1],clims[2])

    # set data points without actual data to NaN (not plotted)
    map_map[ind0] .= NaN

    xind = downsample(1:nx,Nmax)
    yind = downsample(1:ny,Nmax)

    b_e # backend
    contourf!(p1,map_xx[xind],map_yy[yind],map_map[yind,xind],dpi=dpi,lw=0,
              c=c,bg=bg_color,clims=clims,margin=margin*mm,legend=legend,
              axis=axis,xticks=axis,yticks=axis,xlab=xlab,ylab=ylab,lab=false);

end # function plot_map!

"""
    plot_map!(p1, mapS::Union{MapS,MapSd,MapS3D};
              clims::Tuple       = (-500,500),
              dpi::Int           = 200,
              margin::Int        = 2,
              Nmax::Int          = 6*dpi,
              legend::Bool       = true,
              axis::Bool         = true,
              map_color::Symbol  = :usgs,
              bg_color::Symbol   = :white,
              map_units::Symbol  = :rad,
              plot_units::Symbol = :deg
              b_e                = gr())

Plot map on an existing plot.

**Arguments:**
- `p1`:         existing plot
- `mapS`:       `MapS`, `MapSd`, or `MapS3D` scalar magnetic anomaly map struct
- `clims`:      (optional) color scale limits
- `dpi`:        (optional) dots per inch (image resolution)
- `margin`:     (optional) margin around plot [mm]
- `Nmax`:       (optional) maximum number of data points plotted
- `legend`:     (optional) if true, show legend
- `axis`:       (optional) if true, show axes
- `map_color`:  (optional) filled contour color scheme {`:usgs`,`:gray`,`:gray1`,`:gray2`,`:plasma`,`:magma`}
- `bg_color`:   (optional) background color
- `map_units`:  (optional) map  xx/yy units {`:rad`,`:deg`}
- `plot_units`: (optional) plot xx/yy units {`:rad`,`:deg`,`:m`}
- `b_e`:        (optional) plotting backend

**Returns:**
- `nothing`: map is plotted on `p1`
"""
function plot_map!(p1, mapS::Union{MapS,MapSd,MapS3D};
                   clims::Tuple       = (-500,500),
                   dpi::Int           = 200,
                   margin::Int        = 2,
                   Nmax::Int          = 6*dpi,
                   legend::Bool       = true,
                   axis::Bool         = true,
                   map_color::Symbol  = :usgs,
                   bg_color::Symbol   = :white,
                   map_units::Symbol  = :rad,
                   plot_units::Symbol = :deg,
                   b_e                = gr())
    typeof(mapS) <: MapS3D && @info("3D map provided, using map at lowest altitude")
    plot_map!(p1,mapS.map[:,:,1],mapS.xx,mapS.yy;
              clims      = clims,
              dpi        = dpi,
              margin     = margin,
              Nmax       = Nmax,
              legend     = legend,
              axis       = axis,
              map_color  = map_color,
              bg_color   = bg_color,
              map_units  = map_units,
              plot_units = plot_units,
              b_e        = b_e)
end # function plot_map!

"""
    plot_map!(p1, p2, p3, mapV::MapV;
              clims::Tuple       = (-500,500),
              dpi::Int           = 200,
              margin::Int        = 2,
              Nmax::Int          = 6*dpi,
              legend::Bool       = true,
              axis::Bool         = true,
              map_color::Symbol  = :usgs,
              bg_color::Symbol   = :white,
              map_units::Symbol  = :rad,
              plot_units::Symbol = :deg
              b_e                = gr())

Plot map on an existing plot.

**Arguments:**
- `p1`:         existing plot
- `p2`:         existing plot
- `p3`:         existing plot
- `mapV`:       `MapV` vector magnetic anomaly map struct
- `clims`:      (optional) color scale limits
- `dpi`:        (optional) dots per inch (image resolution)
- `margin`:     (optional) margin around plot [mm]
- `Nmax`:       (optional) maximum number of data points plotted
- `legend`:     (optional) if true, show legend
- `axis`:       (optional) if true, show axes
- `map_color`:  (optional) filled contour color scheme {`:usgs`,`:gray`,`:gray1`,`:gray2`,`:plasma`,`:magma`}
- `bg_color`:   (optional) background color
- `map_units`:  (optional) map  xx/yy units {`:rad`,`:deg`}
- `plot_units`: (optional) plot xx/yy units {`:rad`,`:deg`,`:m`}
- `b_e`:        (optional) plotting backend

**Returns:**
- `nothing`: map (if typeof(`map_map`) = `MapV`, `mapX`) is plotted on `p1`
- `nothing`: map (if typeof(`map_map`) = `MapV`, `mapY`) is plotted on `p2`
- `nothing`: map (if typeof(`map_map`) = `MapV`, `mapZ`) is plotted on `p3`
"""
function plot_map!(p1, p2, p3, mapV::MapV;
                   clims::Tuple       = (-500,500),
                   dpi::Int           = 200,
                   margin::Int        = 2,
                   Nmax::Int          = 6*dpi,
                   legend::Bool       = true,
                   axis::Bool         = true,
                   map_color::Symbol  = :usgs,
                   bg_color::Symbol   = :white,
                   map_units::Symbol  = :rad,
                   plot_units::Symbol = :deg,
                   b_e                = gr())
    for (p_,map_) in zip([p1,p2,p3],[mapV.mapX,mapV.mapY,mapV.mapZ])
        plot_map!(p_,map_,mapV.xx,mapV.yy;
                  clims      = clims,
                  dpi        = dpi,
                  margin     = margin,
                  Nmax       = Nmax,
                  legend     = legend,
                  axis       = axis,
                  map_color  = map_color,
                  bg_color   = bg_color,
                  map_units  = map_units,
                  plot_units = plot_units,
                  b_e        = b_e)
    end
end # function plot_map!

"""
    plot_map(map_map::Matrix,
             map_xx::Vector     = [],
             map_yy::Vector     = [];
             clims::Tuple       = (0,0),
             dpi::Int           = 200,
             margin::Int        = 2,
             Nmax::Int          = 6*dpi,
             legend::Bool       = true,
             axis::Bool         = true,
             map_color::Symbol  = :usgs,
             bg_color::Symbol   = :white,
             map_units::Symbol  = :rad,
             plot_units::Symbol = :deg,
             b_e                = gr())

Plot map.

**Arguments:**
- `map_map`:    `ny` x `nx` 2D gridded map data
- `map_xx`:     `nx` map x-direction (longitude) coordinates [rad] or [deg]
- `map_yy`:     `ny` map y-direction (latitude)  coordinates [rad] or [deg]
- `clims`:      (optional) color scale limits
- `dpi`:        (optional) dots per inch (image resolution)
- `margin`:     (optional) margin around plot [mm]
- `Nmax`:       (optional) maximum number of data points plotted
- `legend`:     (optional) if true, show legend
- `axis`:       (optional) if true, show axes
- `map_color`:  (optional) filled contour color scheme {`:usgs`,`:gray`,`:gray1`,`:gray2`,`:plasma`,`:magma`}
- `bg_color`:   (optional) background color
- `map_units`:  (optional) map  xx/yy units {`:rad`,`:deg`}
- `plot_units`: (optional) plot xx/yy units {`:rad`,`:deg`,`:m`}
- `b_e`:        (optional) plotting backend

**Returns:**
- `p1`: plot of map
"""
function plot_map(map_map::Matrix,
                  map_xx::Vector     = [],
                  map_yy::Vector     = [];
                  clims::Tuple       = (0,0),
                  dpi::Int           = 200,
                  margin::Int        = 2,
                  Nmax::Int          = 6*dpi,
                  legend::Bool       = true,
                  axis::Bool         = true,
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
              Nmax       = Nmax,
              legend     = legend,
              axis       = axis,
              map_color  = map_color,
              bg_color   = bg_color,
              map_units  = map_units,
              plot_units = plot_units,
              b_e        = b_e)
end # function plot_map

"""
    plot_map(map_map::Map;
             clims::Tuple       = (0,0),
             dpi::Int           = 200,
             margin::Int        = 2,
             Nmax::Int          = 6*dpi,
             legend::Bool       = true,
             axis::Bool         = true,
             map_color::Symbol  = :usgs,
             bg_color::Symbol   = :white,
             map_units::Symbol  = :rad,
             plot_units::Symbol = :deg,
             b_e                = gr())

Plot map.

**Arguments:**
- `map_map`:    `Map` magnetic anomaly map struct
- `clims`:      (optional) color scale limits
- `dpi`:        (optional) dots per inch (image resolution)
- `margin`:     (optional) margin around plot [mm]
- `Nmax`:       (optional) maximum number of data points plotted
- `legend`:     (optional) if true, show legend
- `axis`:       (optional) if true, show axes
- `map_color`:  (optional) filled contour color scheme {`:usgs`,`:gray`,`:gray1`,`:gray2`,`:plasma`,`:magma`}
- `bg_color`:   (optional) background color
- `map_units`:  (optional) map  xx/yy units {`:rad`,`:deg`}
- `plot_units`: (optional) plot xx/yy units {`:rad`,`:deg`,`:m`}
- `b_e`:        (optional) plotting backend

**Returns:**
- `p1`: plot of map (if typeof(`map_map`) = `MapV`, `mapX`)
- `p2`: if typeof(`map_map`) = `MapV`, `mapY`
- `p3`: if typeof(`map_map`) = `MapV`, `mapZ`
"""
function plot_map(map_map::Map;
                  clims::Tuple       = (0,0),
                  dpi::Int           = 200,
                  margin::Int        = 2,
                  Nmax::Int          = 6*dpi,
                  legend::Bool       = true,
                  axis::Bool         = true,
                  map_color::Symbol  = :usgs,
                  bg_color::Symbol   = :white,
                  map_units::Symbol  = :rad,
                  plot_units::Symbol = :deg,
                  b_e                = gr())
    b_e # backend
    if typeof(map_map) <: Union{MapS,MapSd,MapS3D}
        p1 = plot(legend=legend,lab=false)
        plot_map!(p1,map_map;
                  clims      = clims,
                  dpi        = dpi,
                  margin     = margin,
                  Nmax       = Nmax,
                  legend     = legend,
                  axis       = axis,
                  map_color  = map_color,
                  bg_color   = bg_color,
                  map_units  = map_units,
                  plot_units = plot_units,
                  b_e        = b_e)
        return (p1)
    elseif typeof(map_map) <: MapV
        p1 = plot(legend=legend,lab=false)
        p2 = plot(legend=legend,lab=false)
        p3 = plot(legend=legend,lab=false)
        plot_map!(p1,p2,p3,map_map;
                  clims      = clims,
                  dpi        = dpi,
                  margin     = margin,
                  Nmax       = Nmax,
                  legend     = legend,
                  axis       = axis,
                  map_color  = map_color,
                  bg_color   = bg_color,
                  map_units  = map_units,
                  plot_units = plot_units,
                  b_e        = b_e)
        return (p1, p2, p3)
    end
end # function plot_map

"""
    map_cs(map_color::Symbol=:usgs)

Internal helper function to select map color scale. Default is from the USGS:
https://mrdata.usgs.gov/magnetic/namag.png

**Arguments:**
- `map_color`: (optional) filled contour color scheme {`:usgs`,`:gray`,`:gray1`,`:gray2`,`:plasma`,`:magma`}

**Returns:**
- `c`: color scale
"""
function map_cs(map_color::Symbol=:usgs)

    if map_color == :usgs # standard for geological maps
        f = readdlm(usgs,',')
        c = cgrad([RGB(f[i,:]...) for i in axes(f,1)])
    elseif map_color == :gray # light gray
        c = cgrad(cgrad(:gist_gray)[61:90])
    elseif map_color == :gray1 # light gray (lower end)
        c = cgrad(cgrad(:gist_gray)[61:81])
    elseif map_color == :gray2 # light gray (upper end)
        c = cgrad(cgrad(:gist_gray)[71:90])
    else # :plasma, :magma
        c = cgrad(map_color)
    end

    return (c)
end # function map_cs

"""
    map_clims(c, map_map)

Internal helper function to adjust color scale for histogram equalization
(maximum contrast) and set contour limits based on map data.

**Arguments:**
- `c`:       original color scale
- `map_map`: `ny` x `nx` 2D gridded map data

**Returns:**
- `c`:     new color scale
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
               Nmax::Int          = 5000,
               show_plot::Bool    = true,
               zoom_plot::Bool    = false,
               path_color::Symbol = :ignore)

Plot flight path on an existing plot.

**Arguments:**
- `p1`:         existing plot (e.g., map)
- `lat`:        latitude  [rad]
- `lon`:        longitude [rad]
- `lab`:        (optional) data (legend) label
- `Nmax`:       (optional) maximum number of data points plotted
- `show_plot`:  (optional) if true, show plot
- `zoom_plot`:  (optional) if true, zoom plot onto flight path
- `path_color`: (optional) path color {`:ignore`,`:black`,`:gray`,`:red`,`:orange`,`:yellow`,`:green`,`:cyan`,`:blue`,`:purple`}

**Returns:**
- `nothing`: flight path is plotted on `p1`
"""
function plot_path!(p1, lat, lon;
                    lab = "",
                    Nmax::Int          = 5000,
                    show_plot::Bool    = true,
                    zoom_plot::Bool    = false,
                    path_color::Symbol = :ignore)

    lon = downsample(rad2deg.(deepcopy(lon)),Nmax)
    lat = downsample(rad2deg.(deepcopy(lat)),Nmax)

    if path_color in [:black,:gray,:red,:orange,:yellow,:green,:cyan,:blue,:purple]
        p1 = plot!(p1,lon,lat,lab=lab,legend=true,lc=path_color)
    else
        p1 = plot!(p1,lon,lat,lab=lab,legend=true)
    end

    if zoom_plot
        xlim = get_lim(lon,0.2)
        ylim = get_lim(lat,0.2)
        p1 = plot!(p1,xlim=xlim,ylim=ylim)
    end

    show_plot && display(p1)

    return (p1)
end # function plot_path!

"""
    plot_path!(p1, path::Path, ind=trues(path.N);
               lab = "",
               Nmax::Int          = 5000,
               show_plot::Bool    = true,
               zoom_plot::Bool    = false,
               path_color::Symbol = :ignore)

Plot flight path on an existing plot.

**Arguments:**
- `p1`:         existing plot (e.g., map)
- `path`:       `Path` struct, i.e., `Traj` trajectory struct, `INS` inertial navigation system struct, or `FILTout` filter extracted output struct
- `ind`:        (optional) selected data indices
- `lab`:        (optional) data (legend) label
- `Nmax`:       (optional) maximum number of data points plotted
- `show_plot`:  (optional) if true, show plot
- `zoom_plot`:  (optional) if true, zoom plot onto flight path
- `path_color`: (optional) path color {`:ignore`,`:black`,`:gray`,`:red`,`:orange`,`:yellow`,`:green`,`:cyan`,`:blue`,`:purple`}

**Returns:**
- `nothing`: flight path is plotted on `p1`
"""
function plot_path!(p1, path::Path, ind=trues(path.N);
                    lab = "",
                    Nmax::Int          = 5000,
                    show_plot::Bool    = true,
                    zoom_plot::Bool    = false,
                    path_color::Symbol = :ignore)
    plot_path!(p1,path.lat[ind],path.lon[ind];
               lab        = lab,
               Nmax       = Nmax,
               show_plot  = show_plot,
               zoom_plot  = zoom_plot,
               path_color = path_color)
end # function plot_path!

"""
    plot_path(lat, lon;
              lab = "",
              dpi::Int           = 200,
              margin::Int        = 2,
              Nmax::Int          = 5000,
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
- `Nmax`:       (optional) maximum number of data points plotted
- `show_plot`:  (optional) if true, show plot
- `zoom_plot`:  (optional) if true, zoom plot onto flight path
- `path_color`: (optional) path color {`:ignore`,`:black`,`:gray`,`:red`,`:orange`,`:yellow`,`:green`,`:cyan`,`:blue`,`:purple`}

**Returns:**
- `p1`: plot of flight path
"""
function plot_path(lat, lon;
                   lab = "",
                   dpi::Int           = 200,
                   margin::Int        = 2,
                   Nmax::Int          = 5000,
                   show_plot::Bool    = true,
                   zoom_plot::Bool    = true,
                   path_color::Symbol = :ignore)
    p1 = plot(xlab="longitude [deg]",ylab="latitude [deg]",
              dpi=dpi,margin=margin*mm)
    plot_path!(p1,lat,lon;
               lab        = lab,
               Nmax       = Nmax,
               show_plot  = show_plot,
               zoom_plot  = zoom_plot,
               path_color = path_color)
end # function plot_path

"""
    plot_path(path::Path, ind=trues(path.N);
              lab = "",
              dpi::Int           = 200,
              margin::Int        = 2,
              Nmax::Int          = 5000,
              show_plot::Bool    = true,
              zoom_plot::Bool    = true,
              path_color::Symbol = :ignore)

Plot flight path.

**Arguments:**
- `path`:       `Path` struct, i.e., `Traj` trajectory struct, `INS` inertial navigation system struct, or `FILTout` filter extracted output struct
- `ind`:        (optional) selected data indices
- `lab`:        (optional) data (legend) label
- `dpi`:        (optional) dots per inch (image resolution)
- `margin`:     (optional) margin around plot [mm]
- `Nmax`:       (optional) maximum number of data points plotted
- `show_plot`:  (optional) if true, show plot
- `zoom_plot`:  (optional) if true, zoom plot onto flight path
- `path_color`: (optional) path color {`:ignore`,`:black`,`:gray`,`:red`,`:orange`,`:yellow`,`:green`,`:cyan`,`:blue`,`:purple`}

**Returns:**
- `p1`: plot of flight path
"""
function plot_path(path::Path, ind=trues(path.N);
                   lab = "",
                   dpi::Int           = 200,
                   margin::Int        = 2,
                   Nmax::Int          = 5000,
                   show_plot::Bool    = true,
                   zoom_plot::Bool    = true,
                   path_color::Symbol = :ignore)
    plot_path(path.lat[ind],path.lon[ind];
              lab        = lab,
              dpi        = dpi,
              margin     = margin,
              Nmax       = Nmax,
              show_plot  = show_plot,
              zoom_plot  = zoom_plot,
              path_color = path_color)
end # function plot_path

"""
    plot_events!(p1, t, lab=nothing; legend::Symbol = :outertopright)

Plot in-flight event on existing plot.

**Arguments:**
- `p1`:      existing plot (e.g., time series of magnetometer measurements)
- `t`:       time of in-flight event
- `lab`:     (optional) in-flight event (legend) label
- `legend`:  (optional) legend position (e.g., `:topleft`,`:outertopright`)

**Returns:**
- `p1`: in-flight events are plotted on `p1`
"""
function plot_events!(p1, t, lab=nothing; legend::Symbol = :outertopright)
    t = [t;] # ensure vector
    plot!(p1,t,lab=lab,c=:red,ls=:dash,lt=:vline,legend=legend)
    return (p1)
end # function plot_events!

"""
    plot_events!(p1, df_event::DataFrame, flight::Symbol, keyword::String = "";
                 show_lab::Bool  = true,
                 t0              = 0,
                 t_units::Symbol = :min,
                 legend::Symbol  = :outertopright)

Plot in-flight event(s) on existing plot.

**Arguments:**
- `p1`:       existing plot (e.g., time series of magnetometer measurements)
- `df_event`: lookup table (DataFrame) of in-flight events
|**Field**|**Type**|**Description**
|:--|:--|:--
`flight`|`Symbol`| flight name (e.g., `:Flt1001`)
`tt`    |`Real`  | time of `event` [s]
`event` |`String`| event description
- `flight`:   flight name (e.g., `:Flt1001`)
- `keyword`:  (optional) keyword to search within events, case insensitive
- `show_lab`: (optional) if true, show in-flight event (legend) label(s)
- `t0`:       (optional) time offset [`t_units`]
- `t_units`:  (optional) time units used for plotting {`:sec`,`:min`}
- `legend`:   (optional) legend position (e.g., `:topleft`,`:outertopright`)

**Returns:**
- `p1`: in-flight events are plotted on `p1`
"""
function plot_events!(p1, df_event::DataFrame, flight::Symbol, keyword::String = "";
                      show_lab::Bool  = true,
                      t0              = 0,
                      t_units::Symbol = :min,
                      legend::Symbol  = :outertopright)
    tt_lim = xlims(p1) .+ t0
    t_units == :min && (tt_lim = tt_lim.*60)
    df = filter_events(df_event,flight,keyword;tt_lim=tt_lim)
    for i in axes(df,1)
        lab = show_lab ? string(df[i,:event]) : nothing
        t   = df[i,:tt]
        t_units == :min && (t = t/60)
        plot_events!(p1,t-t0,lab;legend=legend)
    end
    return (p1)
end # function plot_events!

function plot_events!(p1, flight::Symbol, df_event::DataFrame;
                      show_lab::Bool  = true,
                      t0              = 0,
                      t_units::Symbol = :min,
                      legend::Symbol  = :outertopright)
    @warn("this version of plot_events!() will be deprecated, use plot_events!(p1, df_event::DataFrame, flight::Symbol)")
    plot_events!(p1,df_event,flight;
                 show_lab=show_lab,t0=t0,t_units=t_units,legend=legend)
end # function plot_events!

"""
    map_check(map_map::Map, lat, lon, alt = median(map_map.alt)*one.(lat))

Check if latitude and longitude points are on given map.

**Arguments:**
- `map_map`: `Map` magnetic anomaly map struct
- `lat`:     latitude  [rad]
- `lon`:     longitude [rad]
- `alt`:     (optional) altitude [m], only used for `MapS3D`

**Returns:**
- `bool`: if true, all `lat` & `lon` (& `alt`) points are on `map_map`
"""
function map_check(map_map::Map, lat, lon, alt = median(map_map.alt)*one.(lat))
    if typeof(map_map) <: MapV
        itp_map = map_itp(map_map,:X,:linear)
    else
        itp_map = map_itp(map_map,:linear)
    end
    xx_lim  = extrema(map_map.xx)
    yy_lim  = extrema(map_map.yy)
    alt_lim = extrema(map_map.alt)
    N       = length(lat)
    val     = trues(N)
    for i = 1:N
        xx_lim[1] < lon[i] < xx_lim[2]                    || (val[i] = false)
        yy_lim[1] < lat[i] < yy_lim[2]                    || (val[i] = false)
        if typeof(map_map) <: Union{MapS,MapSd,MapV}
            val[i] && (itp_map(lon[i],lat[i]) != 0        || (val[i] = false))
        elseif typeof(map_map) <: MapS3D
            alt_lim[1] < alt[i] < alt_lim[2]              || (val[i] = false)
            val[i] && (itp_map(lon[i],lat[i],alt[i]) != 0 || (val[i] = false))
        end
    end
    return all(val)
end # function map_check

"""
    map_check(map_map::Map, path::Path, ind=trues(path.N))

Check if latitude and longitude points are on given map.

**Arguments:**
- `map_map`: `Map` magnetic anomaly map struct
- `path`:    `Path` struct, i.e., `Traj` trajectory struct, `INS` inertial navigation system struct, or `FILTout` filter extracted output struct
- `ind`:     (optional) selected data indices

**Returns:**
- `bool`: if true, all `path`[`ind`] points are on `map_map`
"""
function map_check(map_map::Map, path::Path, ind=trues(path.N))
    map_check(map_map,path.lat[ind],path.lon[ind],path.alt[ind])
end # function map_check

"""
    map_check(map_map_vec::Vector, path::Path, ind=trues(path.N))

Check if latitude and longitude points are on given maps.

**Arguments:**
- `map_map_vec`: vector of `Map` magnetic anomaly map structs
- `path`:        `Path` struct, i.e., `Traj` trajectory struct, `INS` inertial navigation system struct, or `FILTout` filter extracted output struct
- `ind`:         (optional) selected data indices

**Returns:**
- `bools`: if true, all `path`[`ind`] points are on `map_map_vec`[i]
"""
function map_check(map_map_vec::Vector, path::Path, ind=trues(path.N))
    [map_check(map_map_vec[i],path,ind) for i in eachindex(map_map_vec)]
end # function map_check

"""
    get_map_val(map_map::Map, lat, lon, alt; α=200, return_itp::Bool=false)

Get scalar magnetic anomaly map values along a flight path. `map_map` is upward
and/or downward continued to `alt` as necessary.

**Arguments:**
- `map_map`:    `Map` magnetic anomaly map struct
- `lat`:        latitude  [rad]
- `lon`:        longitude [rad]
- `alt`:        altitude [m]
- `α`:          (optional) regularization parameter for downward continuation
- `return_itp`: (optional) if true, also return `itp_map`

**Returns:**
- `map_val`: scalar magnetic anomaly map values
- `itp_map`: if `return_itp = true`, map interpolation function
"""
function get_map_val(map_map::Map, lat, lon, alt; α=200, return_itp::Bool=false)
    if typeof(map_map) <: Union{MapS,MapSd}
        all(map_map.alt .> 0) && (map_map = upward_fft(map_map,median(alt);α=α))
        itp_map = map_itp(map_map)
        map_val = itp_map.(lon,lat)
    elseif typeof(map_map) <: MapS3D
        alt_min = map_map.alt[1]
        alt_max = map_map.alt[end]
        dalt    = get_step(map_map.alt)
        while minimum(alt) < alt_min
            alt_min -= dalt
        end
        while alt_max < maximum(alt)
            alt_max += dalt
        end
        if (alt_min < map_map.alt[1]) | (map_map.alt[end] < alt_max)
            map_map = upward_fft(map_map,alt_min:dalt:alt_max;α=α)
        end
        itp_map = map_itp(map_map)
        map_val = itp_map.(lon,lat,alt)
    elseif typeof(map_map) <: MapV
        all(map_map.alt .> 0) && (map_map = upward_fft(map_map,median(alt);α=α))
        itp_map = (map_itp(map_map,:X),
                   map_itp(map_map,:Y),
                   map_itp(map_map,:Z))
        map_val = (itp_map[1].(lon,lat),
                   itp_map[2].(lon,lat),
                   itp_map[3].(lon,lat))
    end
    if return_itp
        return (map_val, itp_map)
    else
        return (map_val)
    end
end # function get_map_val

"""
    get_map_val(map_map::Map, path::Path, ind=trues(path.N);
                α=200, return_itp::Bool=false)

Get scalar magnetic anomaly map values along a flight path. `map_map` is upward
and/or downward continued to `alt` as necessary.

**Arguments:**
- `map_map`:    `Map` magnetic anomaly map struct
- `path`:       `Path` struct, i.e., `Traj` trajectory struct, `INS` inertial navigation system struct, or `FILTout` filter extracted output struct
- `ind`:        (optional) selected data indices
- `α`:          (optional) regularization parameter for downward continuation
- `return_itp`: (optional) if true, also return `map_itp`

**Returns:**
- `map_val`: scalar magnetic anomaly map values
- `map_itp`: if `return_itp = true`, map interpolation function
"""
function get_map_val(map_map::Map, path::Path, ind=trues(path.N);
                     α=200, return_itp::Bool=false)
    get_map_val(map_map,path.lat[ind],path.lon[ind],path.alt[ind];α=α,return_itp=return_itp)
end # function get_map_val

"""
    get_map_val(map_map_vec::Vector, path::Path, ind=trues(path.N); α=200)

Get scalar magnetic anomaly map values from multiple maps along a flight path.
Each map in `map_map_vec` is upward and/or downward continued to `alt` as necessary.

**Arguments:**
- `map_map_vec`: vector of `Map` magnetic anomaly map structs
- `path`:        `Path` struct, i.e., `Traj` trajectory struct, `INS` inertial navigation system struct, or `FILTout` filter extracted output struct
- `ind`:         (optional) selected data indices
- `α`:           (optional) regularization parameter for downward continuation

**Returns:**
- `map_vals`: vector of scalar magnetic anomaly map values
"""
function get_map_val(map_map_vec::Vector, path::Path, ind=trues(path.N); α=200)
    [get_map_val(map_map_vec[i],path,ind;
                 α=α,return_itp=false) for i in eachindex(map_map_vec)]
end # function get_map_val

"""
    get_step(x::Vector)

Internal helper function to get the step size (spacing) of elements in `x`.
"""
function get_step(x::Vector)
    step(LinRange(x[1],x[end],length(x)))
end # function get_step

"""
    get_cached_map(map_cache::Map_Cache, lat::Real, lon::Real, alt::Real;
                   silent::Bool = false)

Get cached map at specific location.

**Arguments:**
- `map_cache`: `Map_Cache` map cache struct
- `lat`:       latitude  [rad]
- `lon`:       longitude [rad]
- `alt`:       altitude  [m]
- `silent`:    (optional) if true, no print outs

**Returns:**
- `itp_mapS`: scalar map interpolation function
"""
function get_cached_map(map_cache::Map_Cache, lat::Real, lon::Real, alt::Real;
                        silent::Bool = false)
    alt_lev = -1  # initialize
    o = map_cache # convenience

    try
        for (i, ind) in enumerate(o.map_sort_ind)
            if (o.maps[ind].alt <= alt) & map_check(o.maps[ind],lat,lon)
                alt_lev = max(floor(alt/o.dz)*o.dz, o.maps[ind].alt)
                if (i, alt_lev) ∉ keys(o.map_cache)
                    silent || @info("generating cached map at $alt_lev m")
                    mapS     = upward_fft(o.maps_filled[ind],alt_lev)
                    itp_mapS = map_itp(mapS)
                    o.map_cache[(i,alt_lev)] = itp_mapS
                else
                    itp_mapS = o.map_cache[(i,alt_lev)]
                end
            end
        end
        alt_lev == -1 || silent || @info("using cached map at $alt_lev m")
    catch _
        @info("error encountered in using provided maps")
    end

    if alt_lev == -1
        alt_lev = max(floor(alt/o.dz)*o.dz, o.fallback.alt)
        if alt_lev ∉ keys(o.fallback_cache)
            silent || @info("generating fallback at $alt_lev m")
            mapS     = upward_fft(o.fallback,alt_lev)
            itp_mapS = map_itp(mapS)
            o.fallback_cache[alt_lev] = itp_mapS
        else
            itp_mapS = o.fallback_cache[alt_lev]
        end
        silent || @info("using fallback at $alt_lev m")
    end

    return (itp_mapS)
end # function get_cached_map

"""
    (map_cache::Map_Cache)(lat::Real, lon::Real, alt::Real; silent::Bool=false)

Get cached map value at specific location.

**Arguments:**
- `map_cache`: `Map_Cache` map cache struct
- `lat`:       latitude  [rad]
- `lon`:       longitude [rad]
- `alt`:       altitude  [m]
- `silent`:    (optional) if true, no print outs

**Returns:**
- `map_val`: scalar magnetic anomaly map value
"""
function (map_cache::Map_Cache)(lat::Real, lon::Real, alt::Real; silent::Bool=false)
    get_cached_map(map_cache,lat,lon,alt;silent=silent)(lon,lat)
end # function Map_Cache

"""
    map_border(map_map::Matrix, map_xx::Vector, map_yy::Vector;
               inner::Bool        = true,
               sort_border::Bool  = false,
               return_ind::Bool   = false)

Get map border from an unfilled map.

**Arguments:**
- `map_map`:      `ny` x `nx` 2D gridded map data
- `map_xx`:       `nx` map x-direction (longitude) coordinates
- `map_yy`:       `ny` map y-direction (latitude)  coordinates
- `inner`:        (optional) if true, get inner border, otherwise outer border
- `sort_border`:  (optional) if true, sort border data points sequentially
- `return_ind`:   (optional) if true, return `ind`

**Returns:**
- `xx`:  border x-direction (longitude) coordinates
- `yy`:  border y-direction (latitude)  coordinates
- `ind`: if `return_ind = true`, `BitMatrix` of border indices within `map_map`
"""
function map_border(map_map::Matrix, map_xx::Vector, map_yy::Vector;
                    inner::Bool       = true,
                    sort_border::Bool = false,
                    return_ind::Bool  = false)

    (ind0_,ind1_,nx,ny) = map_params(map_map)
    (Ny,Nx) = (ny,nx) .+ 2
    ind     = falses(Ny,Nx)
    ind0    = trues( Ny,Nx)
    ind1    = falses(Ny,Nx)
    ind0[2:Ny-1,2:Nx-1] = ind0_
    ind1[2:Ny-1,2:Nx-1] = ind1_

    # non-empty point along left/right edge of original map area
    for i in [2,Nx-1]
        for j = 2:Ny-1
            ind[j,i] = ind1[j,i]
        end
    end

    # non-empty point along top/bottom edge of original map area
    for i = 2:Nx-1
        for j in [2,Ny-1]
            ind[j,i] = ind1[j,i]
        end
    end

    if inner # non-empty point next to empty point(s)
        for i = 2:Nx-1
            for j = 2:Ny-1
                if ind1[j,i]
                    ind[j,i] = any(ind0[j-1:j+1,i-1:i+1])
                end
            end
        end
    else # empty point next to non-empty point(s)
        for i = 2:Nx-1
            for j = 2:Ny-1
                if ind0[j,i]
                    ind[j,i] = any(ind1[j-1:j+1,i-1:i+1])
                end
            end
        end
    end

    ind = ind[2:Ny-1,2:Nx-1]

    sort_border && map_border_clean!(ind)

    xx = vec(repeat(map_xx',ny,1)[ind])
    yy = vec(repeat(map_yy ,1,nx)[ind])

    if sort_border
        dx = get_step(map_xx)
        dy = get_step(map_yy)
        (xx,yy) = map_border_sort(xx,yy,dx,dy)
    end

    if return_ind
        return (xx, yy, ind)
    else
        return (xx, yy)
    end
end # function map_border

"""
    map_border(mapS::Union{MapS,MapSd,MapS3D};
               inner::Bool       = true,
               sort_border::Bool = false,
               return_ind::Bool  = false)

Get map border from an unfilled map.

**Arguments:**
- `mapS`:         `MapS`, `MapSd`, or `MapS3D` scalar magnetic anomaly map struct
- `inner`:        (optional) if true, get inner border, otherwise outer border
- `sort_border`:  (optional) if true, sort border data points sequentially
- `return_ind`:   (optional) if true, return `ind`

**Returns:**
- `xx`:  border x-direction (longitude) coordinates
- `yy`:  border y-direction (latitude)  coordinates
- `ind`: if `return_ind = true`, `BitMatrix` of border indices within `map_map`
"""
function map_border(mapS::Union{MapS,MapSd,MapS3D};
                    inner::Bool       = true,
                    sort_border::Bool = false,
                    return_ind::Bool  = false)
    typeof(mapS) <: MapS3D && @info("3D map provided, using map at lowest altitude")
    map_border(mapS.map[:,:,1],mapS.xx,mapS.yy;
               inner=inner,sort_border=sort_border,return_ind=return_ind)
end # function map_border

"""
    map_border_sort(xx::Vector, yy::Vector, dx, dy)

Sort map border data points sequentially.

**Arguments:**
- `xx`: border x-direction (longitude) coordinates
- `yy`: border y-direction (latitude)  coordinates
- `dx`: x-direction map step size
- `dy`: y-direction map step size

**Returns:**
- `xx`: border x-direction (longitude) coordinates, sorted
- `yy`: border y-direction (latitude)  coordinates, sorted
"""
function map_border_sort(xx::Vector, yy::Vector, dx, dy)
    d3 = 3*[dx,dy]
    ll = vcat(xx',yy')
    ll_out = zero.(ll)
    ind = falses(size(ll,2))
    ind[1] = true
    ll_out[:,1] = ll[:,ind]
    for i in axes(ll_out,2)[2:end]
        ll = ll[:, .!ind]
        pt = ll_out[:,i-1]
        ll_nn = ll[:, vec(all(abs.(ll .- pt) .< d3, dims=1))]
        try
            ind_nn = nn(KDTree(ll_nn),pt)[1]
            ind    = vec(all(ll_nn[:,ind_nn] .≈ ll, dims=1))
        catch _
            return (ll_out[1,1:i-1], ll_out[2,1:i-1])
        end
        ll_out[:,i] = ll[:,ind]
    end
    return (ll_out[1,:], ll_out[2,:])
end # function map_border_sort

"""
    map_border_singles(ind::BitMatrix)

Identify single map border data points that "stick out" of border.

**Arguments:**
- `ind`: `BitMatrix` of border indices

**Returns:**
- `ind`: `BitMatrix` of border indices, singles
"""
function map_border_singles(ind::BitMatrix)
    ind_    = ind
    (Ny,Nx) = size(ind) .+ 2
    ind     = falses(Ny,Nx)
    ind[2:Ny-1,2:Nx-1] = deepcopy(ind_)
    sum_ind = 0
    while sum_ind != sum(ind)
        sum_ind = sum(ind)
        for i = 2:Nx-1
            for j = 2:Ny-1
                if ind[j,i]
                    ind[j,i] = sum(ind[j,[i-1,i+1]]+ind[[j-1,j+1],i]) > 1
                end
            end
        end
    end
    return (ind_ .& .!ind[2:Ny-1,2:Nx-1])
end # map_border_singles

"""
    map_border_doubles(ind::BitMatrix)

Identify double map border data points that "stick out" of border.

**Arguments:**
- `ind`: `BitMatrix` of border indices

**Returns:**
- `ind`: `BitMatrix` of border indices, doubles
"""
function map_border_doubles(ind::BitMatrix)
    ind_    = ind
    (Ny,Nx) = size(ind) .+ 2
    ind     = falses(Ny,Nx)
    ind[2:Ny-1,2:Nx-1] = deepcopy(ind_)
    sum_ind = 0
    while sum_ind != sum(ind)
        sum_ind = sum(ind)
        for i = 3:Nx-1
            for j = 3:Ny-1
                if ind[j,i]
                    if all(ind[j-1:j,i])
                        ind[j-1:j,i] .= sum(ind[j-1:j,i-1]+ind[[j-2,j+1],i]) > 0
                    end
                    if all(ind[j-1:j,i])
                        ind[j-1:j,i] .= sum(ind[j-1:j,i+1]+ind[[j-2,j+1],i]) > 0
                    end
                    if all(ind[j,i-1:i])
                        ind[j,i-1:i] .= sum(ind[j-1,i-1:i]+ind[j,[i-2,i+1]]) > 0
                    end
                    if all(ind[j,i-1:i])
                        ind[j,i-1:i] .= sum(ind[j+1,i-1:i]+ind[j,[i-2,i+1]]) > 0
                    end
                end
            end
        end
    end
    return (ind_ .& .!ind[2:Ny-1,2:Nx-1])
end # map_border_doubles

"""
    map_border_clean!(ind::BitMatrix)

Identify single and double map border data points that "stick out" of border.

**Arguments:**
- `ind`: `BitMatrix` of border indices

**Returns:**
- `nothing`: `ind` is cleaned
"""
function map_border_clean!(ind::BitMatrix)
    sum_ind = 0
    while sum_ind != sum(ind)
        sum_ind = sum(ind)
        ind_singles = map_border_singles(ind)
        ind_doubles = map_border_doubles(ind)
        ind .= ind .& .!ind_singles .& .!ind_doubles
    end
end # map_border_clean!

"""
    map_border_clean(ind::BitMatrix)

Identify single and double map border data points that "stick out" of border.

**Arguments:**
- `ind`: `BitMatrix` of border indices

**Returns:**
- `ind`: `BitMatrix` of border indices, cleaned
"""
function map_border_clean(ind::BitMatrix)
    ind = deepcopy(ind)
    map_border_clean!(ind)
    return (ind)
end # map_border_clean

"""
    map_resample(map_map::Matrix, map_xx::Vector, map_yy::Vector,
                 map_xx_new::Vector, map_yy_new::Vector)

Resample map with new grid.

**Arguments:**
- `map_map`:    `ny` x `nx` 2D gridded map data
- `map_xx`:     `nx` map x-direction (longitude) coordinates
- `map_yy`:     `ny` map y-direction (latitude)  coordinates
- `map_xx_new`: `nx_new` map x-direction (longitude) coordinates to use for resampling
- `map_yy_new`: `ny_new` map y-direction (latitude)  coordinates to use for resampling

**Returns:**
- `map_map`: `ny_new` x `nx_new` 2D gridded map data, resampled
"""
function map_resample(map_map::Matrix, map_xx::Vector, map_yy::Vector,
                      map_xx_new::Vector, map_yy_new::Vector)

    map_map_ = deepcopy(map_map)
    (map_xx,ind_xx) = expand_range(map_xx,extrema(map_xx_new),true)
    (map_yy,ind_yy) = expand_range(map_yy,extrema(map_yy_new),true)
    map_map  = zeros(eltype(map_map),length(map_yy),length(map_xx))
    map_map[ind_yy,ind_xx] = map_map_

    ind1     = convert.(eltype(map_map),map_params(map_map,map_xx,map_yy)[2])
    itp_ind1 = map_itp(ind1   ,map_xx,map_yy,:linear)
    itp_map  = map_itp(map_map,map_xx,map_yy,:linear)
    map_map  = zeros(eltype(map_map),length(map_yy_new),length(map_xx_new))

    for (i,x) in enumerate(map_xx_new)
        for (j,y) in enumerate(map_yy_new)
            itp_ind1(x,y) ≈ 1 && (@inbounds map_map[j,i] = itp_map(x,y))
        end
    end

    return (map_map)
end # function map_resample

"""
    map_resample(mapS::MapS, map_xx_new::Vector, map_yy_new::Vector)

Resample map with new grid.

**Arguments:**
- `mapS`:       `MapS` scalar magnetic anomaly map struct
- `map_xx_new`: `nx_new` map x-direction (longitude) coordinates to use for resampling
- `map_yy_new`: `ny_new` map y-direction (latitude)  coordinates to use for resampling

**Returns:**
- `mapS`: `MapS` scalar magnetic anomaly map struct, resampled
"""
function map_resample(mapS::MapS, map_xx_new::Vector, map_yy_new::Vector)
    return MapS(map_resample(mapS.map,mapS.xx,mapS.yy,map_xx_new,map_yy_new),
                map_xx_new, map_yy_new, mapS.alt)
end # function map_resample

"""
    map_resample(mapS::MapS, mapS_new::MapS)

Resample map with new grid.

**Arguments:**
- `mapS`:     `MapS` scalar magnetic anomaly map struct
- `mapS_new`: `MapS` scalar magnetic anomaly map struct to use for resampling

**Returns:**
- `mapS`: `MapS` scalar magnetic anomaly map struct, resampled
"""
function map_resample(mapS::MapS, mapS_new::MapS)
    map_resample(mapS,mapS_new.xx,mapS_new.yy)
end # function map_resample

"""
    map_combine(mapS::MapS, mapS_fallback::MapS = get_map(namad);
                xx_lim::Tuple = get_lim(mapS.xx,0.1),
                yy_lim::Tuple = get_lim(mapS.yy,0.1),
                α             = 200)

Combine two maps at same altitude.

**Arguments:**
- `mapS`:          `MapS` scalar magnetic anomaly map struct
- `mapS_fallback`: (optional) fallback `MapS` scalar magnetic anomaly map struct
- `xx_lim`:        (optional) x-direction map limits `(xx_min,xx_max)`
- `yy_lim`:        (optional) y-direction map limits `(yy_min,yy_max)`
- `α`:             (optional) regularization parameter for downward continuation

**Returns:**
- `mapS`: `MapS` scalar magnetic anomaly map struct, combined
"""
function map_combine(mapS::MapS, mapS_fallback::MapS = get_map(namad);
                     xx_lim::Tuple = get_lim(mapS.xx,0.1),
                     yy_lim::Tuple = get_lim(mapS.yy,0.1),
                     α             = 200)

    # map setup
    mapS = map_trim(mapS)
    (map_xx,ind_xx) = expand_range(mapS.xx,xx_lim)
    (map_yy,ind_yy) = expand_range(mapS.yy,yy_lim)

    # fallback map setup
    @assert clamp.(extrema(mapS_fallback.xx),xx_lim...) == xx_lim "xx_lim are outside mapS_fallback xx limits"
    @assert clamp.(extrema(mapS_fallback.yy),yy_lim...) == yy_lim "yy_lim are outside mapS_fallback yy limits"
    mapS_fallback = upward_fft(mapS_fallback,mapS.alt;α=α)
    mapS_fallback = map_trim(mapS_fallback;pad=1,
                             xx_lim=extrema(map_xx),yy_lim=extrema(map_yy))
    itp_mapS = map_itp(mapS_fallback)

    (lon,lat,ind) = map_border(mapS;
                               inner       = true,
                               sort_border = false,
                               return_ind  = true)
    mapS.map[ind] = (mapS.map[ind] + itp_mapS.(lon,lat)) / 2

    map_map = zeros(eltype(mapS.map),length(map_yy),length(map_xx))
    map_map[ind_yy,ind_xx] = mapS.map
    (ind0,_,Nx,Ny) = map_params(map_map)
    for i = 1:Nx
        for j = 1:Ny
            ind0[j,i] && (map_map[j,i] = itp_mapS(map_xx[i],map_yy[j]))
        end
    end

    return MapS(map_map, map_xx, map_yy, mapS.alt)
end # function map_combine

"""
    map_combine(mapS_vec::Vector, mapS_fallback::MapS = get_map(namad);
                n_levels::Int      = 3,
                dx                 = get_step(mapS_vec[1].xx),
                dy                 = get_step(mapS_vec[1].yy),
                xx_lim::Tuple      = get_lim(mapS_vec[1].xx,0.5),
                yy_lim::Tuple      = get_lim(mapS_vec[1].yy,0.5),
                α                  = 200,
                use_fallback::Bool = true)

Combine maps at different altitudes. Lowest and highest maps are directly used
(with resampling & resizing), with intermediate maps determined by `n_levels`.

**Arguments:**
- `mapS_vec`:      vector of `MapS` scalar magnetic anomaly map structs
- `mapS_fallback`: (optional) fallback `MapS` scalar magnetic anomaly map struct
- `n_levels`:      (optional) number of map altitude levels
- `dx`:            (optional) x-direction map step size (desired)
- `dy`:            (optional) y-direction map step size (desired)
- `xx_lim`:        (optional) x-direction map limits `(xx_min,xx_max)`
- `yy_lim`:        (optional) y-direction map limits `(yy_min,yy_max)`
- `α`:             (optional) regularization parameter for downward continuation
- `use_fallback`:  (optional) if true, use `mapS_fallback` for missing map data

**Returns:**
- `mapS3D`: `MapS3D` 3D (multi-level) scalar magnetic anomaly map struct
"""
function map_combine(mapS_vec::Vector, mapS_fallback::MapS = get_map(namad);
                     n_levels::Int      = 3,
                     dx                 = get_step(mapS_vec[1].xx),
                     dy                 = get_step(mapS_vec[1].yy),
                     xx_lim::Tuple      = get_lim(mapS_vec[1].xx,0.5),
                     yy_lim::Tuple      = get_lim(mapS_vec[1].yy,0.5),
                     α                  = 200,
                     use_fallback::Bool = true)

    @assert eltype(mapS_vec) <: MapS "only MapS allowed"

    # sort maps by altitude
    mapS_alt = [mapS.alt for mapS in mapS_vec]
    mapS_vec = mapS_vec[sortperm(mapS_alt)]
    mapS_alt = [mapS.alt for mapS in mapS_vec]
    alt_lev  = [LinRange(mapS_alt[1],mapS_alt[end],n_levels);]

    # resample grids to match
    map_xx   = [xx_lim[1]:dx:xx_lim[2]+dx-eps(float(xx_lim[2]));]
    map_yy   = [yy_lim[1]:dy:yy_lim[2]+dy-eps(float(yy_lim[2]));]
    mapS_vec = [map_resample(mapS,map_xx,map_yy) for mapS in mapS_vec]

    if use_fallback # fill with fallback map
        mapS_vec = [map_combine(mapS,mapS_fallback;
                    xx_lim=xx_lim,yy_lim=yy_lim,α=α) for mapS in mapS_vec]
    else # fill with knn
        mapS_vec = [map_fill(mapS) for mapS in mapS_vec]
    end

    map_xx  = mapS_vec[1].xx
    map_yy  = mapS_vec[1].yy
    map_map = zeros(eltype(mapS_vec[1].map),
                    length(map_yy),length(map_xx),length(alt_lev))

    # get map at each level with upward continuation as necessary
    for (i,alt) in enumerate(alt_lev)
        j = findfirst(alt .≈ mapS_alt)
        if j !== nothing
            map_map[:,:,i] = mapS_vec[j].map # use map directly if a map altitude is specified
        else
            k = findlast(alt .> mapS_alt)
            map_map[:,:,i] = upward_fft(mapS_vec[k],alt).map # use first map below & upward continue
        end
    end

    return MapS3D(map_map, map_xx, map_yy, alt_lev)
end # function map_combine
