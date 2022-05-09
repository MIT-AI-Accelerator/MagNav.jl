"""
    upward_fft(map_map, dx, dy, dz; expand::Bool=true, α=0)

Upward continuation of a potential field (i.e. magnetic anomaly field) map. 
Uses the Fast Fourier Transform to convert the map to the frequency domain, 
applies an upward continuation filter, and converts back to the spatial domain. 
Optionally expands the map temporarily with periodic padding. Downward 
continuation can be done to a limited degree as well, but be careful as this 
can be unstable and amplify high frequencies (i.e. noise).

Reference: Blakely, Potential Theory in Gravity and Magnetic Applications, 
2009, Chapter 12 & Appendix B (pg. 315-317 & 402).

**Arguments**
- `map_map`: `ny` x `nx` 2D gridded map data
- `dx:`      x-direction map sample interval [m]
- `dy`:      y-direction map sample interval [m]
- `dz`:      z-direction upward continuation distance [m]
- `expand`:  (optional) if true, expand map temporarily to limit edge effects
- `α`:       (optional) regularization parameter for downward continuation

**Returns**
- `map_out`: `ny` x `nx` upward continued 2D gridded map data
"""
function upward_fft(map_map, dx, dy, dz; expand::Bool=true, α=0)
    dz > 0 && (α = 0) # ensure no regularization if upward

    if expand == true
        (ny,nx)  = size(map_map)
        pad      = maximum(ceil.(Int,10*abs(dz)./(dx,dy))) # set pad > 10*dz
        (map_temp,px,py) = map_expand(map_map,pad) # expand with pad
        (Ny,Nx)  = size(map_temp)
        (k,_,_)  = create_k(dx,dy,Nx,Ny) # radial wavenumber grid
        H        = exp.(-k.*dz) ./ (1 .+ α .* k.^2 .* exp.(-k.*dz)) # filter
        map_out  = real(ifft(fft(map_temp).*H))[(1:ny).+py,(1:nx).+px]
    else
        (ny,nx)  = size(map_map)
        (k,_,_)  = create_k(dx,dy,nx,ny) # radial wavenumber grid
        H        = exp.(-k.*dz) ./ (1 .+ α .* k.^2 .* exp.(-k.*dz)) # filter
        map_out  = real(ifft(fft(map_map).*H))
    end

    return (map_out)
end # function upward_fft

"""
    upward_fft(map_map::Union{MapS,MapV}, alt; expand::Bool=true, α=0)

**Arguments**
- `map_map`: `MapS` scalar or `MapV` vector magnetic anomaly map struct
- `alt`:     target upward continuation altitude [m]
- `expand`:  (optional) if true, expand map temporarily to limit edge effects
- `α`:       (optional) regularization parameter for downward continuation

**Returns**
- `map_out`: `MapS` scalar or `MapV` vector magnetic anomaly map struct with upward continued map
"""
function upward_fft(map_map::Union{MapS,MapV}, alt; expand::Bool=true, α=0)
    alt = convert(eltype(map_map.xx),alt)
    if (alt >= map_map.alt) | (α > 0)
        dlon = abs(map_map.xx[end]-map_map.xx[1])/(length(map_map.xx)-1)
        dlat = abs(map_map.yy[end]-map_map.yy[1])/(length(map_map.yy)-1)
        dx   = dlon2de(dlon,mean(map_map.yy))
        dy   = dlat2dn(dlat,mean(map_map.yy))
        dz   = alt-map_map.alt
        if typeof(map_map) <: MapS # scalar map
            map_out = MapS(upward_fft(map_map.map,dx,dy,dz,expand=expand,α=α),
                           map_map.xx,map_map.yy,convert(eltype(map_map),alt))
        else # vector map
            mapX    = upward_fft(map_map.mapX,dx,dy,dz,expand=expand,α=α)
            mapY    = upward_fft(map_map.mapY,dx,dy,dz,expand=expand,α=α)
            mapZ    = upward_fft(map_map.mapZ,dx,dy,dz,expand=expand,α=α)
            map_out = MapV(mapX,mapY,mapZ,map_map.xx,map_map.yy,alt)
        end
        return (map_out)
    else
        @info("for downward continuation, α must be specified")
        return (map_map)
    end
end # function upward_fft

"""
    downward_L(mapS::Union{MapS,MapSd}, alt, α::Vector;
               expand::Bool = true,
               ind          = map_params(mapS)[2])

Downward continuation using a sequence of regularization parameters to create 
a characteristic L-curve. The optimal regularization parameter is at a local
minimum on the L-curve, which is a local maximum of curvature. The global 
maximum of curvature may or may not be the optimal regularization parameter.

**Arguments:**
- `mapS`:   `MapS` or `MapSd` scalar magnetic anomaly map struct
- `alt`:    target downward continuation altitude [m]
- `α`:      (geometric) sequence of regularization parameters
- `expand`: (optional) if true, expand map temporarily to limit edge effects
- `ind`:    (optional) selected map indices (e.g. non-edge data)

**Returns:**
- `norms`: L-infinity norm of difference between sequential D.C. solutions
"""
function downward_L(mapS::Union{MapS,MapSd}, alt, α::Vector;
                    expand::Bool = true,
                    ind          = map_params(mapS)[2])
    norms   = zeros(length(α)-1)
    map_map = deepcopy(mapS.map)
    dlon    = abs(mapS.xx[end]-mapS.xx[1])/(length(mapS.xx)-1)
    dlat    = abs(mapS.yy[end]-mapS.yy[1])/(length(mapS.yy)-1)
    dx      = dlon2de(dlon,mean(mapS.yy))
    dy      = dlat2dn(dlat,mean(mapS.yy))
    dz      = length(mapS.alt) > 1 ? alt - median(mapS.alt[ind]) : alt - mapS.alt
    if expand == true
        (ny,nx)  = size(map_map)
        pad      = maximum(ceil.(Int,10*abs(dz)./(dx,dy))) # set pad > 10*dz
        (map_temp,px,py) = map_expand(map_map,pad) # expand with pad
        (Ny,Nx)  = size(map_temp)
        (k,_,_)  = create_k(dx,dy,Nx,Ny) # radial wavenumber grid
        H_temp   = exp.(-k.*dz)
        H        = H_temp ./ (1 .+ α[1] .* k.^2 .* H_temp) # filter
        map_old  = real(ifft(fft(map_temp).*H))
        map_old  = map_old[(1:ny).+py,(1:nx).+px][ind]
        for i = 2:length(α)
            H       = H_temp ./ (1 .+ α[i] .* k.^2 .* H_temp) # filter
            map_new = real(ifft(fft(map_temp).*H))
            map_new = map_new[(1:ny).+py,(1:nx).+px][ind]
            norms[i-1] = norm(map_new-map_old,Inf)
            map_old = map_new
        end
    else
        (ny,nx)  = size(map_map)
        (k,_,_)  = create_k(dx,dy,nx,ny) # radial wavenumber grid
        H_temp   = exp.(-k.*dz)
        H        = H_temp ./ (1 .+ α[1] .* k.^2 .* H_temp) # filter
        map_old  = real(ifft(fft(map_map).*H))[ind]
        for i = 2:length(α)
            H       = H_temp ./ (1 .+ α[i] .* k.^2 .* H_temp) # filter
            map_new = real(ifft(fft(map_map).*H))[ind]
            norms[i-1] = norm(map_new-map_old,Inf)
            map_old = map_new
        end
    end

    return (norms)
end # function downward_L

"""
    vector_fft(map_map, dx, dy, D, I)

Get potential field (i.e. magnetic anomaly field) map vector components 
using declination and inclination.

**Arguments:**
- `map_map`: `ny` x `nx` 2D gridded map data
- `dx`: x-direction map sample interval [m]
- `dy`: y-direction map sample interval [m]
- `D`:  map declination (earth core field)
- `I`:  map inclination (earth core field)

**Returns:**
- `Bx,By,Bz`: map vector components
"""
function vector_fft(map_map, dx, dy, D, I)
    (ny,nx) = size(map_map)
    (s,u,v) = create_k(dx,dy,nx,ny)

    l = cosd(I)*cosd(D)
    m = cosd(I)*sind(D)
    n = sind(I)

    F = fft(map_map)

    Hx = im*u ./ (im*(u*l+m*v)+n*s)
    Hy = im*v ./ (im*(u*l+m*v)+n*s)
    Hz = s    ./ (im*(u*l+m*v)+n*s)

    Hx[1,1] = 1
    Hy[1,1] = 1
    Hz[1,1] = 1

    Bx = real(ifft(Hx.*F))
    By = real(ifft(Hy.*F))
    Bz = real(ifft(Hz.*F))

    return (Bx, By, Bz)
end # function vector_fft

"""
    create_k(dx, dy, nx::Int, ny::Int)

Create radial wavenumber (spatial frequency) grid.

**Arguments:**
- `dx`: x-direction map sample interval [m]
- `dy`: y-direction map sample interval [m]
- `nx`: x-direction map dimension [-]
- `ny`: y-direction map dimension [-]

**Returns:**
- `k`:  `ny` x `nx` radial wavenumber (i.e. magnitude of wave vector)
- `kx`: `ny` x `nx` x-direction radial wavenumber
- `ky`: `ny` x `nx` y-direction radial wavenumber
"""
function create_k(dx, dy, nx::Int, ny::Int)
    # DFT sample frequencies [rad/m], 1/dx & 1/dy are sampling rates [1/m]
    kx = nx*dx==0 ? zeros(ny,nx) : repeat(2*pi*fftfreq(nx,1/dx)',ny,1)
    ky = ny*dy==0 ? zeros(ny,nx) : repeat(2*pi*fftfreq(ny,1/dy) ,1,nx)
    k  = sqrt.(kx.^2+ky.^2)
    return (k, kx, ky)
end # function create_k

"""
    psd(map_map, dx, dy)

Power spectral density of a potential field (i.e. magnetic anomaly field) map. 
Uses the Fast Fourier Transform to determine the spectral energy distribution 
across the radial wavenumbers (spatial frequencies) in the Fourier transform.

**Arguments**
- `map_map`: `ny` x `nx` 2D gridded map data
- `dx`: x-direction map sample interval [m]
- `dy`: y-direction map sample interval [m]

**Returns:**
- `map_psd`: `ny` x `nx` power spectral density of 2D gridded map data
- `kx`: `ny` x `nx` x-direction radial wavenumber
- `ky`: `ny` x `nx` y-direction radial wavenumber
"""
function psd(map_map, dx, dy)
    (ny,nx)   = size(map_map)
    (_,kx,ky) = create_k(dx,dy,nx,ny)
    map_psd   = abs.(fft(map_map)).^2
    return (map_psd, kx, ky)
end # function psd

"""
    map_expand(map_map, pad::Int=1)

Expand map with padding on each edge to eliminate discontinuities in 
discrete Fourier transform. Map is “wrapped around” to make it periodic.

**Arguments**
- `map_map`: `ny` x `nx` 2D gridded map data
- `pad`: minimum padding along map edges

**Returns:**
- `map_out`: `ny` x `nx` 2D gridded map data with padding
- `padx`: x-direction padding (on first edge)
- `pady`: y-direction padding (on first edge)
"""
function map_expand(map_map, pad::Int=1)

    # original map size
    (ny,nx) = size(map_map)

    # padded map size, faster with smooth number length dimensions
    # (Ny,Nx) = (ny,nx) .+ 2*pad
    # (Ny,Nx) = 2 .^ ceil.(Int, log.(2,(ny,nx).+2*pad) )
    (Ny,Nx) = smooth7.((ny,nx).+2*pad)

    # padding on each edge
    pady    = (floor(Int,(Ny-ny)/2),ceil(Int,(Ny-ny)/2))
    padx    = (floor(Int,(Nx-nx)/2),ceil(Int,(Nx-nx)/2))

    # place original map in middle of new map
    (y1,y2) = (1,ny) .+ pady[1]
    (x1,x2) = (1,nx) .+ padx[1]
    map_out = zeros(Ny,Nx)
    map_out[y1:y2,x1:x2] = map_map

    # fill row edges (right/left)
    for i = y1:y2
        vals = LinRange(map_out[i,x1],map_out[i,x2],Nx-nx+2)[2:end-1]
        map_out[i,1:x1-1  ] = reverse(vals[1:1:padx[1]])
        map_out[i,x2+1:end] = reverse(vals[(1:padx[2]).+padx[1]])
    end

    # fill column edges (top/bottom)
    for i = 1:Nx
        vals = LinRange(map_out[y1,i],map_out[y2,i],Ny-ny+2)[2:end-1]
        map_out[1:y1-1  ,i] = reverse(vals[1:1:pady[1]])
        map_out[y2+1:end,i] = reverse(vals[(1:pady[2]).+pady[1]])
    end

    return (map_out, padx[1], pady[1])
end # function map_expand

"""
    smooth7(x::Int)

Find the lowest 7-smooth number `y` >= input number `x`.

"""
function smooth7(x::Int)
    y = 2*x
    for i = 0:ceil(Int,log(7,x))
        for j = 0:ceil(Int,log(5,x))
            for k = 0:ceil(Int,log(3,x))
                z = 7^i*5^j*3^k
                z < 2*x && (y = min(y, 2^ceil(Int,log(2,x/z))*z))
            end
        end
    end
    return (y)
end # function smooth7

# 1.820 s (100 allocations: 1.67 GiB) # no pad
# 2.318 s (104 allocations: 2.21 GiB) # pad
# 4.175 s (105 allocations: 4.60 GiB) # 2
# 6.470 s (105 allocations: 5.87 GiB) # 3
# 2.769 s (104 allocations: 3.70 GiB) # 2 | 3
# 1.445 s (113 allocations: 2.38 GiB) # 2 & 3
# 1.310 s (104 allocations: 2.27 GiB) # smooth-7
