# upward continuation functions for shifting magnetic anomaly maps

"""
    upward_fft(map_in, dx, dy, dz)

Upward continuation function for shifting a magnetic anomaly map.

**Arguments:**
- `map_in: gridded magnetic anomaly map [nT]
- `dx`: x direction (longitude) map spacing [m]
- `dy`: y direction (latitude)  map spacing [m]
- `dz`: z direction upward continuation distance [m]

**Returns:**
- `map_out`: upward continued gridded magnetic anomaly map [nT]
"""
function upward_fft(map_in, dx, dy, dz) 
    (Ny,Nx) = size(map_in)
    (k,~,~) = create_K(dx,dy,Nx,Ny)
    H       = exp.(-k.*dz)
    map_out = real(ifft(fft(map_in).*H))
    return (map_out)
end # function upward_fft

"""
    vector_fft(map_in, dx, dy, D, I)

Get magnetic anomaly map vector components using declination and inclination.

**Arguments:**
- `map_in: gridded magnetic anomaly map [nT]
- `dx`: x direction (longitude) map spacing [m]
- `dy`: y direction (latitude)  map spacing [m]
- `D`: map declination (earth core field)
- `I`: map inclination (earth core field)

**Returns:**
- `Bx, By, Bz`: magnetic anomaly map vector components [nT]
"""
function vector_fft(map_in, dx, dy, D, I)
    (Ny,Nx) = size(map_in)
    (s,u,v) = create_K(dx,dy,Nx,Ny)

    l = cosd(I)*cosd(D)
    m = cosd(I)*sind(D)
    n = sind(I)

    F = fft(map_in)

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

function create_K(dx, dy, Nx, Ny)

    kx   = (-Nx+mod(Nx,2))/2:1:(Nx+mod(Nx,2))/2-1
    ky   = (-Ny+mod(Ny,2))/2:1:(Ny+mod(Ny,2))/2-1
    kx_m = repeat(kx',length(ky),1)
    ky_m = repeat(ky,1,length(kx))
    dkx  = 2*pi / (Nx*dx)

    if dy != 0
        dky = 2*pi / (Ny*dy)
    else
        dky = 0
    end

    kx_m = kx_m*dkx
    ky_m = ky_m*dky
    k    = sqrt.(kx_m.^2+ky_m.^2)
    k    = ifftshift(k)
    kx   = ifftshift(kx_m)
    ky   = ifftshift(ky_m)

    return (k, kx, ky)
end # function create_K
