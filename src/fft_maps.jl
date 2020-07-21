# upward continuation functions for shifting magnetic anomaly maps

function upward_fft(map_in,dx,dy,dz)
#   upward continuation function for shifting a magnetic anomaly map
#   map_in  gridded magnetic anomaly map [nT]
#   dx      x direction (longitude) map spacing [m]
#   dy      y direction (latitude)  map spacing [m]

    (Ny,Nx) = size(map_in)
    (k,~,~) = create_K(dx,dy,Nx,Ny)
    H       = exp.(-k.*dz)
    map_out = real(ifft(fft(map_in).*H))

    return (map_out)
end # function upward_fft

function create_K(dx,dy,Nx,Ny)

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

    return (k,kx,ky)
end # function create_K
