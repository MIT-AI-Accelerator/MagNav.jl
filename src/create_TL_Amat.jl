function create_TL_Amat(Bx,By,Bz)
#   create Tolles-Lawson A matrix

    Bt       = sqrt.(Bx.^2+By.^2+Bz.^2)
    BtMean   = max(mean(Bt),1e-6)

    cosX     = Bx ./ Bt
    cosY     = By ./ Bt
    cosZ     = Bz ./ Bt

    cosX_dot = central_fdm(cosX)
    cosY_dot = central_fdm(cosY)
    cosZ_dot = central_fdm(cosZ)

    cosXX    = Bt .* cosX.*cosX ./ BtMean
    cosXY    = Bt .* cosX.*cosY ./ BtMean
    cosXZ    = Bt .* cosX.*cosZ ./ BtMean
    cosYY    = Bt .* cosY.*cosY ./ BtMean
    cosYZ    = Bt .* cosY.*cosZ ./ BtMean
    cosZZ    = Bt .* cosZ.*cosZ ./ BtMean

    cosXcosX_dot = Bt .* cosX.*cosX_dot ./ BtMean
    cosXcosY_dot = Bt .* cosX.*cosY_dot ./ BtMean
    cosXcosZ_dot = Bt .* cosX.*cosZ_dot ./ BtMean
    cosYcosX_dot = Bt .* cosY.*cosX_dot ./ BtMean
    cosYcosY_dot = Bt .* cosY.*cosY_dot ./ BtMean
    cosYcosZ_dot = Bt .* cosY.*cosZ_dot ./ BtMean
    cosZcosX_dot = Bt .* cosZ.*cosX_dot ./ BtMean
    cosZcosY_dot = Bt .* cosZ.*cosY_dot ./ BtMean
    cosZcosZ_dot = Bt .* cosZ.*cosZ_dot ./ BtMean

    # add permanent field terms
    A = [cosX cosY cosZ]

    # add induced field terms
    A = [A cosXX cosXY cosXZ cosYY cosYZ cosZZ]

    # add eddy current terms
    A = [A cosXcosX_dot cosXcosY_dot cosXcosZ_dot]
    A = [A cosYcosX_dot cosYcosY_dot cosYcosZ_dot]
    A = [A cosZcosX_dot cosZcosY_dot cosZcosZ_dot]

    return (A)
end # function create_TL_Amat

function central_fdm(x)
    val          = zeros(length(x))
    val[1]       = x[2]   - x[1]
    val[end]     = x[end] - x[end-1]
    val[2:end-1] = (x[3:end] - x[1:end-2]) ./ 2
    return (val)
end # function central_fdm
