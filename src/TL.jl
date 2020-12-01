"""
    create_TL_A(Bx, By, Bz)

Create Tolles-Lawson A matrix using vector magnetometer measurements.

**Arguments:**
- `Bx, By, Bz`: vector magnetometer measurements

**Returns:**
- `A`: Tolles-Lawson A matrix
"""
function create_TL_A(Bx, By, Bz)

    Bt       = sqrt.(Bx.^2+By.^2+Bz.^2)
    BtMean   = max(mean(Bt),1e-6)

    cosX     = Bx ./ Bt
    cosY     = By ./ Bt
    cosZ     = Bz ./ Bt

    cosX_dot = fdm(cosX)
    cosY_dot = fdm(cosY)
    cosZ_dot = fdm(cosZ)

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
end # function create_TL_A

"""
    create_TL_coef(Bx, By, Bz, meas; pass1=0.1, pass2=0.9, fs=10.0)

Create Tolles-Lawson coefficients using vector and scalar magnetometer measurements and bandpass filter.

**Arguments:**
- `Bx, By, Bz`: vector magnetometer measurements
- `meas`: scalar magnetometer measurements
- `pass1`: (optional) first passband frequency [Hz]
- `pass2`: (optional) second passband frequency [Hz]
- `fs`: (optional) sampling frequency [Hz]

**Returns:**
- `coef`: Tolles-Lawson coefficients
"""
function create_TL_coef(Bx, By, Bz, meas; pass1=0.1, pass2=0.9, fs=10.0)

    # create filter
    d = digitalfilter(Bandpass(pass1,pass2;fs=fs),Butterworth(4))

    # filter measurements
    meas_f = filtfilt(d,meas)

    # create Tolles-Lawson A matrix
    A = create_TL_A(Bx,By,Bz)

    # filter each column of A (e.g. cosX)
    A_f = deepcopy(A)
    for i = 1:size(A,2)
        A_f[:,i] = filtfilt(d,A[:,i])
    end

    # all filters create artifacts so trim off first/last 20 elements
    trim     = 20
    A_f_t    = A_f[trim+1:end-trim,:]
    meas_f_t = meas_f[trim+1:end-trim]

    # get Tolles-Lawson coefficients
    coef = (A_f_t'*A_f_t) \ (A_f_t'*meas_f_t)

    return (coef)
end # function create_TL_coef

"""
    fdm(x; central=true)

Finite difference method for vector of data.

**Arguments:**
- `x`: vector of data
- `central`: true for central, false for backward

**Returns:**
- `dif`: vector of finite differences, length of `x`
"""
function fdm(x; central=true)

    dif = zero(x)

    if central
        dif[1]       =  x[2]     - x[1]
        dif[end]     =  x[end]   - x[end-1]
        dif[2:end-1] = (x[3:end] - x[1:end-2]) ./ 2
    else
        dif[2:end]   = (x[2:end] - x[1:end-1])
    end

    return (dif)
end # function fdm
