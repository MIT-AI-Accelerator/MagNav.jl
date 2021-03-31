"""
    create_TL_A(Bx, By, Bz; terms=["permanent","induced","eddy"])

Create Tolles-Lawson `A` matrix using vector magnetometer measurements.

**Arguments:**
- `Bx, By, Bz`: vector magnetometer measurements
- `terms`: (optional) Tolles-Lawson terms to use {"permanent","induced","eddy"}

**Returns:**
- `A`: Tolles-Lawson `A` matrix
"""
function create_TL_A(Bx, By, Bz; terms=["permanent","induced","eddy"])

    Bt       = sqrt.(Bx.^2+By.^2+Bz.^2)
    Bt_scale = mean(Bt) # 50000

    cosX     = Bx ./ Bt
    cosY     = By ./ Bt
    cosZ     = Bz ./ Bt

    cosX_dot = fdm(cosX)
    cosY_dot = fdm(cosY)
    cosZ_dot = fdm(cosZ)

    cosXX    = Bt .* cosX.*cosX ./ Bt_scale
    cosXY    = Bt .* cosX.*cosY ./ Bt_scale
    cosXZ    = Bt .* cosX.*cosZ ./ Bt_scale
    cosYY    = Bt .* cosY.*cosY ./ Bt_scale
    cosYZ    = Bt .* cosY.*cosZ ./ Bt_scale
    cosZZ    = Bt .* cosZ.*cosZ ./ Bt_scale

    cosXcosX_dot = Bt .* cosX.*cosX_dot ./ Bt_scale
    cosXcosY_dot = Bt .* cosX.*cosY_dot ./ Bt_scale
    cosXcosZ_dot = Bt .* cosX.*cosZ_dot ./ Bt_scale
    cosYcosX_dot = Bt .* cosY.*cosX_dot ./ Bt_scale
    cosYcosY_dot = Bt .* cosY.*cosY_dot ./ Bt_scale
    cosYcosZ_dot = Bt .* cosY.*cosZ_dot ./ Bt_scale
    cosZcosX_dot = Bt .* cosZ.*cosX_dot ./ Bt_scale
    cosZcosY_dot = Bt .* cosZ.*cosY_dot ./ Bt_scale
    cosZcosZ_dot = Bt .* cosZ.*cosZ_dot ./ Bt_scale

    A = Array{Float64}(undef,size(Bt,1),0)
    
    # add (3) permanent field terms
    if "permanent" in terms
        A = [A cosX cosY cosZ]
    end

    # add (6) induced field terms
    if "induced" in terms
        A = [A cosXX cosXY cosXZ cosYY cosYZ cosZZ]
    end

    # add (9) eddy current terms
    if "eddy" in terms
        A = [A cosXcosX_dot cosXcosY_dot cosXcosZ_dot]
        A = [A cosYcosX_dot cosYcosY_dot cosYcosZ_dot]
        A = [A cosZcosX_dot cosZcosY_dot cosZcosZ_dot]
	    # A = [A ones(length(Bt))]
    end

    return (A)
end # function create_TL_A

"""
    create_TL_coef(Bx, By, Bz, meas;
                   pass1=0.1, pass2=0.9, fs=10.0,
                   terms=["permanent","induced","eddy"])

Create Tolles-Lawson coefficients using vector and scalar magnetometer 
measurements and a bandpass, low-pass or high-pass filter.

**Arguments:**
- `Bx, By, Bz`: vector magnetometer measurements
- `meas`: scalar magnetometer measurements
- `pass1`: (optional) first passband frequency [Hz]
- `pass2`: (optional) second passband frequency [Hz]
- `fs`: (optional) sampling frequency [Hz]
- `terms`: (optional) Tolles-Lawson terms to use {"permanent","induced","eddy"}

**Returns:**
- `coef`: Tolles-Lawson coefficients
"""
function create_TL_coef(Bx, By, Bz, meas;
                        pass1=0.1, pass2=0.9, fs=10.0,
                        terms=["permanent","induced","eddy"])

    # create filter
    perform_filter = true
    if (pass1 > 0) & (pass2 < fs/2)
        # bandpass
        d = digitalfilter(Bandpass(pass1,pass2;fs=fs),Butterworth(4))
    elseif ((pass1 <= 0) | (pass1 >= fs/2)) & (pass2 < fs/2)
        # low-pass
        d = digitalfilter(Lowpass(pass2;fs=fs),Butterworth(4))
    elseif (pass1 > 0) & ((pass2 <= 0) | (pass2 >= fs/2))
        # high-pass
        d = digitalfilter(Highpass(pass1;fs=fs),Butterworth(4))
    else
        # all-pass
        perform_filter = false
    end

    # filter measurements
    if perform_filter
        meas_f = filtfilt(d,meas)
    else
        meas_f = meas
    end

    # create Tolles-Lawson A matrix
    A = create_TL_A(Bx,By,Bz;terms=terms)

    # filter each column of A (e.g. cosX)
    A_f = deepcopy(A)
    if perform_filter
        for i = 1:size(A,2)
            # std(A[:,i]) != 0.0 ? A_f[:,i] = filtfilt(d,A[:,i]) : nothing
            A_f[:,i] = filtfilt(d,A[:,i])
        end
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
