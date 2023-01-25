"""
    create_TL_A(Bx, By, Bz;
                Bt       = sqrt.(Bx.^2+By.^2+Bz.^2),
                terms    = [:permanent,:induced,:eddy],
                Bt_scale = 50000,
                return_B = false)

Create Tolles-Lawson `A` matrix using vector magnetometer measurements. 
Optionally returns the magnitude and derivatives of total field.

**Arguments:**
- `Bx,By,Bz`: vector magnetometer measurements [nT]
- `Bt`:       (optional) magnitude of vector magnetometer measurements or scalar magnetometer measurements for modified Tolles-Lawson [nT]
- `terms`:    (optional) Tolles-Lawson terms to use {`:permanent`,`:induced`,`:eddy`,`:bias`}
- `Bt_scale`: (optional) scaling factor for induced and eddy current terms [nT]
- `return_B`: (optional) if true, also return magnitude and derivatives of `Bt`

**Returns:**
- `A`:     Tolles-Lawson `A` matrix
- `Bt`:    (optional) magnitude of total field measurements [nT]
- `B_dot`: (optional) derivative of total field unit vector [-]
"""
function create_TL_A(Bx, By, Bz;
                     Bt       = sqrt.(Bx.^2+By.^2+Bz.^2),
                     terms    = [:permanent,:induced,:eddy],
                     Bt_scale = 50000,
                     return_B = false)

    Bx_hat = Bx ./ Bt
    By_hat = By ./ Bt
    Bz_hat = Bz ./ Bt

    Bx_dot = fdm(Bx)
    By_dot = fdm(By)
    Bz_dot = fdm(Bz)

    Bx_hat_Bx = Bx_hat .* Bx ./ Bt_scale
    Bx_hat_By = Bx_hat .* By ./ Bt_scale
    Bx_hat_Bz = Bx_hat .* Bz ./ Bt_scale
    By_hat_By = By_hat .* By ./ Bt_scale
    By_hat_Bz = By_hat .* Bz ./ Bt_scale
    Bz_hat_Bz = Bz_hat .* Bz ./ Bt_scale

    Bx_hat_Bx_dot = Bx_hat .* Bx_dot ./ Bt_scale
    Bx_hat_By_dot = Bx_hat .* By_dot ./ Bt_scale
    Bx_hat_Bz_dot = Bx_hat .* Bz_dot ./ Bt_scale
    By_hat_Bx_dot = By_hat .* Bx_dot ./ Bt_scale
    By_hat_By_dot = By_hat .* By_dot ./ Bt_scale
    By_hat_Bz_dot = By_hat .* Bz_dot ./ Bt_scale
    Bz_hat_Bx_dot = Bz_hat .* Bx_dot ./ Bt_scale
    Bz_hat_By_dot = Bz_hat .* By_dot ./ Bt_scale
    Bz_hat_Bz_dot = Bz_hat .* Bz_dot ./ Bt_scale

    A = Array{eltype(Bt)}(undef,size(Bt,1),0)

    # add (3) permanent field terms - all
    if any([:permanent,:p,:permanent3,:p3] .∈ (terms,))
    	A = [A Bx_hat By_hat Bz_hat]
    end

    # add (6) induced field terms - all
    if any([:induced,:i,:induced6,:i6] .∈ (terms,))
        A = [A Bx_hat_Bx Bx_hat_By Bx_hat_Bz By_hat_By By_hat_Bz Bz_hat_Bz]
    end

    # add (5) induced field terms - all except Bz_hat_Bz
    if any([:induced5,:i5] .∈ (terms,))
        A = [A Bx_hat_Bx Bx_hat_By Bx_hat_Bz By_hat_By By_hat_Bz]
    end

    # add (3) induced field terms - Bx_hat_Bx, By_hat_By, Bz_hat_Bz
    if any([:induced3,:i3] .∈ (terms,))
        A = [A Bx_hat_Bx By_hat_By Bz_hat_Bz]
    end

    # add (9) eddy current terms - all
    if any([:eddy,:e,:eddy9,:e9] .∈ (terms,))
        A = [A Bx_hat_Bx_dot Bx_hat_By_dot Bx_hat_Bz_dot]
        A = [A By_hat_Bx_dot By_hat_By_dot By_hat_Bz_dot]
        A = [A Bz_hat_Bx_dot Bz_hat_By_dot Bz_hat_Bz_dot]
    end

    # add (8) eddy current terms - all except Bz_hat_Bz_dot
    if any([:eddy8,:e8] .∈ (terms,))
        A = [A Bx_hat_Bx_dot Bx_hat_By_dot Bx_hat_Bz_dot]
        A = [A By_hat_Bx_dot By_hat_By_dot By_hat_Bz_dot]
        A = [A Bz_hat_Bx_dot Bz_hat_By_dot]
    end

    # add (3) eddy current terms - Bx_hat_Bx_dot, By_hat_By_dot, Bz_hat_Bz_dot
    if any([:eddy3,:e3] .∈ (terms,))
        A = [A Bx_hat_Bx_dot By_hat_By_dot Bz_hat_Bz_dot]
    end

    # add (3) derivative terms - Bx_dot, By_dot, Bz_dot
    if any([:fdm,:f,:d,:fdm3,:f3,:d3] .∈ (terms,))
        A = [A Bx_dot By_dot Bz_dot]
    end

    # add (1) bias term
    if any([:bias,:b] .∈ (terms,))
        A = [A ones(eltype(Bt),size(Bt))]
    end

    if return_B
        B_dot = [Bx_dot By_dot Bz_dot]
        return (A, Bt, B_dot)
    else
        return (A)
    end
end # function create_TL_A

"""
    create_TL_A(flux::MagV, ind=trues(length(flux.x));
                Bt       = sqrt.(flux.x.^2+flux.y.^2+flux.z.^2)[ind],
                terms    = [:permanent,:induced,:eddy],
                Bt_scale = 50000,
                return_B = false)

Create Tolles-Lawson `A` matrix using vector magnetometer measurements. 
Optionally returns the magnitude and derivatives of total field.

**Arguments:**
- `flux`:     `MagV` vector magnetometer measurement struct
- `ind`:      (optional) selected data indices
- `Bt`:       (optional) magnitude of vector magnetometer measurements or scalar magnetometer measurements for modified Tolles-Lawson [nT]
- `terms`:    (optional) Tolles-Lawson terms to use {`:permanent`,`:induced`,`:eddy`,`:bias`}
- `Bt_scale`: (optional) scaling factor for induced and eddy current terms [nT]
- `return_B`: (optional) if true, also return magnitude and derivatives of `Bt`

**Returns:**
- `A`:     Tolles-Lawson `A` matrix
- `Bt`:    (optional) magnitude of total field measurements [nT]
- `B_dot`: (optional) derivative of total field unit vector [-]
"""
function create_TL_A(flux::MagV, ind=trues(length(flux.x));
                     Bt       = sqrt.(flux.x.^2+flux.y.^2+flux.z.^2)[ind],
                     terms    = [:permanent,:induced,:eddy],
                     Bt_scale = 50000,
                     return_B = false)
    length(Bt) != length(flux.x[ind]) && (Bt = Bt[ind])
    create_TL_A(flux.x[ind],flux.y[ind],flux.z[ind];
                Bt=Bt,terms=terms,Bt_scale=Bt_scale,return_B=return_B)
end # function create_TL_A

"""
    create_TL_coef(Bx, By, Bz, B;
                   Bt         = sqrt.(Bx.^2+By.^2+Bz.^2),
                   λ          = 0,
                   terms      = [:permanent,:induced,:eddy],
	               pass1      = 0.1,
                   pass2      = 0.9,
                   fs         = 10.0,
                   pole::Int  = 4,
	               trim::Int  = 20,
                   Bt_scale   = 50000,
                   return_var = false)

Create Tolles-Lawson coefficients using vector and scalar magnetometer 
measurements and a bandpass, low-pass or high-pass filter.

**Arguments:**
- `Bx,By,Bz`:   vector magnetometer measurements [nT]
- `B`:          scalar magnetometer measurements [nT]
- `Bt`:         (optional) magnitude of vector magnetometer measurements or scalar magnetometer measurements for modified Tolles-Lawson [nT]
- `λ`:          (optional) ridge parameter
- `terms`:      (optional) Tolles-Lawson terms to use {`:permanent`,`:induced`,`:eddy`,`:bias`}
- `pass1`:      (optional) first passband frequency [Hz]
- `pass2`:      (optional) second passband frequency [Hz]
- `fs`:         (optional) sampling frequency [Hz]
- `pole`:       (optional) number of poles for Butterworth filter
- `trim`        (optional) number of elements to trim after filtering
- `Bt_scale`:   (optional) scaling factor for induced and eddy current terms [nT]
- `return_var`: (optional) return `B_var` fit error variance along with `coef`

**Returns:**
- `coef`:  Tolles-Lawson coefficients
- `B_var`: if `return_var=true`, also return fit error variance
"""
function create_TL_coef(Bx, By, Bz, B;
                        Bt         = sqrt.(Bx.^2+By.^2+Bz.^2),
                        λ          = 0,
                        terms      = [:permanent,:induced,:eddy],
                        pass1      = 0.1,
                        pass2      = 0.9,
                        fs         = 10.0,
                        pole::Int  = 4,
                        trim::Int  = 20,
                        Bt_scale   = 50000,
                        return_var = false)

    # create filter
    if ((pass1 > 0) & (pass1 < fs/2)) | ((pass2 > 0) & (pass2 < fs/2))
        perform_filter = true # bandpass, low-pass, or high-pass
        bpf = get_bpf(;pass1=pass1,pass2=pass2,fs=fs,pole=pole)
    else
        perform_filter = false # all-pass
        @info("not filtering (or trimming) Tolles-Lawson data")
    end

    # create Tolles-Lawson `A` matrix
    A = create_TL_A(Bx,By,Bz;Bt=Bt,terms=terms,Bt_scale=Bt_scale)

    # filter columns of A (e.g. Bx_hat) & measurements and trim edges
    perform_filter && (A = bpf_data(A;bpf=bpf)[trim+1:end-trim,:])
    perform_filter && (B = bpf_data(B;bpf=bpf)[trim+1:end-trim,:])

    # linear regression to get Tolles-Lawson coefficients
    coef = vec(linreg(B,A;λ=λ))

    if return_var
        B_var = var(B - A*coef)
        @info("TL fit error variance: $(B_var)")
        return (coef, B_var)
    else
        return (coef)
    end
end # function create_TL_coef

"""
    create_TL_coef(flux::MagV, B, ind=trues(length(flux.x));
                   Bt         = sqrt.(flux.x.^2+flux.y.^2+flux.z.^2)[ind],
                   λ          = 0,
                   terms      = [:permanent,:induced,:eddy],
                   pass1      = 0.1,
                   pass2      = 0.9,
                   fs         = 10.0,
                   pole::Int  = 4,
                   trim::Int  = 20,
                   Bt_scale   = 50000,
                   return_var = false)

Create Tolles-Lawson coefficients using vector and scalar magnetometer 
measurements and a bandpass, low-pass or high-pass filter.

**Arguments:**
- `flux`:       `MagV` vector magnetometer measurement struct
- `B`:          scalar magnetometer measurements [nT]
- `ind`:        (optional) selected data indices
- `Bt`:         (optional) magnitude of vector magnetometer measurements or scalar magnetometer measurements for modified Tolles-Lawson [nT]
- `λ`:          (optional) ridge parameter
- `terms`:      (optional) Tolles-Lawson terms to use {`:permanent`,`:induced`,`:eddy`,`:bias`}
- `pass1`:      (optional) first passband frequency [Hz]
- `pass2`:      (optional) second passband frequency [Hz]
- `fs`:         (optional) sampling frequency [Hz]
- `pole`:       (optional) number of poles for Butterworth filter
- `trim`        (optional) number of elements to trim after filtering
- `Bt_scale`:   (optional) scaling factor for induced and eddy current terms [nT]
- `return_var`: (optional) return `B_var` fit error variance along with `coef`

**Returns:**
- `coef`:  Tolles-Lawson coefficients
- `B_var`: fit error variance
"""
function create_TL_coef(flux::MagV, B, ind=trues(length(flux.x));
                        Bt         = sqrt.(flux.x.^2+flux.y.^2+flux.z.^2)[ind],
                        λ          = 0,
                        terms      = [:permanent,:induced,:eddy],
                        pass1      = 0.1,
                        pass2      = 0.9,
                        fs         = 10.0,
                        pole::Int  = 4,
                        trim::Int  = 20,
                        Bt_scale   = 50000,
                        return_var = false)
    length(Bt) != length(flux.x[ind]) && (Bt = Bt[ind])
    create_TL_coef(flux.x[ind],flux.y[ind],flux.z[ind],B[ind];
                   Bt=Bt,λ=λ,terms=terms,pass1=pass1,pass2=pass2,fs=fs,
                   pole=pole,trim=trim,Bt_scale=Bt_scale,return_var=return_var)
end # function create_TL_coef

"""
    fdm(x::Vector; central::Bool=true, fourth::Bool=false)

Finite difference method (FDM) on vector of input data.

**Arguments:**
- `x`:       input data
- `central`: (optional) if true, 1st-order central difference, otherwise backward
- `fourth`:  (optional) if true, 4th-order central difference

**Returns:**
- `dif`: length of `x` finite differences
"""
function fdm(x::Vector; central::Bool=true, fourth::Bool=false)

    if fourth & (length(x) > 4)
        dif_1   = zeros(eltype(x),2)
        dif_end = zeros(eltype(x),2)
        dif_mid = (   x[1:end-4] + 
                   -4*x[2:end-3] + 
                    6*x[3:end-2] + 
                   -4*x[4:end-1] + 
                      x[5:end  ] ) ./ 16 # divided by dx^4
    elseif central & (length(x) > 2)
        dif_1   =  x[2]     - x[1]
        dif_end =  x[end]   - x[end-1]
        dif_mid = (x[3:end] - x[1:end-2]) ./ 2
    elseif (length(x) > 1)
        dif_1   =  x[2]       - x[1]
        dif_end =  x[end]     - x[end-1]
        dif_mid = (x[2:end-1] - x[1:end-2])
    else
        return zero(x)
    end

    dif = [dif_1; dif_mid; dif_end]

    return (dif)
end # function fdm
