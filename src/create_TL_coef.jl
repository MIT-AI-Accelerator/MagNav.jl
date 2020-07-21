function create_TL_coef(Bx,By,Bz,meas;kwargs...)
#   create Tolles-Lawson coefficients
    defaults = (; pass1 = 0.1,
                  pass2 = 0.9,
                  fs    = 10.0)
    settings = merge(defaults,kwargs)
    pass1 = settings.pass1
    pass2 = settings.pass2
    fs    = settings.fs

    # create filter
    d = digitalfilter(Bandpass(pass1,pass2;fs=fs),Butterworth(4))

    # filter measurements
    meas_f = filtfilt(d,meas)

    # create Tolles-Lawson A matrix
    A = create_TL_Amat(Bx,By,Bz)

    # filter each column of A (e.g. cosX)
    A_f = A
    for i = 1:size(A,2)
        A_f[:,i] = filtfilt(d,A[:,i])
    end

    # all filters create artifacts so trim off first/last 20 elements
    trim     = 20
    A_f_t    = A_f[trim+1:end-trim,:]
    meas_f_t = meas_f[trim+1:end-trim]

    # get Tolles-Lawson coefficients
    TL_coef = (A_f_t'*A_f_t) \ (A_f_t'*meas_f_t)

    return (TL_coef)
end # function create_TL_coef
