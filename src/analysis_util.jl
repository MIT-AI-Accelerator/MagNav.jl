"""
    dn2dlat(dn, lat)

Convert north-south position (northing) difference to latitude difference.

**Arguments:**
- `dn`: north-south position (northing) difference [m]
- `lat`: nominal latitude [rad]

**Returns:**
- `dlat`: latitude difference [rad]
"""
function dn2dlat(dn, lat)
    dlat = dn * sqrt(1-(e_earth*sin(lat))^2) / r_earth
    return (dlat)
end # function dn2dlat

"""
    de2dlon(de, lat)

Convert east-west position (easting) difference to longitude difference.

**Arguments:**
- `de`: east-west position (easting) difference [m]
- `lat`: nominal latitude [rad]

**Returns:**
- `dlon`: longitude difference [rad]
"""
function de2dlon(de, lat)
    dlon = de * sqrt(1-(e_earth*sin(lat))^2) / r_earth / cos(lat)
    return (dlon)
end # function de2dlon

"""
    dlat2dn(dlat, lat)

Convert latitude difference to north-south position (northing) difference.

**Arguments:**
- `dlat`: latitude difference [rad]
- `lat`: nominal latitude [rad]

**Returns:**
- `dn`: north-south position (northing) difference [m]
"""
function dlat2dn(dlat, lat)
    dn = dlat / sqrt(1-(e_earth*sin(lat))^2) * r_earth
    return (dn)
end # function dlat2dn

"""
    dlon2de(dlon, lat)

Convert longitude difference to east-west position (easting) difference.

**Arguments:**
- `dlon`: longitude difference [rad]
- `lat`: nominal latitude [rad]

**Returns:**
- `de`: east-west position (easting) difference [m]
"""
function dlon2de(dlon, lat)
    de = dlon / sqrt(1-(e_earth*sin(lat))^2) * r_earth * cos(lat)
    return (de)
end # function dlon2de

"""
    linreg(y, x; λ=0)

Linear regression with input data matrix.

**Arguments:**
- `y`: observed data
- `x`: input data
- `λ`: (optional) ridge parameter

**Returns:**
- `coef`: linear regression coefficients
"""
function linreg(y, x; λ=0)
    coef = (x'*x + λ*I) \ (x'*y)
    length(coef) > 1 && (coef = vec(coef))
    return (coef)
end # function linreg

"""
    linreg(y; λ=0)

Linear regression to determine best fit line for x = eachindex(y).

**Arguments:**
- `y`: observed data
- `λ`: (optional) ridge parameter

**Returns:**
- `coef`: (2) linear regression coefficients
"""
function linreg(y; λ=0)
    x = [one.(y) eachindex(y)]
    coef = linreg(y,x;λ=λ)
    return (coef)
end # function linreg

"""
    detrend(y, x=[eachindex(y);]; mean_only::Bool=false)

Detrend signal (remove mean and optionally slope).

**Arguments:**
- `y`: observed data
- `x`: (optional) input data
- `mean_only`: (optional) if true, only remove mean (not slope)

**Returns:**
- `y_new`: detrended observed data
"""
function detrend(y, x=[eachindex(y);]; mean_only::Bool=false)
    if mean_only
        y_new = y .- mean(y)
    else
        x = [one.(y) x]
        coef = (x'*x) \ (x'*y)
        y_new = y - x*coef
    end
    return (y_new)
end # function detrend

"""
    get_bpf(; pass1=0.1, pass2=0.9, fs=10.0, pole::Int=4)

Create a Butterworth bandpass (or low-pass or high-pass) filter object. Set 
`pass1 = -1` for low-pass filter or `pass2 = -1` for high-pass filter.

**Arguments:**
- `pass1`: (optional) first  passband frequency [Hz]
- `pass2`: (optional) second passband frequency [Hz]
- `fs`:    (optional) sampling frequency [Hz]
- `pole`:  (optional) number of poles for Butterworth filter

**Returns:**
- `bpf`: filter object
"""
function get_bpf(; pass1=0.1, pass2=0.9, fs=10.0, pole::Int=4)
    if     ((pass1 >  0) & (pass1 <  fs/2)) & ((pass2 >  0) & (pass2 <  fs/2))
        p = Bandpass(pass1,pass2;fs=fs) # bandpass
    elseif ((pass1 <= 0) | (pass1 >= fs/2)) & ((pass2 >  0) & (pass2 <  fs/2))
        p = Lowpass(pass2;fs=fs)        # low-pass
    elseif ((pass1 >  0) & (pass1 <  fs/2)) & ((pass2 <= 0) | (pass2 >= fs/2))
        p = Highpass(pass1;fs=fs)       # high-pass
    else
        error("$pass1 & $pass2 passband frequencies are not valid")
    end
    return (digitalfilter(p,Butterworth(pole)))
end # function get_bpf

"""
    bpf_data(x::Matrix; bpf=get_bpf())

Bandpass (or low-pass or high-pass) filter columns of matrix.

**Arguments:**
- `x`: matrix of input data (e.g. Tolles-Lawson `A` matrix)
- `bpf`: (optional) filter object

**Returns:**
- `x_f`: matrix of filtered data
"""
function bpf_data(x::Matrix; bpf=get_bpf())
    x_f = deepcopy(x)
    for i in axes(x,2)
        (std(x[:,i]) <= eps(eltype(x))) || (x_f[:,i] = filtfilt(bpf,x[:,i]))
    end
    return (x_f)
end # function bpf_data

"""
    bpf_data(x::Vector; bpf=get_bpf())

Bandpass (or low-pass or high-pass) filter vector.

**Arguments:**
- `x`: input data (e.g. magnetometer measurements)
- `bpf`: (optional) filter object

**Returns:**
- `x_f`: filtered data
"""
function bpf_data(x::Vector; bpf=get_bpf())
    filtfilt(bpf,x)
end # function bpf_data

"""
    bpf_data!(x; bpf=get_bpf())

Bandpass (or low-pass or high-pass) filter vector or columns of matrix.

**Arguments:**
- `x`: vector or matrix of input data
- `bpf`: (optional) filter object

**Returns:**
- `x_f`: vector or matrix of filtered data (mutated)
"""
function bpf_data!(x; bpf=get_bpf())
    x .= bpf_data(x;bpf=bpf)
end # bpf_data!

"""
    get_x(xyz::XYZ, ind = trues(xyz.traj.N),
          features_setup::Vector{Symbol}   = [:mag_1_uc,:TL_A_flux_a];
          features_no_norm::Vector{Symbol} = Symbol[],
          terms             = [:permanent,:induced,:eddy],
          sub_diurnal::Bool = false,
          sub_igrf::Bool    = false,
          bpf_mag::Bool     = false)

Get `x` matrix.

**Arguments:**
- `xyz`:              `XYZ` flight data struct
- `ind`:              selected data indices
- `features_setup`:   list of features to include
- `features_no_norm`: (optional) list of features to not normalize
- `terms`:            (optional) Tolles-Lawson terms to use {`:permanent`,`:induced`,`:eddy`,`:bias`}
- `sub_diurnal`:      (optional) if true, subtract diurnal from scalar magnetometer measurements
- `sub_igrf`:         (optional) if true, subtract IGRF from scalar magnetometer measurements
- `bpf_mag`:          (optional) if true, bpf scalar magnetometer measurements

**Returns:**
- `nn_x`:     `x` matrix
- `no_norm`:  indices of features to not be normalized
- `features`: full list of features (including components of TL `A`, etc.)
"""
function get_x(xyz::XYZ, ind = trues(xyz.traj.N),
               features_setup::Vector{Symbol}   = [:mag_1_uc,:TL_A_flux_a];
               features_no_norm::Vector{Symbol} = Symbol[],
               terms             = [:permanent,:induced,:eddy],
               sub_diurnal::Bool = false,
               sub_igrf::Bool    = false,
               bpf_mag::Bool     = false)

    d = Dict()
    N = length(xyz.traj.lat[ind])
    nn_x     = Array{eltype(xyz.traj.lat)}(undef,N,0)
    no_norm  = Array{eltype(ind)}(undef,0,1)
    features = Array{Symbol}(undef,0,1)

    for use_vec in field_check(xyz,MagV)
        TL_A = create_TL_A(getfield(xyz,use_vec),ind;terms=terms)
        push!(d,Symbol("TL_A_",use_vec)=>TL_A)
    end

    # subtract diurnal and/or IGRF as specified
    sub = zeros(N)
    sub_diurnal && (sub += xyz.diurnal[ind])
    sub_igrf    && (sub += xyz.igrf[ind])

    fields   = fieldnames(typeof(xyz))
    list_c   = [Symbol("mag_",i,"_c" ) for i = 1:num_mag_max]
    list_uc  = [Symbol("mag_",i,"_uc") for i = 1:num_mag_max]
    mags_c   = list_c[  list_c  .∈ (fields,)]
    mags_uc  = list_uc[ list_uc .∈ (fields,)]
    mags_all = [mags_c; mags_uc]

    for mag in mags_all
        lab = mag
        val = getfield(xyz,mag)[ind] - sub
        bpf_mag && bpf_data!(val)
        push!(d,lab=>val)
    end

    for mag in mags_all
        lab = Symbol(mag,"_dot")
        val = fdm(getfield(xyz,mag)[ind] - sub)
        push!(d,lab=>val)
    end

    # 4th-order central difference
    # Reference: Loughlin Tuck, Characterization and compensation of magnetic 
    # interference resulting from unmanned aircraft systems, 2019. (pg. 28)
    for mag in mags_all
        lab = Symbol(mag,"_dot4")
        val = fdm(getfield(xyz,mag)[ind] - sub;fourth=true)
        push!(d,lab=>val)
    end

    for i = 1:3
        for mag in mags_all
            lab = Symbol(mag,"_lag_",i)
            val = getfield(xyz,mag)[ind] - sub
            val = val[[1:i;1:end-i]]
            push!(d,lab=>val)
        end
    end

    for (i,mag1) in enumerate(mags_c)
        for (j,mag2) in enumerate(mags_c)
            lab = Symbol("mag_",i,"_",j,"_c")
            val = (getfield(xyz,mag1) - getfield(xyz,mag2))[ind]
            push!(d,lab=>val)
        end
    end

    for (i,mag1) in enumerate(mags_uc)
        for (j,mag2) in enumerate(mags_uc)
            lab = Symbol("mag_",i,"_",j,"_uc")
            val = (getfield(xyz,mag1) - getfield(xyz,mag2))[ind]
            push!(d,lab=>val)
        end
    end

    (roll,pitch,yaw) = dcm2euler(xyz.ins.Cnb[:,:,ind],:body2nav) # correct definition
    push!(d,:dcm=>[reshape(euler2dcm(roll,pitch,yaw,:nav2body),(9,N))';]) # ordered this way initally, leaving for now consistency
    push!(d,:dcm_1=>euler2dcm(roll,pitch,yaw,:nav2body)[1,1,:])           # note :nav2body contains same terms as :body2nav, just transposed
    push!(d,:dcm_2=>euler2dcm(roll,pitch,yaw,:nav2body)[2,1,:])
    push!(d,:dcm_3=>euler2dcm(roll,pitch,yaw,:nav2body)[3,1,:])
    push!(d,:dcm_4=>euler2dcm(roll,pitch,yaw,:nav2body)[1,2,:])
    push!(d,:dcm_5=>euler2dcm(roll,pitch,yaw,:nav2body)[2,2,:])
    push!(d,:dcm_6=>euler2dcm(roll,pitch,yaw,:nav2body)[3,2,:])
    push!(d,:dcm_7=>euler2dcm(roll,pitch,yaw,:nav2body)[1,3,:])
    push!(d,:dcm_8=>euler2dcm(roll,pitch,yaw,:nav2body)[2,3,:])
    push!(d,:dcm_9=>euler2dcm(roll,pitch,yaw,:nav2body)[3,3,:])
    push!(d,:crcy  =>cos.(roll ).*cos.(yaw))
    push!(d,:cpcy  =>cos.(pitch).*cos.(yaw))
    push!(d,:crsy  =>cos.(roll ).*sin.(yaw))
    push!(d,:cpsy  =>cos.(pitch).*sin.(yaw))
    push!(d,:srcy  =>sin.(roll ).*cos.(yaw))
    push!(d,:spcy  =>sin.(pitch).*cos.(yaw))
    push!(d,:srsy  =>sin.(roll ).*sin.(yaw))
    push!(d,:spsy  =>sin.(pitch).*sin.(yaw))
    push!(d,:crcpcy=>cos.(roll ).*cos.(pitch).*cos.(yaw))
    push!(d,:srcpcy=>sin.(roll ).*cos.(pitch).*cos.(yaw))
    push!(d,:crspcy=>cos.(roll ).*sin.(pitch).*cos.(yaw))
    push!(d,:srspcy=>sin.(roll ).*sin.(pitch).*cos.(yaw))
    push!(d,:crcpsy=>cos.(roll ).*cos.(pitch).*sin.(yaw))
    push!(d,:srcpsy=>sin.(roll ).*cos.(pitch).*sin.(yaw))
    push!(d,:crspsy=>cos.(roll ).*sin.(pitch).*sin.(yaw))
    push!(d,:srspsy=>sin.(roll ).*sin.(pitch).*sin.(yaw))
    push!(d,:crcp  =>cos.(roll ).*cos.(pitch))
    push!(d,:srcp  =>sin.(roll ).*cos.(pitch))
    push!(d,:crsp  =>cos.(roll ).*sin.(pitch))
    push!(d,:srsp  =>sin.(roll ).*sin.(pitch))

    for rpy in [:roll,:pitch,:yaw]
        rpy == :roll  && (x = roll)
        rpy == :pitch && (x = pitch)
        rpy == :yaw   && (x = yaw)
        push!(d,Symbol(rpy,"_fdm")=>fdm(x))
        push!(d,Symbol(rpy,"_sin")=>sin.(x))
        push!(d,Symbol(rpy,"_cos")=>cos.(x))
        push!(d,Symbol(rpy,"_sin_fdm")=>fdm(sin.(x)))
        push!(d,Symbol(rpy,"_cos_fdm")=>fdm(cos.(x)))
    end

    # low-pass filter current sensors
    if N > 12
        lpf = get_bpf(;pass1=0.0,pass2=0.2,fs=1/xyz.traj.dt);
        hasproperty(xyz, :cur_strb)   && push!(d,:lpf_cur_strb  =>bpf_data(xyz.cur_strb[ind];  bpf=lpf))
        hasproperty(xyz, :cur_outpwr) && push!(d,:lpf_cur_outpwr=>bpf_data(xyz.cur_outpwr[ind];bpf=lpf))
        hasproperty(xyz, :cur_ac_hi)  && push!(d,:lpf_cur_ac_hi =>bpf_data(xyz.cur_ac_hi[ind]; bpf=lpf))
        hasproperty(xyz, :cur_ac_lo)  && push!(d,:lpf_cur_ac_lo =>bpf_data(xyz.cur_ac_lo[ind]; bpf=lpf))
        hasproperty(xyz, :cur_com_1)  && push!(d,:lpf_cur_com_1 =>bpf_data(xyz.cur_com_1[ind]; bpf=lpf))
    end

    push!(d,:ins_lat=>xyz.ins.lat[ind])
    push!(d,:ins_lon=>xyz.ins.lon[ind])
    push!(d,:ins_alt=>xyz.ins.alt[ind])

    for f in features_setup
        if f in keys(d)
            u = d[f]
        elseif f in fields
            u = getfield(xyz,f)[ind]
        else
            error("$f is not a valid feature")
        end

        nf = size(u,2)
        v  = f in features_no_norm ? trues(nf) : falses(nf)
        w  = nf > 1 ? [Symbol(f,"_",i) for i = 1:nf] : f

        if isnan(sum(u))
            error("$f contains NaNs, remove from features_setup")
        else
            nn_x     = [nn_x      u]
            no_norm  = [no_norm;  v]
            features = [features; w]
        end
    end

    return (nn_x, vec(no_norm), vec(features))
end # function get_x

"""
    get_x(xyz::Vector{XYZ}, ind,
          features_setup::Vector{Symbol}   = [:mag_1_uc,:TL_A_flux_a];
          features_no_norm::Vector{Symbol} = Symbol[],
          terms             = [:permanent,:induced,:eddy],
          sub_diurnal::Bool = false,
          sub_igrf::Bool    = false,
          bpf_mag::Bool     = false)

Get `x` matrix from multiple `XYZ` flight data structs.

**Arguments:**
- `xyz`:              vector of `XYZ` flight data structs
- `ind`:              vector of selected data indices
- `features_setup`:   list of features to include
- `features_no_norm`: (optional) list of features to not normalize
- `terms`:            (optional) Tolles-Lawson terms to use {`:permanent`,`:induced`,`:eddy`,`:bias`}
- `sub_diurnal`:      (optional) if true, subtract diurnal from scalar magnetometer measurements
- `sub_igrf`:         (optional) if true, subtract IGRF from scalar magnetometer measurements
- `bpf_mag`:          (optional) if true, bpf scalar magnetometer measurements

**Returns:**
- `nn_x`:     `x` matrix
- `no_norm`:  indices of features to not be normalized
- `features`: full list of features (including components of TL `A`, etc.)
"""
function get_x(xyz::Vector, ind::Vector,
               features_setup::Vector{Symbol}   = [:mag_1_uc,:TL_A_flux_a];
               features_no_norm::Vector{Symbol} = Symbol[],
               terms             = [:permanent,:induced,:eddy],
               sub_diurnal::Bool = false,
               sub_igrf::Bool    = false,
               bpf_mag::Bool     = false)

    nn_x     = nothing
    no_norm  = nothing
    features = nothing

    for i in eachindex(xyz)
        if i == 1
            (nn_x,no_norm,features) = get_x(xyz[i],ind[i],features_setup;
                                        features_no_norm = features_no_norm,
                                        terms            = terms,
                                        sub_diurnal      = sub_diurnal,
                                        sub_igrf         = sub_igrf,
                                        bpf_mag          = bpf_mag)
        else
            nn_x = vcat(nn_x,get_x(xyz[i],ind[i],features_setup;
                                   features_no_norm = features_no_norm,
                                   terms            = terms,
                                   sub_diurnal      = sub_diurnal,
                                   sub_igrf         = sub_igrf,
                                   bpf_mag          = bpf_mag)[1])
        end
    end

    return (nn_x, no_norm, features)
end # function get_x

"""
    get_x(lines, df_line::DataFrame, df_flight::DataFrame,
          features_setup::Vector{Symbol}   = [:mag_1_uc,:TL_A_flux_a];
          features_no_norm::Vector{Symbol} = Symbol[],
          terms             = [:permanent,:induced,:eddy],
          sub_diurnal::Bool = false,
          sub_igrf::Bool    = false,
          bpf_mag::Bool     = false,
          l_seq::Int        = -1,
          silent::Bool      = true)

Get `x` matrix for multiple lines, possibly multiple flights.

**Arguments:**
- `lines`:            selected line number(s)
- `df_line`:          lookup table (DataFrame) of `lines`
- `df_flight`:        lookup table (DataFrame) of flight files
- `features_setup`:   list of features to include
- `features_no_norm`: (optional) list of features to not normalize
- `terms`:            (optional) Tolles-Lawson terms to use {`:permanent`,`:induced`,`:eddy`,`:bias`}
- `sub_diurnal`:      (optional) if true, subtract diurnal from scalar magnetometer measurements
- `sub_igrf`:         (optional) if true, subtract IGRF from scalar magnetometer measurements
- `bpf_mag`:          (optional) if true, bpf scalar magnetometer measurements
- `l_seq`:            (optional) trim data by `mod(N,l_seq)`, `-1` to ignore
- `silent`:           (optional) if true, no print outs

**Returns:**
- `nn_x`:     `x` matrix
- `no_norm`:  indices of features to not be normalized
- `features`: full list of features (including components of TL `A`, etc.)
"""
function get_x(lines, df_line::DataFrame, df_flight::DataFrame,
               features_setup::Vector{Symbol}   = [:mag_1_uc,:TL_A_flux_a];
               features_no_norm::Vector{Symbol} = Symbol[],
               terms             = [:permanent,:induced,:eddy],
               sub_diurnal::Bool = false,
               sub_igrf::Bool    = false,
               bpf_mag::Bool     = false,
               l_seq::Int        = -1,
               silent::Bool      = true)

    # check if lines are in df_line, remove if not
    for l in lines
        if !(l in df_line.line)
            @info("line $l is not in df_line, skipping")
            lines = lines[lines.!=l]
        end
    end

    # check if lines are duplicated in df_line, throw error if so
    length(lines) != length(unique(lines)) && error("df_line contains duplicated lines")

    # check if flight data matches up
    flights  = [df_line.flight[df_line.line .== l][1] for l in lines]
    xyz_sets = [df_flight.xyz_set[df_flight.flight .== f][1] for f in flights]
    any(xyz_sets.!=xyz_sets[1]) && error("incompatible xyz_sets in df_flight")

    # initial values
    flt_old  = :FltInitial
    nn_x     = nothing
    no_norm  = nothing
    xyz      = nothing
    features = nothing

    for line in lines
        flt = df_line.flight[df_line.line .== line][1]
        if flt != flt_old
            xyz = get_XYZ(flt,df_flight;silent=silent)
        end
        flt_old = flt

        ind = get_ind(xyz,line,df_line;l_seq=l_seq)

        if nn_x === nothing
            (nn_x,no_norm,features) = get_x(xyz,ind,features_setup;
                                            features_no_norm = features_no_norm,
                                            terms            = terms,
                                            sub_diurnal      = sub_diurnal,
                                            sub_igrf         = sub_igrf,
                                            bpf_mag          = bpf_mag)
        else
            nn_x = vcat(nn_x,get_x(xyz,ind,features_setup;
                                   features_no_norm = features_no_norm,
                                   terms            = terms,
                                   sub_diurnal      = sub_diurnal,
                                   sub_igrf         = sub_igrf,
                                   bpf_mag          = bpf_mag)[1])
        end
    end

    return (nn_x, no_norm, features)
end # function get_x

"""
    get_y(xyz::XYZ, ind = trues(xyz.traj.N),
          map_val           = -1);
          y_type::Symbol    = :d,
          use_mag::Symbol   = :mag_1_uc,
          use_mag_c::Symbol = :mag_1_c,
          sub_diurnal::Bool = false,
          sub_igrf::Bool    = false)

Get `y` target vector.

**Arguments:**
- `xyz`:     `XYZ` flight data struct
- `ind`:     selected data indices
- `map_val`: (optional) magnetic anomaly map value, only used for `y_type = :b`
- `y_type`:  (optional) `y` target type
    - `:a` = anomaly field  #1, compensated tail stinger total field scalar magnetometer measurements
    - `:b` = anomaly field  #2, interpolated `magnetic anomaly map` values
    - `:c` = aircraft field #1, difference between uncompensated cabin total field scalar magnetometer measurements and interpolated `magnetic anomaly map` values
    - `:d` = aircraft field #2, difference between uncompensated cabin and compensated tail stinger total field scalar magnetometer measurements
    - `:e` = BPF'd total field, bandpass filtered uncompensated cabin total field scalar magnetometer measurements
- `use_mag`:     (optional) scalar magnetometer to use {`:mag_1_uc`, etc.}, only used for `y_type = :c, :d, :e`
- `use_mag_c`:   (optional) compensated scalar magnetometer to use {`:mag_1_c`, etc.}, only used for `y_type = :a, :d`
- `sub_diurnal`: (optional) if true, subtract diurnal from scalar magnetometer measurements
- `sub_igrf`:    (optional) if true, subtract IGRF from scalar magnetometer measurements

**Returns:**
- `nn_y`: `y` target vector
"""
function get_y(xyz::XYZ, ind = trues(xyz.traj.N),
               map_val           = -1;
               y_type::Symbol    = :d,
               use_mag::Symbol   = :mag_1_uc,
               use_mag_c::Symbol = :mag_1_c,
               sub_diurnal::Bool = false,
               sub_igrf::Bool    = false)

    # selected scalar mags, as needed
    if y_type in [:c,:d,:e]
        if typeof(getfield(xyz,use_mag)) <: MagV
            mag_uc = getfield(xyz,use_mag)(ind).t
        else
            mag_uc = getfield(xyz,use_mag)[ind]
        end
    end

    y_type in [:a,:d] && (mag_c  = getfield(xyz,use_mag_c)[ind])

    # subtract diurnal and/or IGRF as specified
    sub = zero(xyz.traj.lat[ind])
    sub_diurnal && (sub += xyz.diurnal[ind])
    sub_igrf    && (sub += xyz.igrf[ind])

    # get y for specified y_type
    if y_type == :a # option A
        nn_y = mag_c - sub
    elseif y_type == :b # option B
        nn_y = map_val
    elseif y_type == :c # option C
        nn_y = mag_uc - sub - map_val
    elseif y_type == :d # option D
        nn_y = mag_uc - mag_c
    elseif y_type == :e # option E
        fs   = 1 / xyz.traj.dt
        nn_y = bpf_data(mag_uc - sub; bpf=get_bpf(;fs=fs))
    else
        error("$y_type target type is not valid")
    end

    return (nn_y)
end # function get_y

"""
    get_y(lines, df_line::DataFrame, df_flight::DataFrame,
          df_map::DataFrame;
          y_type::Symbol    = :d,
          use_mag::Symbol   = :mag_1_uc,
          use_mag_c::Symbol = :mag_1_c,
          sub_diurnal::Bool = false,
          sub_igrf::Bool    = false,
          l_seq::Int        = -1,
          silent::Bool      = true)

Get `y` target vector for multiple lines, possibly multiple flights.

**Arguments:**
- `lines`:     selected line number(s)
- `df_line`:   lookup table (DataFrame) of `lines`
- `df_flight`: lookup table (DataFrame) of flight files
- `df_map`:    lookup table (DataFrame) of map files
- `y_type`:    (optional) `y` target type
    - `:a` = anomaly field  #1, compensated tail stinger total field scalar magnetometer measurements
    - `:b` = anomaly field  #2, interpolated `magnetic anomaly map` values
    - `:c` = aircraft field #1, difference between uncompensated cabin total field scalar magnetometer measurements and interpolated `magnetic anomaly map` values
    - `:d` = aircraft field #2, difference between uncompensated cabin and compensated tail stinger total field scalar magnetometer measurements
    - `:e` = BPF'd total field, bandpass filtered uncompensated cabin total field scalar magnetometer measurements
- `use_mag`:     (optional) scalar magnetometer to use {`:mag_1_uc`, etc.}, only used for `y_type = :c, :d, :e`
- `use_mag_c`:   (optional) compensated scalar magnetometer to use {`:mag_1_c`, etc.}, only used for `y_type = :a, :d`
- `sub_diurnal`: (optional) if true, subtract diurnal from scalar magnetometer measurements
- `sub_igrf`:    (optional) if true, subtract IGRF from scalar magnetometer measurements
- `l_seq`:       (optional) trim data by `mod(N,l_seq)`, `-1` to ignore
- `silent`:      (optional) if true, no print outs

**Returns:**
- `nn_y`: `y` target vector
"""
function get_y(lines, df_line::DataFrame, df_flight::DataFrame,
               df_map::DataFrame;
               y_type::Symbol    = :d,
               use_mag::Symbol   = :mag_1_uc,
               use_mag_c::Symbol = :mag_1_c,
               sub_diurnal::Bool = false,
               sub_igrf::Bool    = false,
               l_seq::Int        = -1,
               silent::Bool      = true)

    # check if lines are in df_line, remove if not
    for l in lines
        if !(l in df_line.line)
            @info("line $l is not in df_line, skipping")
            lines = lines[lines.!=l]
        end
    end

    # check if lines are duplicated in df_line, throw error if so
    length(lines) != length(unique(lines)) && error("df_line contains duplicated lines")

    # initial values
    flt_old = :FltInitial
    nn_y    = nothing
    xyz     = nothing

    for line in lines
        flt = df_line.flight[df_line.line .== line][1]
        if flt != flt_old
            xyz = get_XYZ(flt,df_flight;silent=silent)
        end
        flt_old = flt

        ind = get_ind(xyz,line,df_line;l_seq=l_seq)

        # map values along trajectory (if needed)
        if y_type in [:b,:c]
            map_name = df_line.map_name[df_line.line .== line][1]
            mapS     = get_map(map_name,df_map)
            traj_alt = median(xyz.traj.alt[ind])
            mapS.alt > 0 && (mapS = upward_fft(mapS,traj_alt;α=200))
            itp_mapS = map_itp(mapS)
            map_val  = itp_mapS.(xyz.traj.lon[ind],xyz.traj.lat[ind])
        else
            map_val  = -1
        end

        if nn_y === nothing
            nn_y = get_y(xyz,ind,map_val;
                         y_type      = y_type,
                         use_mag     = use_mag,
                         use_mag_c   = use_mag_c,
                         sub_diurnal = sub_diurnal,
                         sub_igrf    = sub_igrf)
        else
            nn_y = vcat(nn_y,get_y(xyz,ind,map_val;
                                   y_type      = y_type,
                                   use_mag     = use_mag,
                                   use_mag_c   = use_mag_c,
                                   sub_diurnal = sub_diurnal,
                                   sub_igrf    = sub_igrf))
        end
    end

    return (nn_y)
end # function get_y

"""
    get_Axy(lines, df_line::DataFrame,
            df_flight::DataFrame, df_map::DataFrame,
            features_setup::Vector{Symbol}   = [:mag_1_uc,:TL_A_flux_a];
            features_no_norm::Vector{Symbol} = Symbol[],
            y_type::Symbol    = :d,
            use_mag::Symbol   = :mag_1_uc,
            use_mag_c::Symbol = :mag_1_c,
            use_vec::Symbol   = :flux_a,
            terms             = [:permanent,:induced,:eddy],
            terms_A           = [:permanent,:induced,:eddy,:bias],
            sub_diurnal::Bool = false,
            sub_igrf::Bool    = false,
            bpf_mag::Bool     = false,
            l_seq::Int        = -1,
            mod_TL::Bool      = false,
            map_TL::Bool      = false,
            silent::Bool      = true)

Get Tolles-Lawson `A` matrix, `x` matrix, and `y` target vector for multiple 
lines, possibly multiple flights.

**Arguments:**
- `lines`:            selected line number(s)
- `df_line`:          lookup table (DataFrame) of `lines`
- `df_flight`:        lookup table (DataFrame) of flight files
- `df_map`:           lookup table (DataFrame) of map files
- `features_setup`:   list of features to include
- `features_no_norm`: (optional) list of features to not normalize
- `y_type`:           (optional) `y` target type
    - `:a` = anomaly field  #1, compensated tail stinger total field scalar magnetometer measurements
    - `:b` = anomaly field  #2, interpolated `magnetic anomaly map` values
    - `:c` = aircraft field #1, difference between uncompensated cabin total field scalar magnetometer measurements and interpolated `magnetic anomaly map` values
    - `:d` = aircraft field #2, difference between uncompensated cabin and compensated tail stinger total field scalar magnetometer measurements
    - `:e` = BPF'd total field, bandpass filtered uncompensated cabin total field scalar magnetometer measurements
- `use_mag`:     (optional) scalar magnetometer to use {`:mag_1_uc`, etc.}, only used for `y_type = :c, :d, :e`
- `use_mag_c`:   (optional) compensated scalar magnetometer to use {`:mag_1_c`, etc.}, only used for `y_type = :a, :d`
- `use_vec`:     (optional) vector magnetometer (fluxgate) to use for Tolles-Lawson `A` matrix {`:flux_a`, etc.}
- `terms`:       (optional) Tolles-Lawson terms to use for `A` within `x` matrix {`:permanent`,`:induced`,`:eddy`,`:bias`}
- `terms_A`:     (optional) Tolles-Lawson terms to use for "external" `A` matrix {`:permanent`,`:induced`,`:eddy`,`:bias`}
- `sub_diurnal`: (optional) if true, subtract diurnal from scalar magnetometer measurements
- `sub_igrf`:    (optional) if true, subtract IGRF from scalar magnetometer measurements
- `bpf_mag`:     (optional) if true, bpf scalar magnetometer measurements in `x` matrix
- `l_seq`:       (optional) trim data by `mod(N,l_seq)`, `-1` to ignore
- `mod_TL`:      (optional) if true, create modified  Tolles-Lawson `A` matrix with `use_mag` 
- `map_TL`:      (optional) if true, create map-based Tolles-Lawson `A` matrix 
- `silent`:      (optional) if true, no print outs

**Returns:**
- `nn_A`:     Tolles-Lawson `A` matrix
- `nn_x`:     `x` matrix
- `nn_y`:     `y` target vector
- `no_norm`:  indices of features to not be normalized
- `features`: full list of features (including components of TL `A`, etc.)
- `l_segs`:   vector of lengths of `lines`, sum(l_segs) == length(y)
"""
function get_Axy(lines, df_line::DataFrame,
                 df_flight::DataFrame, df_map::DataFrame,
                 features_setup::Vector{Symbol}   = [:mag_1_uc,:TL_A_flux_a];
                 features_no_norm::Vector{Symbol} = Symbol[],
                 y_type::Symbol    = :d,
                 use_mag::Symbol   = :mag_1_uc,
                 use_mag_c::Symbol = :mag_1_c,
                 use_vec::Symbol   = :flux_a,
                 terms             = [:permanent,:induced,:eddy],
                 terms_A           = [:permanent,:induced,:eddy,:bias],
                 sub_diurnal::Bool = false,
                 sub_igrf::Bool    = false,
                 bpf_mag::Bool     = false,
                 l_seq::Int        = -1,
                 mod_TL::Bool      = false,
                 map_TL::Bool      = false,
                 silent::Bool      = true)

    # check if lines are in df_line, remove if not
    for l in lines
        if !(l in df_line.line)
            @info("line $l is not in df_line, skipping")
            lines = lines[lines.!=l]
        end
    end

    # check if lines are duplicated in df_line, throw error if so
    length(lines) != length(unique(lines)) && error("df_line contains duplicated lines")

    # check if flight data matches up
    flights  = [df_line.flight[df_line.line .== l][1] for l in lines]
    xyz_sets = [df_flight.xyz_set[df_flight.flight .== f][1] for f in flights]
    any(xyz_sets.!=xyz_sets[1]) && error("incompatible xyz_sets in df_flight")

    # initial values
    flt_old  = :FltInitial
    A_test   = create_TL_A([1.0],[1.0],[1.0];terms=terms_A)
    nn_A     = Array{eltype(A_test)}(undef,0,length(A_test))
    nn_x     = nothing
    no_norm  = nothing
    features = nothing
    nn_y     = nothing
    xyz      = nothing
    l_segs   = zeros(Int,length(lines))

    for line in lines
        flt = df_line.flight[df_line.line .== line][1]
        if flt != flt_old
            xyz = get_XYZ(flt,df_flight;silent=silent)
        end
        flt_old = flt

        ind = get_ind(xyz,line,df_line;l_seq=l_seq)
        l_segs[findfirst(l_segs .== 0)] = length(xyz.traj.lat[ind])

        # x matrix
        if nn_x === nothing
            (nn_x,no_norm,features) = get_x(xyz,ind,features_setup;
                                            features_no_norm = features_no_norm,
                                            terms            = terms,
                                            sub_diurnal      = sub_diurnal,
                                            sub_igrf         = sub_igrf,
                                            bpf_mag          = bpf_mag)
        else
            nn_x = vcat(nn_x,get_x(xyz,ind,features_setup;
                                   features_no_norm = features_no_norm,
                                   terms            = terms,
                                   sub_diurnal      = sub_diurnal,
                                   sub_igrf         = sub_igrf,
                                   bpf_mag          = bpf_mag)[1])
        end

        # map values along trajectory (if needed)
        if y_type in [:b,:c]
            map_name = df_line.map_name[df_line.line .== line][1]
            mapS     = get_map(map_name,df_map)
            traj_alt = median(xyz.traj.alt[ind])
            mapS.alt > 0 && (mapS = upward_fft(mapS,traj_alt;α=200))
            itp_mapS = map_itp(mapS)
            map_val  = itp_mapS.(xyz.traj.lon[ind],xyz.traj.lat[ind])
        else
            map_val  = -1
        end

        # `A` matrix for selected vector magnetometer
        field_check(xyz,use_vec,MagV)
        if mod_TL
            A = create_TL_A(getfield(xyz,use_vec),ind;
                            Bt=getfield(xyz,use_mag),terms=terms_A)
        elseif map_TL
            A = create_TL_A(getfield(xyz,use_vec),ind;
                            Bt=map_val,terms=terms_A)
        else
            A = create_TL_A(getfield(xyz,use_vec),ind;terms=terms_A)
        end
        fs = 1 / xyz.traj.dt
        y_type == :e && bpf_data!(A;bpf=get_bpf(;fs=fs))

        nn_A = vcat(nn_A,A)

        # y vector
        if nn_y === nothing
            nn_y = get_y(xyz,ind,map_val;
                         y_type      = y_type,
                         use_mag     = use_mag,
                         use_mag_c   = use_mag_c,
                         sub_diurnal = sub_diurnal,
                         sub_igrf    = sub_igrf)
        else
            nn_y = vcat(nn_y,get_y(xyz,ind,map_val;
                                   y_type      = y_type,
                                   use_mag     = use_mag,
                                   use_mag_c   = use_mag_c,
                                   sub_diurnal = sub_diurnal,
                                   sub_igrf    = sub_igrf))
        end
    end

    return (nn_A, nn_x, nn_y, no_norm, features, l_segs)
end # function get_Axy

"""
    get_nn_m(xS::Int=10, yS::Int=1;
             hidden               = [8],
             activation::Function = swish,
             final_bias::Bool     = true,
             skip_con::Bool       = false)

Get neural network model. Valid for 0-3 hidden layers.

**Arguments:**
- `xS`: size of input layer
- `yS`: size of output layer
- `hidden`:     (optional) hidden layers & nodes, e.g. `[8,8]` for 2 hidden layers, 8 nodes each
- `activation`: (optional) activation function
    - `relu`  = rectified linear unit
    - `σ`     = sigmoid (logistic function)
    - `swish` = self-gated
    - `tanh`  = hyperbolic tan
    - run `plot_activation()` for a visual
- `final_bias`: (optional) if true, include final layer bias
- `skip_con`:   (optional) if true, use skip connections, must have length(`hidden`) == 1

**Returns:**
- `m`: neural network model
"""
function get_nn_m(xS::Int=10, yS::Int=1;
                  hidden               = [8],
                  activation::Function = swish,
                  final_bias::Bool     = true,
                  skip_con::Bool       = false)

    l = length(hidden)

    if l == 0
        m = Chain(Dense(xS,yS;bias=final_bias))
    elseif skip_con
        if l == 1
            m = Chain(SkipConnection(Dense(xS,hidden[1],activation),
                                     (mx, x) -> cat(mx, x, dims=1)),
                      Dense(xS+hidden[1],yS;bias=final_bias))
        else
            error("$l hidden layers not valid with skip connections")
        end
    elseif l == 1
        m = Chain(Dense(xS,hidden[1],activation),
                  Dense(hidden[1],yS;bias=final_bias))
    elseif l == 2
        m = Chain(Dense(xS,hidden[1],activation),
                  Dense(hidden[1],hidden[2],activation),
                  Dense(hidden[2],yS;bias=final_bias))
    elseif l == 3
        m = Chain(Dense(xS,hidden[1],activation),
                  Dense(hidden[1],hidden[2],activation),
                  Dense(hidden[2],hidden[3],activation),
                  Dense(hidden[3],yS;bias=final_bias))
    else
        error("$l hidden layers not valid")
    end

    return (m)
end # function get_nn_m

"""
    get_nn_m(weights::Params, activation::Function=swish; load::Bool=true)

Get neural network model.

**Arguments:**
- `weights`: neural network model weights
- `activation`: (optional) activation function
    - `relu`  = rectified linear unit
    - `σ`     = sigmoid (logistic function)
    - `swish` = self-gated
    - `tanh`  = hyperbolic tan
    - run `plot_activation()` for a visual
- `load`: if true, load `weights` into `m`
**Returns:**
- `m`: neural network model
"""
function get_nn_m(weights::Params, activation::Function=swish; load::Bool=true)

    n  = length(weights)
    xS = size(weights[1],2)
    yS = size(weights[n],1)

    final_bias = isodd(n) ? false : true

    hidden = [size(weights[i],1) for i = 2:2:n-1]

    if final_bias
        skip_con = size(weights[n-1],2) == size(weights[n-2],1) ? false : true
    else
        skip_con = size(weights[n  ],2) == size(weights[n-1],1) ? false : true
    end

    m = get_nn_m(xS, yS;
                 hidden     = hidden,
                 activation = activation,
                 final_bias = final_bias,
                 skip_con   = skip_con)

    load && Flux.loadparams!(m,weights)

    return (m)
end # function get_nn_m

"""
    sparse_group_lasso(m, α=1)

**Arguments:**
- `m`: neural network model
- `α`: (optional) Lasso (`α=0`) vs group Lasso (`α=1`) balancing parameter {0:1}

**Returns:**
- `w_norm`: sparse group Lasso term
"""
function sparse_group_lasso(m, α=1)
    sparse_group_lasso(Flux.params(m),α)
end # function sparse_group_lasso

"""
    sparse_group_lasso(weights::Params, α=1)

Internal helper function to get the sparse group Lasso term for sparse-input 
regularization, which is the combined L1 & L2 norm of the first-layer neural 
network weights corresponding to each input feature.

Reference: Feng & Simon, Sparse-Input Neural Networks for High-dimensional 
Nonparametric Regression and Classification, 2017. (pg. 4)

**Arguments:**
- `weights`: neural network model weights
- `α`: (optional) Lasso (`α=0`) vs group Lasso (`α=1`) balancing parameter {0:1}

**Returns:**
- `w_norm`: sparse group Lasso term
"""
function sparse_group_lasso(weights::Params, α=1)
    return ([(1-α)*norm(weights[1][:,i],1) + 
                α *norm(weights[1][:,i],2) for i in axes(weights[1],2)])
end # function sparse_group_lasso

"""
    err_segs(y_hat, y, l_segs; silent::Bool=true)

Internal helper function to remove mean error from multiple individual flight 
lines within a larger dataset.

**Arguments:**
- `y_hat`:  predicted data
- `y`:      observed data
- `l_segs`: vector of lengths of `lines`, sum(l_segs) == length(y)

**Returns:**
- `err`: mean-corrected (per line) error
"""
function err_segs(y_hat, y, l_segs; silent::Bool=true)
    err = y_hat - y
    for i in eachindex(l_segs)
        (i1,i2) = cumsum(l_segs)[i] .- (l_segs[i]-1,0)
        err[i1:i2] .-= mean(err[i1:i2])
        err_std = round(std(err[i1:i2]),digits=2)
        silent || @info("line #$i error: $err_std nT")
    end
    return (err)
end # function err_segs

"""
    norm_sets(train;
              norm_type::Symbol = :standardize,
              no_norm           = falses(size(train,2)))

Normalize (or standardize) features (columns) of training data.

**Arguments:**
- `train`: training data
- `norm_type`: (optional) normalization type:
    - `:standardize` = Z-score normalization
    - `:normalize`   = min-max normalization
    - `:scale`       = scale by maximum absolute value, bias = 0
    - `:none`        = scale by 1, bias = 0
- `no_norm`: (optional) indices of features to not be normalized

**Returns:**
- `train_bias`:  training data biases (means, mins, or zeros)
- `train_scale`: training data scaling factors (std devs, maxs-mins, or ones)
- `train`:       normalized training data
"""
function norm_sets(train;
                   norm_type::Symbol = :standardize,
                   no_norm           = falses(size(train,2)))

    if !(typeof(no_norm) <: AbstractVector{Bool})
        no_norm = axes(train,2) .∈ (no_norm,)
    end

    if norm_type == :standardize
        train_bias  = mean(train,dims=1)
        train_scale = std( train,dims=1)
    elseif norm_type == :normalize
        train_bias  = minimum(train,dims=1)
        train_scale = maximum(train,dims=1) - train_bias
    elseif norm_type == :scale
        train_bias  = size(train,2) > 1 ? zero(train)[1:1,:] : zero(train)[1,:]
        train_scale = maximum(abs.(train),dims=1)
    elseif norm_type == :none
        train_bias  = size(train,2) > 1 ? zero(train)[1:1,:] : zero(train)[1,:]
        train_scale = size(train,2) > 1 ? one.(train)[1:1,:] : one.(train)[1,:]
    else
        error("$norm_type normalization type not defined")
    end

    for i in axes(train,2)
        no_norm[i] && (train_bias[i]  = 0.0)
        no_norm[i] && (train_scale[i] = 1.0)
    end

    train = (train .- train_bias) ./ train_scale

    return (train_bias, train_scale, train)
end # function norm_sets

"""
    norm_sets(train, test;
              norm_type::Symbol = :standardize,
              no_norm           = falses(size(train,2)))

Normalize (or standardize) features (columns) of training and testing data.

**Arguments:**
- `train`: training data
- `test`:  testing data
- `norm_type`: (optional) normalization type:
    - `:standardize` = Z-score normalization
    - `:normalize`   = min-max normalization
    - `:scale`       = scale by maximum absolute value, bias = 0
    - `:none`        = scale by 1, bias = 0
- `no_norm`: (optional) indices of features to not be normalized

**Returns:**
- `train_bias`:  training data biases (means, mins, or zeros)
- `train_scale`: training data scaling factors (std devs, maxs-mins, or ones)
- `train`:       normalized training data
- `test`:        normalized testing data
"""
function norm_sets(train, test;
                   norm_type::Symbol = :standardize,
                   no_norm           = falses(size(train,2)))

    if !(typeof(no_norm) <: AbstractVector{Bool})
        no_norm = axes(train,2) .∈ (no_norm,)
    end

    (train_bias,train_scale,train) = norm_sets(train;
                                               norm_type = norm_type,
                                               no_norm   = no_norm)

    test = (test .- train_bias) ./ train_scale

    return (train_bias, train_scale, train, test)
end # function norm_sets

"""
    norm_sets(train, val, test;
              norm_type::Symbol = :standardize,
              no_norm           = falses(size(train,2)))

Normalize (or standardize) features (columns) of training, validation, and 
testing data.

**Arguments:**
- `train`: training data
- `val`:   validation data
- `test`:  testing data
- `norm_type`: (optional) normalization type:
    - `:standardize` = Z-score normalization
    - `:normalize`   = min-max normalization
    - `:scale`       = scale by maximum absolute value, bias = 0
    - `:none`        = scale by 1, bias = 0
- `no_norm`: (optional) indices of features to not be normalized

**Returns:**
- `train_bias`:  training data biases (means, mins, or zeros)
- `train_scale`: training data scaling factors (std devs, maxs-mins, or ones)
- `train`:       normalized training data
- `val`:         normalized validation data
- `test`:        normalized testing data
"""
function norm_sets(train, val, test;
                   norm_type::Symbol = :standardize,
                   no_norm           = falses(size(train,2)))

    if !(typeof(no_norm) <: AbstractVector{Bool})
        no_norm = axes(train,2) .∈ (no_norm,)
    end

    (train_bias,train_scale,train) = norm_sets(train;
                                               norm_type = norm_type,
                                               no_norm   = no_norm)

    val  = (val  .- train_bias) ./ train_scale
    test = (test .- train_bias) ./ train_scale

    return (train_bias, train_scale, train, val, test)
end # function norm_sets

"""
    denorm_sets(train_bias, train_scale, train)

Denormalize (or destandardize) features (columns) of training data.

**Arguments:**
- `train_bias`:  training data biases (means, mins, or zeros)
- `train_scale`: training data scaling factors (std devs, maxs-mins, or ones)
- `train`:       normalized training data

**Returns:**
- `train`: training data
"""
function denorm_sets(train_bias, train_scale, train)
    train = train .* train_scale .+ train_bias
    return (train)
end # function denorm_sets

"""
    denorm_sets(train_bias, train_scale, train, test)

Denormalize (or destandardize) features (columns) of training and testing data.

**Arguments:**
- `train_bias`:  training data biases (means, mins, or zeros)
- `train_scale`: training data scaling factors (std devs, maxs-mins, or ones)
- `train`:       normalized training data
- `test`:        normalized testing data

**Returns:**
- `train`: training data
- `test`: testing data
"""
function denorm_sets(train_bias, train_scale, train, test)
    train = train .* train_scale .+ train_bias
    test  = test  .* train_scale .+ train_bias
    return (train, test)
end # function denorm_sets

"""
    denorm_sets(train_bias, train_scale, train, val, test)

Denormalize (or destandardize) features (columns) of training, validation, 
and testing data.

**Arguments:**
- `train_bias`:  training data biases (means, mins, or zeros)
- `train_scale`: training data scaling factors (std devs, maxs-mins, or ones)
- `train`:       normalized training data
- `val`:         normalized validation data
- `test`:        normalized testing data

**Returns:**
- `train`: training data
- `val`:   validation data
- `test`:  testing data
"""
function denorm_sets(train_bias, train_scale, train, val, test)
    train = train .* train_scale .+ train_bias
    val   = val   .* train_scale .+ train_bias
    test  = test  .* train_scale .+ train_bias
    return (train, val, test)
end # function denorm_sets

"""
    unpack_data_norms(data_norms)

Internal helper function that unpacks data normalizations, some of which may 
not be present due to earlier package versions being used.

**Arguments:**
- `data_norms`: Tuple of data normalizations, e.g. `(A_bias,A_scale,v_scale,x_bias,x_scale,y_bias,y_scale)`

**Returns:**
- `data_norms`: length-7 Tuple of data normalizations, `(A_bias,A_scale,v_scale,x_bias,x_scale,y_bias,y_scale)`
"""
function unpack_data_norms(data_norms::Tuple)
    if length(data_norms) == 7
        (A_bias,A_scale,v_scale,x_bias,x_scale,y_bias,y_scale) = data_norms
    elseif length(data_norms) == 6
        (A_bias,A_scale,x_bias,x_scale,y_bias,y_scale) = data_norms
        v_scale = I(size(x_scale,2))
    elseif length(data_norms) == 5
        (v_scale,x_bias,x_scale,y_bias,y_scale) = data_norms
        (A_bias,A_scale) = (0,1)
    elseif length(data_norms) == 4
        (x_bias,x_scale,y_bias,y_scale) = data_norms
        (A_bias,A_scale) = (0,1)
        v_scale = I(size(x_scale,2))
    elseif length(data_norms) > 7
        error("length of data_norms = $(length(data_norms)) > 7")
    else
        error("length of data_norms = $(length(data_norms)) < 4")
    end
    return (A_bias,A_scale,v_scale,x_bias,x_scale,y_bias,y_scale)
end # function unpack_data_norms

"""
    get_ind(tt::Vector, line::Vector;
            ind    = trues(length(tt)),
            lines  = (),
            tt_lim = (),
            splits = (1))

Get BitArray of indices for further analysis from specified indices (subset), 
line number(s), and/or time range. Any or all of these may be used.

**Arguments:**
- `tt`:     time [s]
- `line`:   line number(s)
- `ind`:    (optional) selected data indices. Defaults to use all data indices
- `lines`:  (optional) selected line number(s). Defaults to use all lines
- `tt_lim`: (optional) 2-element (inclusive) start and stop time limits, in seconds past midnight. Defaults to use full time range
- `splits`: (optional) data splits, must sum to 1

**Returns:**
- `ind`: BitArray (or tuple of BitArray) of selected data indices
"""
function get_ind(tt::Vector, line::Vector;
                 ind    = trues(length(tt)),
                 lines  = (),
                 tt_lim = (),
                 splits = (1))

    sum(splits) ≈ 1.0 || error("sum of splits = $(sum(splits)) ≠ 1")

    if typeof(ind) <: AbstractVector{Bool}
        indices = deepcopy(ind)
    else
        indices = eachindex(tt) .∈ (ind,)
    end

    N = length(tt)

    if ~isempty(lines)
        ind = falses(N)
        for l in lines
            ind .= ind .| (line .== l)
        end
    else
        ind = trues(N)
    end

    if length(tt_lim) == 1
        ind = ind .& (tt .<= minimum(tt_lim[1]))
    elseif length(tt_lim) == 2
        ind = ind .& (tt .>= minimum(tt_lim[1])) .& (tt .<= maximum(tt_lim[2]))
    elseif length(tt_lim) > 2
        error("length of tt_lim = ",length(tt_lim)," > 2")
    end

    ind = ind .& indices

    sum(ind) == 0 && @info("ind contains all falses")

    if length(splits) == 1
        return (ind)
    elseif length(splits) == 2
        ind1 = ind .& (cumsum(ind) .<= round(Int,sum(ind)*splits[1]))
        ind2 = ind .& .!ind1
        return (ind1, ind2)
    elseif length(splits) == 3
        ind1 = ind .& (cumsum(ind) .<= round(Int,sum(ind)*splits[1]))
        ind2 = ind .& (cumsum(ind) .<= round(Int,sum(ind)*(splits[1]+splits[2]))) .& .!ind1
        ind3 = ind .& .!(ind1 .| ind2)
        return (ind1, ind2, ind3)
    else
        error("number of splits = ",length(splits)," > 3")
    end

end # function get_ind

"""
    get_ind(xyz::XYZ;
            ind    = trues(xyz.traj.N),
            lines  = (),
            tt_lim = extrema(xyz.traj.tt),
            splits = (1))

Get BitArray of indices for further analysis from specified indices (subset), 
line number(s), and/or time range. Any or all of these may be used.

**Arguments:**
- `xyz`:    `XYZ` flight data struct
- `ind`:    (optional) selected data indices. Defaults to use all data indices
- `lines`:  (optional) selected line number(s). Defaults to use all lines
- `tt_lim`: (optional) 2-element (inclusive) start and stop time limits, in seconds past midnight. Defaults to use full time range
- `splits`: (optional) data splits, must sum to 1

**Returns:**
- `ind`: BitArray (or tuple of BitArray) of selected data indices
"""
function get_ind(xyz::XYZ;
                 ind    = trues(xyz.traj.N),
                 lines  = (),
                 tt_lim = extrema(xyz.traj.tt),
                 splits = (1))
    fields = fieldnames(typeof(xyz))
    line_  = :line in fields ? xyz.line : one.(xyz.traj.tt[ind])
    get_ind(xyz.traj.tt,line_;
            ind    = ind,
            lines  = lines,
            tt_lim = tt_lim,
            splits = splits)
end # function get_ind

"""
    get_ind(xyz::XYZ, line::Real, df_line::DataFrame;
            splits     = (1),
            l_seq::Int = -1)

Get BitArray of indices for further analysis via DataFrame lookup.

**Arguments:**
- `xyz`:     `XYZ` flight data struct
- `line`:    line number
- `df_line`: lookup table (DataFrame) of `lines`
- `splits`:  (optional) data splits, must sum to 1
- `l_seq`:   (optional) trim data by `mod(N,l_seq)`, `-1` to ignore

**Returns:**
- `ind`: BitArray (or tuple of BitArray) of selected data indices
"""
function get_ind(xyz::XYZ, line::Real, df_line::DataFrame;
                 splits     = (1),
                 l_seq::Int = -1)

    tt_lim = [df_line.t_start[df_line.line .== line][1],
              df_line.t_end[  df_line.line .== line][end]]
    fields = fieldnames(typeof(xyz))
    line_  = :line in fields ? xyz.line : one.(xyz.traj.tt[ind])
    inds   = get_ind(xyz.traj.tt,line_;
                     lines  = [line],
                     tt_lim = tt_lim,
                     splits = splits)

    if l_seq > 0
        if typeof((inds)) <: Tuple
            for ind in inds
                for _ = 1:mod(length(xyz.traj.lat[ind]),l_seq)
                    ind[findlast(ind.==1)] = 0
                end
            end
        else
            for _ = 1:mod(sum(inds),l_seq)
                inds[findlast(inds.==1)] = 0
            end
        end
    end

    return (inds)
end # function get_ind

"""
    get_ind(xyz::XYZ, lines, df_line::DataFrame;
            splits     = (1),
            l_seq::Int = -1)

Get BitArray of selected data indices for further analysis via DataFrame lookup.

**Arguments:**
- `xyz`:     `XYZ` flight data struct
- `lines`:   selected line number(s)
- `df_line`: lookup table (DataFrame) of `lines`
- `splits`:  (optional) data splits, must sum to 1
- `l_seq`:   (optional) trim data by `mod(N,l_seq)`, `-1` to ignore

**Returns:**
- `ind`: BitArray (or tuple of BitArray) of selected data indices
"""
function get_ind(xyz::XYZ, lines, df_line::DataFrame;
                 splits     = (1),
                 l_seq::Int = -1)

    sum(splits) ≈ 1.0 || error("sum of splits = $(sum(splits)) ≠ 1")

    ind = falses(xyz.traj.N)
    for line in lines
        if line in df_line.line
            ind .= ind .| get_ind(xyz,line,df_line;l_seq=l_seq)
        end
    end

    if length(splits) == 1
        return (ind)
    elseif length(splits) == 2
        ind1 = ind .& (cumsum(ind) .<= round(Int,sum(ind)*splits[1]))
        ind2 = ind .& .!ind1
        return (ind1, ind2)
    elseif length(splits) == 3
        ind1 = ind .& (cumsum(ind) .<= round(Int,sum(ind)*splits[1]))
        ind2 = ind .& (cumsum(ind) .<= round(Int,sum(ind)*(splits[1]+splits[2]))) .& .!ind1
        ind3 = ind .& .!(ind1 .| ind2)
        return (ind1, ind2, ind3)
    else
        error("number of splits = ",length(splits)," > 3")
    end

end # function get_ind

"""
    chunk_data(x, y, l_seq::Int)

Break data into non-overlapping sequences of length `l_seq`.

**Arguments:**
- `x`:     input data
- `y`:     observed data
- `l_seq`: sequence length (length of sliding window)

**Returns:**
- `x_seqs`: vector of vector of vector of values, i.e., each sequence is a vector of feature vectors
- `y_seqs`: vector of vector of values, i.e., each sequence is a vector of scalar targets
"""
function chunk_data(x, y, l_seq::Int)

    nx = size(x,1)
    ns = floor(Int,nx/l_seq)

    mod(nx,l_seq) == 0 || @info("data was not trimmed for l_seq = $l_seq, may result in worse performance")

    x_seqs = [[convert.(Float32,x[(j-1)*l_seq+i,:]) for i = 1:l_seq]  for j = 1:ns]
    y_seqs = [ convert.(Float32,y[(j-1)*l_seq       .+    (1:l_seq)]) for j = 1:ns]

    return (x_seqs, y_seqs)
end # function chunk_data

"""
    predict_rnn_full(m, x)

Apply model `m` to full sequence of inputs `x`.

**Arguments:**
- `m`: recurrent neural network model
- `x`: input sequence (of features)

**Returns:**
- `y_hat`: vector of scalar targets for each input in `x`
"""
function predict_rnn_full(m, x)
    # reset hidden state (just once)
    Flux.reset!(m)

    # apply model to sequence and convert output to vector
    y_hat = [yi[1] for yi in m.([convert.(Float32,x[i,:]) for i in axes(x,1)])]

    return (y_hat)
end # function predict_rnn_full

"""
    predict_rnn_windowed(m, x, l_seq)

Apply model `m` to inputs by sliding window of size `l_seq` along `x`.

**Arguments:**
- `m`:     recurrent neural network model
- `x`:     input sequence (of features)
- `l_seq`: sequence length (length of sliding window)

**Returns:**
- `y_hat`: vector of scalar targets for each input in `x`
"""
function predict_rnn_windowed(m, x, l_seq)
    nx = size(x,1)
    y_hat = zeros(Float32,nx)

    # assume l_seq = 4
    # 1 2 3 4 5 6 7 8
    #     i     j

    for j = 1:nx
        
        # create window
        i = j < l_seq ? 1 : j - l_seq + 1

        x_seq = [convert.(Float32,x[k,:]) for k = i:j]

        # apply model
        Flux.reset!(m)
        y_seq = m.(x_seq)

        # store last output value in window
        y_hat[j] = y_seq[end][1]
    end

    return (y_hat)
end # function predict_rnn_windowed

"""
    krr(x_train, y_train, x_test, y_test;
        k=PolynomialKernel(;degree=1), λ=0.5)

Kernel ridge regression (KRR).

**Arguments:**
- `x_train`: training input data
- `y_train`: training observed data
- `x_test`:  testing input data
- `y_test`:  testing observed data
- `k`:       (optional) kernel
- `λ`:       (optional) ridge parameter

**Returns:**
- `y_train_hat`: training predicted data
- `y_test_hat`:  testing  predicted data
"""
function krr(x_train, y_train, x_test, y_test;
             k=PolynomialKernel(;degree=1), λ=0.5)

    K  = kernelmatrix(k,x_train')
    K★ = kernelmatrix(k,x_test',x_train')
    kt = (K + λ*I) \ y_train

    y_train_hat = K  * kt
    y_test_hat  = K★ * kt

    # w = x_train' * ((K + λ*I) \ y_train)
    # return (w)
    return (y_train_hat,y_test_hat)
end # function krr

"""
    predict_shapley(m, x::DataFrame)

Internal helper function that wraps a neural network model to create a 
DataFrame of predictions.

**Arguments:**
- `m`: neural network model
- `x`: input data DataFrame

**Returns:**
- `y_hat`: 1-column DataFrame of predictions from `m`
"""
function predict_shapley(m, x::DataFrame)
    DataFrame(y_hat=vec(m(collect(Matrix(x)'))))
end # function predict_shapley

"""
    eval_shapley(m, x, features::Vector{Symbol},
                 N::Int      = min(10000,size(x,1)),
                 num_mc::Int = 10)

Compute stochastic Shapley effects for global feature importance.

Reference: https://nredell.github.io/ShapML.jl/dev/#Examples-1

**Arguments:**
- `m`:        neural network model
- `x`:        input data
- `features`: full list of features in `x`
- `N`:        (optional) number of samples (instances) to use for explanation
- `num_mc`:   (optional) number of Monte Carlo simulations

**Returns:**
- `df_shap`: DataFrame of Shapley effects
- `baseline_shap`: intercept of Shapley effects
"""
function eval_shapley(m, x, features::Vector{Symbol},
                      N::Int      = min(10000,size(x,1)),
                      num_mc::Int = 10)

    seed!(2) # for reproducibility

    # put input data into DataFrame
    df = DataFrame(x,features)

    # compute stochastic Shapley values
    df_temp = shap(explain          = df[shuffle(axes(df,1))[1:N],:],
                   reference        = df,
                   model            = m,
                   predict_function = predict_shapley,
                   sample_size      = num_mc,
                   seed             = 2)

    # setup output DataFrame
    df_shap = combine(groupby(df_temp, [:feature_name]),
              :shap_effect => (x -> mean(abs.(x))) => :mean_effect)
    df_shap = sort(df_shap, order(:mean_effect, rev=true))
    baseline_shap = round(df_temp.intercept[1], digits=2)

    return (df_shap, baseline_shap)
end # function eval_shapley

"""
    plot_shapley(df_shap, baseline_shap,
                 range_shap::AbstractUnitRange=axes(df_shap,1);
                 title::String = "features \$range_shap",
                 dpi::Int      = 200)

Plot horizontal bar graph of feature importance (Shapley effects).

**Arguments:**
- `df_shap`:       DataFrame of Shapley effects
- `baseline_shap`: intercept of Shapley effects
- `range_shap`:    (optional) range of Shapley effects to plot (limit to length ~20)
- `dpi`:           (optional) dots per inch (image resolution)
- `title`:         (optional) plot title

**Returns:**
- `p1`: plot of Shapley effects
"""
function plot_shapley(df_shap, baseline_shap,
                      range_shap::AbstractUnitRange=axes(df_shap,1);
                      title::String = "features $range_shap",
                      dpi::Int      = 200)

    # print warning about too many features
    s = "range_shap length of $(length(range_shap)) may produce congested plot"
    length(range_shap) > 20 && @info(s)

    # get data & axis labels
    x    = df_shap.mean_effect[range_shap]
    y    = df_shap.feature_name[range_shap]
    xlab = "|Shapley effect| (baseline = $baseline_shap)"
    ylab = "feature"
    # title = "Feature Importance - Mean Absolute Shapley Value"

    # plot horizontal bar graph
    p1 = bar(x,yticks=(eachindex(range_shap),y),lab=false,
             dpi=dpi,xlab=xlab,ylab=ylab,title=title,
             orientation=:h,yflip=true,margin=4*mm)

    return (p1)
end # function plot_shapley

"""
    eval_gsa(m, x, N::Int=min(10000,size(x,1)))

Global sensitivity analysis (GSA) with the Morris Method. 
Reference: https://mitmath.github.io/18337/lecture17/global_sensitivity

Reference: https://gsa.sciml.ai/stable/methods/morris/

**Arguments:**
- `m`: neural network model
- `x`: input data
- `N`: (optional) number of samples (instances) to use for explanation

**Returns:**
- `means`: means of elementary effects
"""
function eval_gsa(m, x, N::Int=min(10000,size(x,1)))

    seed!(2) # for reproducibility

    method = Morris(num_trajectory=N,relative_scale=true)
    means  = vec(gsa(m,method,vec(extrema(x,dims=1));N=N).means)
    
    return (means)
end # function eval_gsa
