"""
    dn2dlat(dn, lat)

Convert north-south position (northing) difference to latitude difference.

**Arguments:**
- `dn`:  north-south position (northing) difference [m]
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
- `de`:  east-west position (easting) difference [m]
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
- `lat`:  nominal latitude [rad]

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
- `lat`:  nominal latitude [rad]

**Returns:**
- `de`: east-west position (easting) difference [m]
"""
function dlon2de(dlon, lat)
    de = dlon / sqrt(1-(e_earth*sin(lat))^2) * r_earth * cos(lat)
    return (de)
end # function dlon2de

"""
    linreg(y, x; λ=0)

Linear regression with data matrix.

**Arguments:**
- `y`: length-`N` observed data vector
- `x`: `N` x `Nf` input data matrix (`Nf` is number of features)
- `λ`: (optional) ridge parameter

**Returns:**
- `coef`: linear regression coefficients
"""
function linreg(y, x; λ=0)
    (x'*x + λ*I) \ (x'*y)
end # function linreg

"""
    linreg(y; λ=0)

Linear regression to determine best fit line for x = eachindex(y).

**Arguments:**
- `y`: length-`N` observed data vector
- `λ`: (optional) ridge parameter

**Returns:**
- `coef`: length-`2` vector of linear regression coefficients
"""
function linreg(y; λ=0)
    x    = [one.(y) eachindex(y)]
    coef = linreg(y,x;λ=λ)
    return (coef)
end # function linreg

"""
    detrend(y, x=[eachindex(y);]; λ=0, mean_only::Bool=false)

Detrend signal (remove mean and optionally slope).

**Arguments:**
- `y`:         length-`N` observed data vector
- `x`:         (optional) `N` x `Nf` input data matrix (`Nf` is number of features)
- `λ`:         (optional) ridge parameter
- `mean_only`: (optional) if true, only remove mean (not slope)

**Returns:**
- `y`: length-`N` observed data vector, detrended
"""
function detrend(y, x=[eachindex(y);]; λ=0, mean_only::Bool=false)
    if mean_only
        y = y .- mean(y)
    else
        x    = [one.(y) x]
        coef = linreg(y,x;λ=λ)
        y    = y - x*coef
    end
    return (y)
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
        error("$pass1 & $pass2 passband frequencies are invalid")
    end
    return (digitalfilter(p,Butterworth(pole)))
end # function get_bpf

"""
    bpf_data(x::Matrix; bpf=get_bpf())

Bandpass (or low-pass or high-pass) filter columns of matrix.

**Arguments:**
- `x`:   data matrix (e.g., Tolles-Lawson `A` matrix)
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
- `x`:   data vector (e.g., magnetometer measurements)
- `bpf`: (optional) filter object

**Returns:**
- `x_f`: data vector, filtered
"""
function bpf_data(x::Vector; bpf=get_bpf())
    filtfilt(bpf,x)
end # function bpf_data

"""
    bpf_data!(x; bpf=get_bpf())

Bandpass (or low-pass or high-pass) filter vector or columns of matrix.

**Arguments:**
- `x`:   data vector or matrix
- `bpf`: (optional) filter object

**Returns:**
- `nothing`: `x` is mutated with filtered data
"""
function bpf_data!(x; bpf=get_bpf())
    x .= bpf_data(x;bpf=bpf)
    return (nothing)
end # function bpf_data!

"""
    downsample(x, Nmax::Int=1000)

Downsample data `x` to `Nmax` (or fewer) data points.

**Arguments:**
- `x`:    data vector or matrix
- `Nmax`: (optional) maximum number of data points

**Returns:**
- `x`: vector or matrix of downsampled data
"""
function downsample(x, Nmax::Int=1000)
    N = size(x,1)
    N > Nmax && (x = x[1:ceil(Int,N/Nmax):N,:])
    size(x,2) == 1 && (x = vec(x))
    return (x)
end # function downsample

"""
    get_x(xyz::XYZ, ind = trues(xyz.traj.N),
          features_setup::Vector{Symbol}   = [:mag_1_uc,:TL_A_flux_a];
          features_no_norm::Vector{Symbol} = Symbol[],
          terms             = [:permanent,:induced,:eddy],
          sub_diurnal::Bool = false,
          sub_igrf::Bool    = false,
          bpf_mag::Bool     = false)

Get `x` data matrix.

**Arguments:**
- `xyz`:              `XYZ` flight data struct
- `ind`:              selected data indices
- `features_setup`:   vector of features to include
- `features_no_norm`: (optional) vector of features to not normalize
- `terms`:            (optional) Tolles-Lawson terms to use {`:permanent`,`:induced`,`:eddy`,`:bias`}
- `sub_diurnal`:      (optional) if true, subtract diurnal from scalar magnetometer measurements
- `sub_igrf`:         (optional) if true, subtract IGRF from scalar magnetometer measurements
- `bpf_mag`:          (optional) if true, bpf scalar magnetometer measurements

**Returns:**
- `x`:        ` data matrix (`Nf` is number of features)
- `no_norm`:  length-`Nf` Boolean indices of features to not be normalized
- `features`: length-`Nf` feature vector (including components of TL `A`, etc.)
"""
function get_x(xyz::XYZ, ind = trues(xyz.traj.N),
               features_setup::Vector{Symbol}   = [:mag_1_uc,:TL_A_flux_a];
               features_no_norm::Vector{Symbol} = Symbol[],
               terms             = [:permanent,:induced,:eddy],
               sub_diurnal::Bool = false,
               sub_igrf::Bool    = false,
               bpf_mag::Bool     = false)

    N        = length(xyz.traj.lat[ind])
    d        = Dict{Symbol,Array{Float64}}()
    x        = Matrix{eltype(xyz.traj.lat)}(undef,N,0)
    no_norm  = Vector{Bool}(undef,0)
    features = Vector{Symbol}(undef,0)

    @assert N > 2 "ind must contain at least 3 data points"

    for use_vec in field_check(xyz,MagV)
        A = create_TL_A(getfield(xyz,use_vec),ind;terms=terms)
        push!(d,Symbol("TL_A_",use_vec)=>A)
    end

    # subtract diurnal and/or IGRF as specified
    sub = zeros(eltype(xyz.traj.lat),N)
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

    # 4th derivative central difference
    # Reference: Loughlin Tuck, Characterization and compensation of magnetic
    # interference resulting from unmanned aircraft systems, 2019. (pg. 28)
    for mag in mags_all
        lab = Symbol(mag,"_dot4")
        val = fdm(getfield(xyz,mag)[ind] - sub;scheme=:fourth)
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
    push!(d,:dcm=>[reshape(euler2dcm(roll,pitch,yaw,:nav2body),(9,N))';]) # ordered this way initially, leaving for now consistency
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
        rpy == :roll  && (rpy_ = roll)
        rpy == :pitch && (rpy_ = pitch)
        rpy == :yaw   && (rpy_ = yaw)
        push!(d,Symbol(rpy,"_fdm")=>fdm(rpy_))
        push!(d,Symbol(rpy,"_sin")=>sin.(rpy_))
        push!(d,Symbol(rpy,"_cos")=>cos.(rpy_))
        push!(d,Symbol(rpy,"_sin_fdm")=>fdm(sin.(rpy_)))
        push!(d,Symbol(rpy,"_cos_fdm")=>fdm(cos.(rpy_)))
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
            error("$f feature is invalid, remove")
        end

        Nf = size(u,2)
        v  = f in features_no_norm ? trues(Nf) : falses(Nf)
        w  = Nf > 1 ? [Symbol(f,"_",i) for i = 1:Nf] : f

        if isnan(sum(u))
            error("$f feature contains NaNs, remove")
        else
            x        = [x         u]
            no_norm  = [no_norm;  v]
            features = [features; w]
        end
    end

    return (x, no_norm, features)
end # function get_x

"""
    get_x(xyz_vec::Vector{XYZ}, ind_vec,
          features_setup::Vector{Symbol}   = [:mag_1_uc,:TL_A_flux_a];
          features_no_norm::Vector{Symbol} = Symbol[],
          terms             = [:permanent,:induced,:eddy],
          sub_diurnal::Bool = false,
          sub_igrf::Bool    = false,
          bpf_mag::Bool     = false)

Get `x` data matrix from multiple `XYZ` flight data structs.

**Arguments:**
- `xyz_vec`:          vector of `XYZ` flight data structs
- `ind_vec`:          vector of selected data indices
- `features_setup`:   vector of features to include
- `features_no_norm`: (optional) vector of features to not normalize
- `terms`:            (optional) Tolles-Lawson terms to use {`:permanent`,`:induced`,`:eddy`,`:bias`}
- `sub_diurnal`:      (optional) if true, subtract diurnal from scalar magnetometer measurements
- `sub_igrf`:         (optional) if true, subtract IGRF from scalar magnetometer measurements
- `bpf_mag`:          (optional) if true, bpf scalar magnetometer measurements

**Returns:**
- `x`:        `N` x `Nf` data matrix (`Nf` is number of features)
- `no_norm`:  length-`Nf` Boolean indices of features to not be normalized
- `features`: length-`Nf` feature vector (including components of TL `A`, etc.)
"""
function get_x(xyz_vec::Vector, ind_vec::Vector,
               features_setup::Vector{Symbol}   = [:mag_1_uc,:TL_A_flux_a];
               features_no_norm::Vector{Symbol} = Symbol[],
               terms             = [:permanent,:induced,:eddy],
               sub_diurnal::Bool = false,
               sub_igrf::Bool    = false,
               bpf_mag::Bool     = false)

    (x,no_norm,features) = get_x(xyz_vec[1],ind_vec[1],features_setup;
                                 features_no_norm = features_no_norm,
                                 terms            = terms,
                                 sub_diurnal      = sub_diurnal,
                                 sub_igrf         = sub_igrf,
                                 bpf_mag          = bpf_mag)

    for (xyz,ind) in zip(xyz_vec[2:end],ind_vec[2:end])
        x = vcat(x,get_x(xyz,ind,features_setup;
                         features_no_norm = features_no_norm,
                         terms            = terms,
                         sub_diurnal      = sub_diurnal,
                         sub_igrf         = sub_igrf,
                         bpf_mag          = bpf_mag)[1])
    end

    return (x, no_norm, features)
end # function get_x

"""
    get_x(lines, df_line::DataFrame, df_flight::DataFrame,
          features_setup::Vector{Symbol}   = [:mag_1_uc,:TL_A_flux_a];
          features_no_norm::Vector{Symbol} = Symbol[],
          terms              = [:permanent,:induced,:eddy],
          sub_diurnal::Bool  = false,
          sub_igrf::Bool     = false,
          bpf_mag::Bool      = false,
          reorient_vec::Bool = false,
          l_window::Int      = -1,
          silent::Bool       = true)

Get `x` data matrix from multiple flight lines, possibly multiple flights.

**Arguments:**
- `lines`:            selected line number(s)
- `df_line`:          lookup table (DataFrame) of `lines`
|**Field**|**Type**|**Description**
|:--|:--|:--
`flight`   |`Symbol`| flight name (e.g., `:Flt1001`)
`line`     |`Real`  | line number, i.e., segments within `flight`
`t_start`  |`Real`  | start time of `line` to use [s]
`t_end`    |`Real`  | end   time of `line` to use [s]
`full_line`|`Bool`  | (optional) if true, `t_start` to `t_end` is full line
`map_name` |`Symbol`| (optional) name of magnetic anomaly map relevant to `line`
`map_type` |`Symbol`| (optional) type of magnetic anomaly map for `map_name` {`drape`,`HAE`}
`line_alt` |`Real`  | (optional) nominal altitude of `line` [m]
- `df_flight`:        lookup table (DataFrame) of flight data HDF5 files
|**Field**|**Type**|**Description**
|:--|:--|:--
`flight`  |`Symbol`| flight name (e.g., `:Flt1001`)
`xyz_type`|`Symbol`| subtype of `XYZ` to use for flight data {`:XYZ0`,`:XYZ1`,`:XYZ20`,`:XYZ21`}
`xyz_set` |`Real`  | flight dataset number (used to prevent improper mixing of datasets, such as different magnetometer locations)
`xyz_h5`  |`String`| path/name of flight data HDF5 file (`.h5` extension optional)
- `features_setup`:   vector of features to include
- `features_no_norm`: (optional) vector of features to not normalize
- `terms`:            (optional) Tolles-Lawson terms to use {`:permanent`,`:induced`,`:eddy`,`:bias`}
- `sub_diurnal`:      (optional) if true, subtract diurnal from scalar magnetometer measurements
- `sub_igrf`:         (optional) if true, subtract IGRF from scalar magnetometer measurements
- `bpf_mag`:          (optional) if true, bpf scalar magnetometer measurements
- `reorient_vec`:     (optional) if true, align vector magnetometer measurements with body frame
- `l_window`:         (optional) trim data by `N % l_window`, `-1` to ignore
- `silent`:           (optional) if true, no print outs

**Returns:**
- `x`:        `N` x `Nf` data matrix (`Nf` is number of features)
- `no_norm`:  length-`Nf` Boolean indices of features to not be normalized
- `features`: length-`Nf` feature vector (including components of TL `A`, etc.)
- `l_segs`:   length-`N_lines` vector of lengths of `lines`, sum(l_segs) = `N`
"""
function get_x(lines, df_line::DataFrame, df_flight::DataFrame,
               features_setup::Vector{Symbol}   = [:mag_1_uc,:TL_A_flux_a];
               features_no_norm::Vector{Symbol} = Symbol[],
               terms              = [:permanent,:induced,:eddy],
               sub_diurnal::Bool  = false,
               sub_igrf::Bool     = false,
               bpf_mag::Bool      = false,
               reorient_vec::Bool = false,
               l_window::Int      = -1,
               silent::Bool       = true)

    # check if lines are in df_line, remove if not
    for l in lines
        if !(l in df_line.line)
            silent || @info("line $l is not in df_line, skipping")
            lines = lines[lines.!=l]
        end
    end

    # check if lines are duplicated in df_line, throw error if so
    @assert length(lines) == length(unique(lines)) "df_line contains duplicated lines"

    # check if flight data matches up
    flights  = [df_line.flight[df_line.line .== l][1] for l in lines]
    xyz_sets = [df_flight.xyz_set[df_flight.flight .== f][1] for f in flights]
    @assert all(xyz_sets.==xyz_sets[1]) "incompatible xyz_sets in df_flight"

    # initial values
    flt_old  = :FltInitial
    x        = nothing
    no_norm  = nothing
    features = nothing
    xyz      = nothing
    l_segs   = zeros(Int,length(lines))

    for line in lines
        flt = df_line.flight[df_line.line .== line][1]
        if flt != flt_old
            xyz = get_XYZ(flt,df_flight;reorient_vec=reorient_vec,silent=silent)
        end
        flt_old = flt

        ind = get_ind(xyz,line,df_line;l_window=l_window)
        l_segs[findfirst(l_segs .== 0)] = length(xyz.traj.lat[ind])

        if x isa Nothing
            (x,no_norm,features) = get_x(xyz,ind,features_setup;
                                         features_no_norm = features_no_norm,
                                         terms            = terms,
                                         sub_diurnal      = sub_diurnal,
                                         sub_igrf         = sub_igrf,
                                         bpf_mag          = bpf_mag)
        else
            x = vcat(x,get_x(xyz,ind,features_setup;
                             features_no_norm = features_no_norm,
                             terms            = terms,
                             sub_diurnal      = sub_diurnal,
                             sub_igrf         = sub_igrf,
                             bpf_mag          = bpf_mag)[1])
        end
    end

    return (x, no_norm, features, l_segs)
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
- `map_val`: (optional) scalar magnetic anomaly map values, only used for `y_type = :b`
- `y_type`:  (optional) `y` target type
    - `:a` = anomaly field  #1, compensated tail stinger total field scalar magnetometer measurements
    - `:b` = anomaly field  #2, interpolated `magnetic anomaly map` values
    - `:c` = aircraft field #1, difference between uncompensated cabin total field scalar magnetometer measurements and interpolated `magnetic anomaly map` values
    - `:d` = aircraft field #2, difference between uncompensated cabin and compensated tail stinger total field scalar magnetometer measurements
    - `:e` = BPF'd total field, bandpass filtered uncompensated cabin total field scalar magnetometer measurements
- `use_mag`:     (optional) uncompensated scalar magnetometer to use for `y` target vector {`:mag_1_uc`, etc.}, only used for `y_type = :c, :d, :e`
- `use_mag_c`:   (optional) compensated scalar magnetometer to use for `y` target vector {`:mag_1_c`, etc.}, only used for `y_type = :a, :d`
- `sub_diurnal`: (optional) if true, subtract diurnal from scalar magnetometer measurements
- `sub_igrf`:    (optional) if true, subtract IGRF from scalar magnetometer measurements

**Returns:**
- `y`: length-`N` target vector
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
        if getfield(xyz,use_mag) isa MagV
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
        y = mag_c - sub
    elseif y_type == :b # option B
        y = map_val
    elseif y_type == :c # option C
        y = mag_uc - sub - map_val
    elseif y_type == :d # option D
        y = mag_uc - mag_c
    elseif y_type == :e # option E
        fs   = 1 / xyz.traj.dt
        y = bpf_data(mag_uc - sub; bpf=get_bpf(;fs=fs))
    else
        error("$y_type target type is invalid")
    end

    return (y)
end # function get_y

"""
    get_y(lines, df_line::DataFrame, df_flight::DataFrame,
          df_map::DataFrame;
          y_type::Symbol    = :d,
          use_mag::Symbol   = :mag_1_uc,
          use_mag_c::Symbol = :mag_1_c,
          sub_diurnal::Bool = false,
          sub_igrf::Bool    = false,
          l_window::Int     = -1,
          silent::Bool      = true)

Get `y` target vector from multiple flight lines, possibly multiple flights.

**Arguments:**
- `lines`:     selected line number(s)
- `df_line`:   lookup table (DataFrame) of `lines`
|**Field**|**Type**|**Description**
|:--|:--|:--
`flight`   |`Symbol`| flight name (e.g., `:Flt1001`)
`line`     |`Real`  | line number, i.e., segments within `flight`
`t_start`  |`Real`  | start time of `line` to use [s]
`t_end`    |`Real`  | end   time of `line` to use [s]
`full_line`|`Bool`  | (optional) if true, `t_start` to `t_end` is full line
`map_name` |`Symbol`| (optional) name of magnetic anomaly map relevant to `line`, only used for `y_type = :b, :c`
`map_type` |`Symbol`| (optional) type of magnetic anomaly map for `map_name` {`drape`,`HAE`}
`line_alt` |`Real`  | (optional) nominal altitude of `line` [m]
- `df_flight`: lookup table (DataFrame) of flight data HDF5 files
|**Field**|**Type**|**Description**
|:--|:--|:--
`flight`  |`Symbol`| flight name (e.g., `:Flt1001`)
`xyz_type`|`Symbol`| subtype of `XYZ` to use for flight data {`:XYZ0`,`:XYZ1`,`:XYZ20`,`:XYZ21`}
`xyz_set` |`Real`  | flight dataset number (used to prevent improper mixing of datasets, such as different magnetometer locations)
`xyz_h5`  |`String`| path/name of flight data HDF5 file (`.h5` extension optional)
- `df_map`:    lookup table (DataFrame) of map data HDF5 files
|**Field**|**Type**|**Description**
|:--|:--|:--
`map_name`|`Symbol`| name of magnetic anomaly map
`map_h5`  |`String`| path/name of map data HDF5 file (`.h5` extension optional)
`map_type`|`Symbol`| (optional) type of magnetic anomaly map {`drape`,`HAE`}
`map_alt` |`Real`  | (optional) map altitude, -1 for drape map [m]
- `y_type`:    (optional) `y` target type
    - `:a` = anomaly field  #1, compensated tail stinger total field scalar magnetometer measurements
    - `:b` = anomaly field  #2, interpolated `magnetic anomaly map` values
    - `:c` = aircraft field #1, difference between uncompensated cabin total field scalar magnetometer measurements and interpolated `magnetic anomaly map` values
    - `:d` = aircraft field #2, difference between uncompensated cabin and compensated tail stinger total field scalar magnetometer measurements
    - `:e` = BPF'd total field, bandpass filtered uncompensated cabin total field scalar magnetometer measurements
- `use_mag`:     (optional) uncompensated scalar magnetometer to use for `y` target vector {`:mag_1_uc`, etc.}, only used for `y_type = :c, :d, :e`
- `use_mag_c`:   (optional) compensated scalar magnetometer to use for `y` target vector {`:mag_1_c`, etc.}, only used for `y_type = :a, :d`
- `sub_diurnal`: (optional) if true, subtract diurnal from scalar magnetometer measurements
- `sub_igrf`:    (optional) if true, subtract IGRF from scalar magnetometer measurements
- `l_window`:    (optional) trim data by `N % l_window`, `-1` to ignore
- `silent`:      (optional) if true, no print outs

**Returns:**
- `y`: length-`N` target vector
"""
function get_y(lines, df_line::DataFrame, df_flight::DataFrame,
               df_map::DataFrame;
               y_type::Symbol    = :d,
               use_mag::Symbol   = :mag_1_uc,
               use_mag_c::Symbol = :mag_1_c,
               sub_diurnal::Bool = false,
               sub_igrf::Bool    = false,
               l_window::Int     = -1,
               silent::Bool      = true)

    # check if lines are in df_line, remove if not
    for l in lines
        if !(l in df_line.line)
            silent || @info("line $l is not in df_line, skipping")
            lines = lines[lines.!=l]
        end
    end

    # check if lines are duplicated in df_line, throw error if so
    @assert length(lines) == length(unique(lines)) "df_line contains duplicated lines"

    # initial values
    flt_old = :FltInitial
    y       = nothing
    xyz     = nothing

    for line in lines
        flt = df_line.flight[df_line.line .== line][1]
        if flt != flt_old
            xyz = get_XYZ(flt,df_flight;silent=silent)
        end
        flt_old = flt

        ind = get_ind(xyz,line,df_line;l_window=l_window)

        # map values along trajectory (if needed)
        if y_type in [:b,:c]
            map_name = df_line.map_name[df_line.line .== line][1]
            map_val  = get_map_val(get_map(map_name,df_map),xyz.traj,ind;α=200)
        else
            map_val  = -1
        end

        if y isa Nothing
            y = get_y(xyz,ind,map_val;
                      y_type      = y_type,
                      use_mag     = use_mag,
                      use_mag_c   = use_mag_c,
                      sub_diurnal = sub_diurnal,
                      sub_igrf    = sub_igrf)
        else
            y = vcat(y,get_y(xyz,ind,map_val;
                             y_type      = y_type,
                             use_mag     = use_mag,
                             use_mag_c   = use_mag_c,
                             sub_diurnal = sub_diurnal,
                             sub_igrf    = sub_igrf))
        end
    end

    return (y)
end # function get_y

"""
    get_Axy(lines, df_line::DataFrame,
            df_flight::DataFrame, df_map::DataFrame,
            features_setup::Vector{Symbol}   = [:mag_1_uc,:TL_A_flux_a];
            features_no_norm::Vector{Symbol} = Symbol[],
            y_type::Symbol     = :d,
            use_mag::Symbol    = :mag_1_uc,
            use_mag_c::Symbol  = :mag_1_c,
            use_vec::Symbol    = :flux_a,
            terms              = [:permanent,:induced,:eddy],
            terms_A            = [:permanent,:induced,:eddy,:bias],
            sub_diurnal::Bool  = false,
            sub_igrf::Bool     = false,
            bpf_mag::Bool      = false,
            reorient_vec::Bool = false,
            l_window::Int      = -1,
            mod_TL::Bool       = false,
            map_TL::Bool       = false,
            return_B::Bool     = false,
            silent::Bool       = true)

Get "external" Tolles-Lawson `A` matrix, `x` data matrix, and `y` target vector
from multiple flight lines, possibly multiple flights. Optionally return `Bt`
and `B_dot` used to create the "external" Tolles-Lawson `A` matrix.

**Arguments:**
- `lines`:     selected line number(s)
- `df_line`:   lookup table (DataFrame) of `lines`
|**Field**|**Type**|**Description**
|:--|:--|:--
`flight`   |`Symbol`| flight name (e.g., `:Flt1001`)
`line`     |`Real`  | line number, i.e., segments within `flight`
`t_start`  |`Real`  | start time of `line` to use [s]
`t_end`    |`Real`  | end   time of `line` to use [s]
`full_line`|`Bool`  | (optional) if true, `t_start` to `t_end` is full line
`map_name` |`Symbol`| (optional) name of magnetic anomaly map relevant to `line`, only used for `y_type = :b, :c`
`map_type` |`Symbol`| (optional) type of magnetic anomaly map for `map_name` {`drape`,`HAE`}
`line_alt` |`Real`  | (optional) nominal altitude of `line` [m]
- `df_flight`: lookup table (DataFrame) of flight data HDF5 files
|**Field**|**Type**|**Description**
|:--|:--|:--
`flight`  |`Symbol`| flight name (e.g., `:Flt1001`)
`xyz_type`|`Symbol`| subtype of `XYZ` to use for flight data {`:XYZ0`,`:XYZ1`,`:XYZ20`,`:XYZ21`}
`xyz_set` |`Real`  | flight dataset number (used to prevent improper mixing of datasets, such as different magnetometer locations)
`xyz_h5`  |`String`| path/name of flight data HDF5 file (`.h5` extension optional)
- `df_map`:    lookup table (DataFrame) of map data HDF5 files
|**Field**|**Type**|**Description**
|:--|:--|:--
`map_name`|`Symbol`| name of magnetic anomaly map
`map_h5`  |`String`| path/name of map data HDF5 file (`.h5` extension optional)
`map_type`|`Symbol`| (optional) type of magnetic anomaly map {`drape`,`HAE`}
`map_alt` |`Real`  | (optional) map altitude, -1 for drape map [m]
- `features_setup`:   vector of features to include
- `features_no_norm`: (optional) vector of features to not normalize
- `y_type`:           (optional) `y` target type
    - `:a` = anomaly field  #1, compensated tail stinger total field scalar magnetometer measurements
    - `:b` = anomaly field  #2, interpolated `magnetic anomaly map` values
    - `:c` = aircraft field #1, difference between uncompensated cabin total field scalar magnetometer measurements and interpolated `magnetic anomaly map` values
    - `:d` = aircraft field #2, difference between uncompensated cabin and compensated tail stinger total field scalar magnetometer measurements
    - `:e` = BPF'd total field, bandpass filtered uncompensated cabin total field scalar magnetometer measurements
- `use_mag`:      (optional) uncompensated scalar magnetometer to use for `y` target vector {`:mag_1_uc`, etc.}, only used for `y_type = :c, :d, :e`
- `use_mag_c`:    (optional) compensated scalar magnetometer to use for `y` target vector {`:mag_1_c`, etc.}, only used for `y_type = :a, :d`
- `use_vec`:      (optional) vector magnetometer (fluxgate) to use for "external" Tolles-Lawson `A` matrix {`:flux_a`, etc.}
- `terms`:        (optional) Tolles-Lawson terms to use for `A` within `x` data matrix {`:permanent`,`:induced`,`:eddy`,`:bias`}
- `terms_A`:      (optional) Tolles-Lawson terms to use for "external" Tolles-Lawson `A` matrix {`:permanent`,`:induced`,`:eddy`,`:bias`}
- `sub_diurnal`:  (optional) if true, subtract diurnal from scalar magnetometer measurements
- `sub_igrf`:     (optional) if true, subtract IGRF from scalar magnetometer measurements
- `bpf_mag`:      (optional) if true, bpf scalar magnetometer measurements in `x` data matrix
- `reorient_vec`: (optional) if true, align vector magnetometer measurements with body frame
- `l_window`:     (optional) trim data by `N % l_window`, `-1` to ignore
- `mod_TL`:       (optional) if true, create modified  "external" Tolles-Lawson `A` matrix with `use_mag`
- `map_TL`:       (optional) if true, create map-based "external" Tolles-Lawson `A` matrix
- `return_B`:     (optional) if true, also return `Bt` & `B_dot`
- `silent`:       (optional) if true, no print outs

**Returns:**
- `A`:        `N` x `N_TL` "external" Tolles-Lawson `A` matrix (`N_TL` is number of Tolles-Lawson coefficients)
- `x`:        `N` x `Nf` data matrix (`Nf` is number of features)
- `y`:        length-`N` target vector
- `no_norm`:  length-`Nf` Boolean indices of features to not be normalized
- `features`: length-`Nf` feature vector (including components of TL `A`, etc.)
- `l_segs`:   length-`N_lines` vector of lengths of `lines`, sum(l_segs) = `N`
- `Bt`:       if `return_B = true`, length-`N` magnitude of total field measurements used to create `A` [nT]
- `B_dot`:    if `return_B = true`, `N` x `3` finite differences of total field vector used to create `A` [nT]
"""
function get_Axy(lines, df_line::DataFrame,
                 df_flight::DataFrame, df_map::DataFrame,
                 features_setup::Vector{Symbol}   = [:mag_1_uc,:TL_A_flux_a];
                 features_no_norm::Vector{Symbol} = Symbol[],
                 y_type::Symbol     = :d,
                 use_mag::Symbol    = :mag_1_uc,
                 use_mag_c::Symbol  = :mag_1_c,
                 use_vec::Symbol    = :flux_a,
                 terms              = [:permanent,:induced,:eddy],
                 terms_A            = [:permanent,:induced,:eddy,:bias],
                 sub_diurnal::Bool  = false,
                 sub_igrf::Bool     = false,
                 bpf_mag::Bool      = false,
                 reorient_vec::Bool = false,
                 l_window::Int      = -1,
                 mod_TL::Bool       = false,
                 map_TL::Bool       = false,
                 return_B::Bool     = false,
                 silent::Bool       = true)

    # check if lines are in df_line, remove if not
    for l in lines
        if !(l in df_line.line)
            silent || @info("line $l is not in df_line, skipping")
            lines = lines[lines.!=l]
        end
    end

    # check if lines are duplicated in df_line, throw error if so
    @assert length(lines) == length(unique(lines)) "df_line contains duplicated lines"

    # check if flight data matches up, throw error if not
    flights  = [df_line.flight[df_line.line .== l][1] for l in lines]
    xyz_sets = [df_flight.xyz_set[df_flight.flight .== f][1] for f in flights]
    @assert all(xyz_sets.==xyz_sets[1]) "incompatible xyz_sets in df_flight"

    # initial values
    flt_old  = :FltInitial
    A_test   = create_TL_A([1.0],[1.0],[1.0];terms=terms_A)
    A        = Matrix{eltype(A_test)}(undef,0,length(A_test))
    Bt       = Vector{eltype(A_test)}(undef,0)
    B_dot    = Matrix{eltype(A_test)}(undef,0,3)
    x        = nothing
    y        = nothing
    no_norm  = nothing
    features = nothing
    xyz      = nothing
    l_segs   = zeros(Int,length(lines))

    for line in lines
        flt = df_line.flight[df_line.line .== line][1]
        if flt != flt_old
            xyz = get_XYZ(flt,df_flight;reorient_vec=reorient_vec,silent=silent)
        end
        flt_old = flt

        ind = get_ind(xyz,line,df_line;l_window=l_window)
        l_segs[findfirst(l_segs .== 0)] = length(xyz.traj.lat[ind])

        # x matrix
        if x isa Nothing
            (x,no_norm,features) = get_x(xyz,ind,features_setup;
                                         features_no_norm = features_no_norm,
                                         terms            = terms,
                                         sub_diurnal      = sub_diurnal,
                                         sub_igrf         = sub_igrf,
                                         bpf_mag          = bpf_mag)
        else
            x = vcat(x,get_x(xyz,ind,features_setup;
                             features_no_norm = features_no_norm,
                             terms            = terms,
                             sub_diurnal      = sub_diurnal,
                             sub_igrf         = sub_igrf,
                             bpf_mag          = bpf_mag)[1])
        end

        # map values along trajectory (if needed)
        if y_type in [:b,:c]
            map_name = df_line.map_name[df_line.line .== line][1]
            map_val  = get_map_val(get_map(map_name,df_map),xyz.traj,ind;α=200)
        else
            map_val  = -1
        end

        # `A` matrix for selected vector magnetometer & `B` measurements
        field_check(xyz,use_vec,MagV)
        if mod_TL
            (A_,Bt_,B_dot_) = create_TL_A(getfield(xyz,use_vec),ind;
                                          Bt       = getfield(xyz,use_mag),
                                          terms    = terms_A,
                                          return_B = true)
        elseif map_TL
            (A_,Bt_,B_dot_) = create_TL_A(getfield(xyz,use_vec),ind;
                                          Bt       = map_val,
                                          terms    = terms_A,
                                          return_B = true)
        else
            (A_,Bt_,B_dot_) = create_TL_A(getfield(xyz,use_vec),ind;
                                          terms    = terms_A,
                                          return_B = true)
        end
        fs = 1 / xyz.traj.dt
        y_type == :e && bpf_data!(A_;bpf=get_bpf(;fs=fs))

        A     = vcat(A,A_)
        Bt    = vcat(Bt,Bt_)
        B_dot = vcat(B_dot,B_dot_)

        # y vector
        if y isa Nothing
            y = get_y(xyz,ind,map_val;
                      y_type      = y_type,
                      use_mag     = use_mag,
                      use_mag_c   = use_mag_c,
                      sub_diurnal = sub_diurnal,
                      sub_igrf    = sub_igrf)
        else
            y = vcat(y,get_y(xyz,ind,map_val;
                             y_type      = y_type,
                             use_mag     = use_mag,
                             use_mag_c   = use_mag_c,
                             sub_diurnal = sub_diurnal,
                             sub_igrf    = sub_igrf))
        end
    end

    if return_B
        return (A, Bt, B_dot, x, y, no_norm, features, l_segs)
    else
        return (A, x, y, no_norm, features, l_segs)
    end
end # function get_Axy

"""
    LPE{T2 <: AbstractFloat}

Learnable positional encoding (LPE) struct used for transformer encoder.

|**Field**|**Type**|**Description**
|:--|:--|:--
|`embeddings`|Matrix{`T2`}| embeddings
|`dropout`   |`Dropout`   | dropout layer
"""
struct LPE{T2 <: AbstractFloat}
    embeddings :: Matrix{T2}
    dropout    :: Dropout
end # struct LPE

@functor LPE

"""
    LPE(N_head::Int, l_window::Int, dropout_prob)

Internal helper function to create learnable positional encoding struct.

**Arguments:**
- `N_head`:       number of attention heads
- `l_window`:     temporal window length
- `dropout_prob`: dropout rate

**Returns:**
- `lpe`: `LPE` learnable positional encoding struct
"""
function LPE(N_head::Int, l_window::Int, dropout_prob)
    embeddings = Float32.(rand(Uniform(-0.02, 0.02), N_head, l_window))
    return LPE(embeddings, Dropout(dropout_prob))
end # function LPE

"""
    (lpe::LPE)(x::AbstractArray)

Internal helper function to operate on learnable positional encoding struct.

**Arguments:**
- `lpe`: `LPE` learnable positional encoding struct
- `x`:   data tensor

**Returns:**
- `x`: data tensor with learnable positional encoding applied
"""
function (lpe::LPE)(x::AbstractArray)
    lpe.dropout(x .+ lpe.embeddings)
end # function LPE

"""
    batchnorm(x::AbstractArray)

Internal helper function to apply batch normalization across the first
dimension of a tensor.

**Arguments**
- `x`: data tensor (e.g., number of features, window length, mini-batch size)

**Returns**
- `x_b`: `x` with batch normalization applied across the first dimension
"""
function batchnorm(x::AbstractArray)
    c   = size(x,1)
    d   = (2,1,3)
    x_b = permutedims(BatchNorm(c)(permutedims(x,d)), d)
    return (x_b)
end # function batchnorm

"""
    get_nn_m(Nf::Int, Ny::Int=1;
             hidden                = [8],
             activation::Function  = swish,
             final_bias::Bool      = true,
             skip_con::Bool        = false,
             model_type::Symbol    = :m1
             l_window::Int         = 5,
             tf_layer_type::Symbol = :postlayer,
             tf_norm_type::Symbol  = :batch,
             dropout_prob          = 0.2,
             N_tf_head::Int        = 8,
             tf_gain               = 1.0)

Get neural network model. Valid for 0-3 hidden layers, except for
`model_type = :m3w, :m3tf`, which are only valid for 1 or 2 hidden layers.

**Arguments:**
- `Nf`:            length of input (feature) layer
- `Ny`:            (optional) length of output layer
- `hidden`:        (optional) hidden layers & nodes (e.g., `[8,8]` for 2 hidden layers, 8 nodes each)
- `activation`:    (optional) activation function
    - `relu`  = rectified linear unit
    - `σ`     = sigmoid (logistic function)
    - `swish` = self-gated
    - `tanh`  = hyperbolic tan
    - run `plot_activation()` for a visual
- `final_bias`:    (optional) if true, include final layer bias
- `skip_con`:      (optional) if true, use skip connections, must have length(`hidden`) == 1
- `model_type`:    (optional) aeromagnetic compensation model type
- `l_window`:      (optional) temporal window length, only used for `model_type = :m3w, :m3tf`
- `tf_layer_type`: (optional) transformer normalization layer before or after skip connection {`:prelayer`,`:postlayer`}, only used for `model_type = :m3tf`
- `tf_norm_type`:  (optional) normalization for transformer encoder {`:batch`,`:layer`,`:none`}, only used for `model_type = :m3tf`
- `dropout_prob`:  (optional) dropout rate, only used for `model_type = :m3w, :m3tf`
- `N_tf_head`:     (optional) number of attention heads, only used for `model_type = :m3tf`
- `tf_gain`:       (optional) weight initialization parameter, only used for `model_type = :m3tf`

**Returns:**
- `m`: neural network model
"""
function get_nn_m(Nf::Int, Ny::Int=1;
                  hidden                = [8],
                  activation::Function  = swish,
                  final_bias::Bool      = true,
                  skip_con::Bool        = false,
                  model_type::Symbol    = :m1,
                  l_window::Int         = 5,
                  tf_layer_type::Symbol = :postlayer,
                  tf_norm_type::Symbol  = :batch,
                  dropout_prob          = 0.2,
                  N_tf_head::Int        = 8,
                  tf_gain               = 1.0)

    l = length(hidden)

    if model_type == :m3tf

        # pre & post layer normalization transformer architectures
        # https://tnq177.github.io/data/transformers_without_tears.pdf

        @assert l > 0 "hidden must have at least 1 element"
        @assert hidden[1] % N_tf_head == 0 "hidden[1] must be divisible by N_tf_head"

        N_head = hidden[1]
        init   = Flux.glorot_uniform(gain=tf_gain)

        if tf_norm_type == :layer
            norm_layer1 = LayerNorm(N_head)
            norm_layer2 = LayerNorm(N_head)
        elseif tf_norm_type == :batch
            norm_layer1 = x -> batchnorm(x)
            norm_layer2 = x -> batchnorm(x)
        elseif tf_norm_type == :none
            norm_layer1 = identity
            norm_layer2 = identity
        else
            error("tf_norm_type $tf_norm_type is invalid, choose {:layer,:batch,:none}")
        end

        multi_head_attention = MultiHeadAttention(N_head => N_head => N_head;
                                                  nheads       = N_tf_head,
                                                  dropout_prob = dropout_prob,
                                                  init         = init)

        # only add activation to final feedforward NN, change for stacked encoders
        if l == 1
            ffnn = Chain(Dense(N_head, N_head, activation; init=init),
                               Dropout(dropout_prob))
        elseif l == 2
            ffnn = Chain(Dense(N_head, hidden[2], activation; init=init),
                               Dropout(dropout_prob),
                               Dense(hidden[2], N_head, activation; init=init),
                               Dropout(dropout_prob))
        else
            error("$l > 2 hidden layers is invalid with transformer")
        end

        if tf_layer_type == :prelayer
            sublayer1 = SkipConnection(Chain(norm_layer1, x-> multi_head_attention(x,x,x)[1]), +)
            sublayer2 = SkipConnection(Chain(norm_layer2, ffnn), +)
            encoder_layer = Chain(sublayer1, sublayer2)
        elseif tf_layer_type == :postlayer
            sublayer1 = SkipConnection(x-> multi_head_attention(x,x,x)[1], +)
            sublayer2 = SkipConnection(ffnn, +)
            encoder_layer = Chain(sublayer1, norm_layer1, sublayer2, norm_layer2)
        else
            error("tf_layer_type $tf_layer_type is invalid, choose {:prelayer,:postlayer}")
        end

        # initial_layer = Dense(Nf, N_head, init=init)
        final_layer   = Chain(flatten, Dense(N_head*l_window, Ny; bias=final_bias, init=init))

        # encoder layer
        m = Chain(Dense(Nf, N_head, init=init),
                  x -> x .* sqrt(N_head),
                  LPE(N_head, l_window, dropout_prob),
                  encoder_layer,
                  final_layer)

    elseif model_type == :m3w

        if l == 0
            error("hidden must have at least 1 element")
        elseif l == 1
            if 0 < dropout_prob < 1
                m = Chain(flatten,
                          Dense(Nf*l_window,hidden[1],activation),
                          Dropout(dropout_prob),
                          Dense(hidden[1],Ny;bias=final_bias))
            else
                m = Chain(flatten,
                          Dense(Nf*l_window,hidden[1],activation),
                          Dense(hidden[1],Ny;bias=final_bias))
            end
        elseif l == 2
            if 0 < dropout_prob < 1
                m = Chain(flatten,
                          Dense(Nf*l_window,hidden[1],activation),
                          Dropout(dropout_prob),
                          Dense(hidden[1],hidden[2],activation),
                          Dense(hidden[2],Ny;bias=final_bias))
            else
                m = Chain(flatten,
                          Dense(Nf*l_window,hidden[1],activation),
                          Dense(hidden[1],hidden[2],activation),
                          Dense(hidden[2],Ny;bias=final_bias))
            end
        else
            error("$l > 2 hidden layers is invalid with windowing")
        end

    else

        if l == 0
            m = Chain(Dense(Nf,Ny;bias=final_bias))
        elseif skip_con
            if l == 1
                m = Chain(SkipConnection(Dense(Nf,hidden[1],activation),
                                         (mx, x) -> cat(mx, x, dims=1)),
                          Dense(Nf+hidden[1],Ny;bias=final_bias))
            else
                error("$l > 1 hidden layers is invalid with skip connections")
            end
        elseif l == 1
            m = Chain(Dense(Nf,hidden[1],activation),
                      Dense(hidden[1],Ny;bias=final_bias))
        elseif l == 2
            m = Chain(Dense(Nf,hidden[1],activation),
                      Dense(hidden[1],hidden[2],activation),
                      Dense(hidden[2],Ny;bias=final_bias))
        elseif l == 3
            m = Chain(Dense(Nf,hidden[1],activation),
                      Dense(hidden[1],hidden[2],activation),
                      Dense(hidden[2],hidden[3],activation),
                      Dense(hidden[3],Ny;bias=final_bias))
        else
            error("$l > 3 hidden layers is invalid")
        end

    end

    return (m)
end # function get_nn_m

"""
    sparse_group_lasso(weights::Params, α=1)

Internal helper function to get the sparse group Lasso term for sparse-input
regularization, which is the combined L1 & L2 norm of the first-layer neural
network weights corresponding to each input feature.

Reference: Feng & Simon, Sparse-Input Neural Networks for High-dimensional
Nonparametric Regression and Classification, 2017. (pg. 4)

**Arguments:**
- `weights`: neural network model weights
- `α`:       (optional) Lasso (`α=0`) vs group Lasso (`α=1`) balancing parameter {0:1}

**Returns:**
- `w_norm`: sparse group Lasso term
"""
function sparse_group_lasso(weights::Params, α=1)
    return ([(1-α)*norm(weights[1][:,i],1) +
                α *norm(weights[1][:,i],2) for i in axes(weights[1],2)])
end # function sparse_group_lasso

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
    err_segs(y_hat, y, l_segs; silent::Bool=true)

Remove mean error from multiple individual flight lines within larger dataset.

**Arguments:**
- `y_hat`:  length-`N` prediction vector
- `y`:      length-`N` target vector
- `l_segs`: length-`N_lines` vector of lengths of `lines`, sum(l_segs) = `N`
- `silent`: (optional) if true, no print outs

**Returns:**
- `err`: length-`N` mean-corrected (per line) error
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
- `train`:     `N_train` x `Nf` training data
- `norm_type`: (optional) normalization type:
    - `:standardize` = Z-score normalization
    - `:normalize`   = min-max normalization
    - `:scale`       = scale by maximum absolute value, bias = 0
    - `:none`        = scale by 1, bias = 0
- `no_norm`: (optional) length-`Nf` Boolean indices of features to not be normalized

**Returns:**
- `train_bias`:  `1` x `Nf` training data biases (means, mins, or zeros)
- `train_scale`: `1` x `Nf` training data scaling factors (std devs, maxs-mins, or ones)
- `train`:       `N_train` x `Nf` training data, normalized
"""
function norm_sets(train;
                   norm_type::Symbol = :standardize,
                   no_norm           = falses(size(train,2)))

    if !(no_norm isa AbstractVector{Bool})
        no_norm = axes(train,2) .∈ (no_norm,)
    end

    if norm_type == :standardize
        train_bias  = mean(train,dims=1)
        train_scale = std( train,dims=1)
    elseif norm_type == :normalize
        train_bias  = minimum(train,dims=1)
        train_scale = maximum(train,dims=1) - train_bias
    elseif norm_type == :scale
        train_bias  = size(train,2) > 1 ? zero(train[1:1,:]) : zero(train[1:1])
        train_scale = maximum(abs.(train),dims=1)
    elseif norm_type == :none
        train_bias  = size(train,2) > 1 ? zero(train[1:1,:]) : zero(train[1:1])
        train_scale = size(train,2) > 1 ? one.(train[1:1,:]) : one.(train[1:1])
    else
        error("$norm_type normalization type not defined")
    end

    for i in axes(train,2)
        no_norm[i] && (train_bias[i]  = zero(eltype(train)))
        no_norm[i] && (train_scale[i] = one( eltype(train)))
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
- `train`:     `N_train` x `Nf` training data
- `test`:      `N_test`  x `Nf` testing data
- `norm_type`: (optional) normalization type:
    - `:standardize` = Z-score normalization
    - `:normalize`   = min-max normalization
    - `:scale`       = scale by maximum absolute value, bias = 0
    - `:none`        = scale by 1, bias = 0
- `no_norm`: (optional) length-`Nf` Boolean indices of features to not be normalized

**Returns:**
- `train_bias`:  `1` x `Nf` training data biases (means, mins, or zeros)
- `train_scale`: `1` x `Nf` training data scaling factors (std devs, maxs-mins, or ones)
- `train`:       `N_train` x `Nf` training data, normalized
- `test`:        `N_test`  x `Nf` testing  data, normalized
"""
function norm_sets(train, test;
                   norm_type::Symbol = :standardize,
                   no_norm           = falses(size(train,2)))

    if !(no_norm isa AbstractVector{Bool})
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
- `train`:     `N_train` x `Nf` training data
- `val`:       `N_val`   x `Nf` validation data
- `test`:      `N_test`  x `Nf` testing data
- `norm_type`: (optional) normalization type:
    - `:standardize` = Z-score normalization
    - `:normalize`   = min-max normalization
    - `:scale`       = scale by maximum absolute value, bias = 0
    - `:none`        = scale by 1, bias = 0
- `no_norm`: (optional) length-`Nf` Boolean indices of features to not be normalized

**Returns:**
- `train_bias`:  `1` x `Nf` training data biases (means, mins, or zeros)
- `train_scale`: `1` x `Nf` training data scaling factors (std devs, maxs-mins, or ones)
- `train`:       `N_train` x `Nf` training   data, normalized
- `val`:         `N_val`   x `Nf` validation data, normalized
- `test`:        `N_test`  x `Nf` testing    data, normalized
"""
function norm_sets(train, val, test;
                   norm_type::Symbol = :standardize,
                   no_norm           = falses(size(train,2)))

    if !(no_norm isa AbstractVector{Bool})
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
- `train_bias`:  `1` x `Nf` training data biases (means, mins, or zeros)
- `train_scale`: `1` x `Nf` training data scaling factors (std devs, maxs-mins, or ones)
- `train`:       `N_train` x `Nf` training data, normalized

**Returns:**
- `train`: `N_train` x `Nf` training data, denormalized
"""
function denorm_sets(train_bias, train_scale, train)
    train = train .* train_scale .+ train_bias
    return (train)
end # function denorm_sets

"""
    denorm_sets(train_bias, train_scale, train, test)

Denormalize (or destandardize) features (columns) of training and testing data.

**Arguments:**
- `train_bias`:  `1` x `Nf` training data biases (means, mins, or zeros)
- `train_scale`: `1` x `Nf` training data scaling factors (std devs, maxs-mins, or ones)
- `train`:       `N_train` x `Nf` training data, normalized
- `test`:        `N_test`  x `Nf` testing  data, normalized

**Returns:**
- `train`: `N_train` x `Nf` training data, denormalized
- `test`:  `N_test`  x `Nf` testing  data, denormalized
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
- `train_bias`:  `1` x `Nf` training data biases (means, mins, or zeros)
- `train_scale`: `1` x `Nf` training data scaling factors (std devs, maxs-mins, or ones)
- `train`:       `N_train` x `Nf` training   data, normalized
- `val`:         `N_val`   x `Nf` validation data, normalized
- `test`:        `N_test`  x `Nf` testing    data, normalized

**Returns:**
- `train`: `N_train` x `Nf` training   data, denormalized
- `val`:   `N_val`   x `Nf` validation data, denormalized
- `test`:  `N_test`  x `Nf` testing    data, denormalized
"""
function denorm_sets(train_bias, train_scale, train, val, test)
    train = train .* train_scale .+ train_bias
    val   = val   .* train_scale .+ train_bias
    test  = test  .* train_scale .+ train_bias
    return (train, val, test)
end # function denorm_sets

"""
    unpack_data_norms(data_norms)

Internal helper function to unpack data normalizations, some of which may
not be present due to earlier package versions being used.

**Arguments:**
- `data_norms`: length-`4` to `7` tuple of data normalizations,
    - `4`: `(`_______________________`x_bias,x_scale,y_bias,y_scale)`
    - `5`: `(`_______________`v_scale,x_bias,x_scale,y_bias,y_scale)`
    - `6`: `(A_bias,A_scale,`________`x_bias,x_scale,y_bias,y_scale)`
    - `7`: `(A_bias,A_scale,``v_scale,x_bias,x_scale,y_bias,y_scale)`

**Returns:**
- `data_norms`: length-`7` tuple of data normalizations, `(A_bias,A_scale,v_scale,x_bias,x_scale,y_bias,y_scale)`
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
    return (A_bias, A_scale, v_scale, x_bias, x_scale, y_bias, y_scale)
end # function unpack_data_norms

"""
    get_ind(tt::Vector, line::Vector;
            ind    = trues(length(tt)),
            lines  = (),
            tt_lim = (),
            splits = (1))

Get BitVector of indices for further analysis from specified indices (subset),
line number(s), and/or time range. Any or all of these may be used.

**Arguments:**
- `tt`:     time [s]
- `line`:   line number(s)
- `ind`:    (optional) selected data indices. Defaults to use all data indices
- `lines`:  (optional) selected line number(s). Defaults to use all lines
- `tt_lim`: (optional) 2-element (inclusive) start & end time limits. Defaults to use full time range [s]
- `splits`: (optional) data splits, must sum to 1

**Returns:**
- `ind`: BitVector (or tuple of BitVector) of selected data indices
"""
function get_ind(tt::Vector, line::Vector;
                 ind    = trues(length(tt)),
                 lines  = (),
                 tt_lim = (),
                 splits = (1))

    @assert sum(splits) ≈ 1 "sum of splits = $(sum(splits)) ≠ 1"
    @assert length(tt_lim) <= 2 "length of tt_lim = $(length(tt_lim)) > 2"
    @assert length(splits) <= 3 "number of splits = $(length(splits)) > 3"

    if ind isa AbstractVector{Bool}
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
    end

end # function get_ind

"""
    get_ind(xyz::XYZ;
            ind    = trues(xyz.traj.N),
            lines  = (),
            tt_lim = extrema(xyz.traj.tt),
            splits = (1))

Get BitVector of indices for further analysis from specified indices (subset),
line number(s), and/or time range. Any or all of these may be used.

**Arguments:**
- `xyz`:    `XYZ` flight data struct
- `ind`:    (optional) selected data indices. Defaults to use all data indices
- `lines`:  (optional) selected line number(s). Defaults to use all lines
- `tt_lim`: (optional) 2-element (inclusive) start & end time limits. Defaults to use full time range [s]
- `splits`: (optional) data splits, must sum to 1

**Returns:**
- `ind`: BitVector (or tuple of BitVector) of selected data indices
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
            splits        = (1),
            l_window::Int = -1)

Get BitVector of indices for further analysis via DataFrame lookup.

**Arguments:**
- `xyz`:      `XYZ` flight data struct
- `line`:     line number
- `df_line`:  lookup table (DataFrame) of `lines`
|**Field**|**Type**|**Description**
|:--|:--|:--
`flight`   |`Symbol`| flight name (e.g., `:Flt1001`)
`line`     |`Real`  | line number, i.e., segments within `flight`
`t_start`  |`Real`  | start time of `line` to use [s]
`t_end`    |`Real`  | end   time of `line` to use [s]
`full_line`|`Bool`  | (optional) if true, `t_start` to `t_end` is full line
`map_name` |`Symbol`| (optional) name of magnetic anomaly map relevant to `line`
`map_type` |`Symbol`| (optional) type of magnetic anomaly map for `map_name` {`drape`,`HAE`}
`line_alt` |`Real`  | (optional) nominal altitude of `line` [m]
- `splits`:   (optional) data splits, must sum to 1
- `l_window`: (optional) trim data by `N % l_window`, `-1` to ignore

**Returns:**
- `ind`: BitVector (or tuple of BitVector) of selected data indices
"""
function get_ind(xyz::XYZ, line::Real, df_line::DataFrame;
                 splits        = (1),
                 l_window::Int = -1)

    tt_lim = [df_line.t_start[df_line.line .== line][1],
              df_line.t_end[  df_line.line .== line][end]]
    fields = fieldnames(typeof(xyz))
    line_  = :line in fields ? xyz.line : one.(xyz.traj.tt[ind])
    inds   = get_ind(xyz.traj.tt,line_;
                     lines  = [line],
                     tt_lim = tt_lim,
                     splits = splits)

    if l_window > 0
        if (inds) isa Tuple
            for ind in inds
                N_trim = length(xyz.traj.lat[ind]) % l_window
                for _ = 1:N_trim
                    ind[findlast(ind.==1)] = 0
                end
            end
        else
            N_trim = sum(inds) % l_window
            for _ = 1:N_trim
                inds[findlast(inds.==1)] = 0
            end
        end
    end

    return (inds)
end # function get_ind

"""
    get_ind(xyz::XYZ, lines, df_line::DataFrame;
            splits        = (1),
            l_window::Int = -1)

Get BitVector of selected data indices for further analysis via DataFrame lookup.

**Arguments:**
- `xyz`:        `XYZ` flight data struct
- `lines`:      selected line number(s)
- `df_line`:    lookup table (DataFrame) of `lines`
|**Field**|**Type**|**Description**
|:--|:--|:--
`flight`   |`Symbol`| flight name (e.g., `:Flt1001`)
`line`     |`Real`  | line number, i.e., segments within `flight`
`t_start`  |`Real`  | start time of `line` to use [s]
`t_end`    |`Real`  | end   time of `line` to use [s]
`full_line`|`Bool`  | (optional) if true, `t_start` to `t_end` is full line
`map_name` |`Symbol`| (optional) name of magnetic anomaly map relevant to `line`
`map_type` |`Symbol`| (optional) type of magnetic anomaly map for `map_name` {`drape`,`HAE`}
`line_alt` |`Real`  | (optional) nominal altitude of `line` [m]
- `splits`:     (optional) data splits, must sum to 1
- `l_window`:   (optional) trim data by `N % l_window`, `-1` to ignore

**Returns:**
- `ind`: BitVector (or tuple of BitVector) of selected data indices
"""
function get_ind(xyz::XYZ, lines, df_line::DataFrame;
                 splits        = (1),
                 l_window::Int = -1)

    @assert sum(splits) ≈ 1 "sum of splits = $(sum(splits)) ≠ 1"
    @assert length(splits) <= 3 "number of splits = $(length(splits)) > 3"

    ind = falses(xyz.traj.N)
    for line in lines
        if line in df_line.line
            ind .= ind .| get_ind(xyz,line,df_line;l_window=l_window)
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
    end

end # function get_ind

"""
    chunk_data(x, y, l_window::Int)

Break data into non-overlapping sequences of length-`l_window`.

**Arguments:**
- `x`:        `N` x `Nf` data matrix (`Nf` is number of features)
- `y`:        length-`N` target vector
- `l_window`: temporal window length

**Returns:**
- `x_seqs`: vector of vector of vector of values, i.e., each sequence is a vector of feature vectors
- `y_seqs`: vector of vector of values, i.e., each sequence is a vector of scalar targets
"""
function chunk_data(x, y, l_window::Int)

    x = Float32.(x)
    y = Float32.(y)
    N = size(x,1)
    N_window = floor(Int,N/l_window)

    N % l_window == 0 || @info("data was not trimmed for l_window = $l_window, may result in worse performance")

    x_seqs = [[x[(j-1)*l_window+i,:] for i = 1:l_window]  for j = 1:N_window]
    y_seqs = [ y[(j-1)*l_window       .+    (1:l_window)] for j = 1:N_window]

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
    x     = Float32.(x)
    y_hat = [yi[1] for yi in m.([x[i,:] for i in axes(x,1)])]

    return (y_hat)
end # function predict_rnn_full

"""
    predict_rnn_windowed(m, x, l_window::Int)

Apply model `m` to inputs by sliding a window of length-`l_window` along `x`.

**Arguments:**
- `m`:        recurrent neural network model
- `x`:        input sequence (of features)
- `l_window`: temporal window length

**Returns:**
- `y_hat`: vector of scalar targets for each input in `x`
"""
function predict_rnn_windowed(m, x, l_window::Int)

    x     = Float32.(x)
    N     = size(x,1)
    y_hat = zeros(eltype(x),N)

    # assume l_window = 4
    # 1 2 3 4 5 6 7 8
    #     i     j

    for j = 1:N

        # create window
        i = j < l_window ? 1 : j - l_window + 1

        x_seq = [x[k,:] for k = i:j]

        # apply model
        Flux.reset!(m)
        y_seq = m.(x_seq)

        # store last output value in window
        y_hat[j] = y_seq[end][1]

    end

    return (y_hat)
end # function predict_rnn_windowed

"""
    krr(x_train, y_train, x_test;
        k=PolynomialKernel(;degree=1), λ=0.5)

Kernel ridge regression (KRR).

**Arguments:**
- `x_train`: `N_train` x `Nf` training data matrix (`Nf` is number of features)
- `y_train`: length-`N_train` training target vector
- `x_test`:  `N_test` x `Nf`  testing  data matrix (`Nf` is number of features)
- `k`:       (optional) kernel
- `λ`:       (optional) ridge parameter

**Returns:**
- `y_train_hat`: length-`N_train` training prediction vector
- `y_test_hat`:  length-`N_train` testing  prediction vector
"""
function krr(x_train, y_train, x_test;
             k=PolynomialKernel(;degree=1), λ=0.5)

    K  = kernelmatrix(k,x_train;obsdim=1)
    K★ = kernelmatrix(k,x_test,x_train;obsdim=1)
    kt = (K + λ*I) \ y_train

    y_train_hat = K  * kt
    y_test_hat  = K★ * kt

    # w = x_train' * ((K + λ*I) \ y_train)
    # return (w)
    return (y_train_hat, y_test_hat)
end # function krr

"""
    predict_shapley(m, df::DataFrame)

Internal helper function to wrap a neural network model to create a DataFrame
of predictions.

**Arguments:**
- `m`:  neural network model
- `df`: `N` x `Nf` DataFrame (`Nf` is number of features)

**Returns:**
- `y_hat`: 1-column DataFrame of predictions from `m` using `df`
"""
function predict_shapley(m, df::DataFrame)
    DataFrame(y_hat=vec(m(collect(Matrix(df)'))))
end # function predict_shapley

"""
    eval_shapley(m, x, features::Vector{Symbol},
                 N::Int      = min(10000,size(x,1)),
                 num_mc::Int = 10)

Compute stochastic Shapley effects for global feature importance.

Reference: https://nredell.github.io/ShapML.jl/dev/#Examples-1

**Arguments:**
- `m`:        neural network model
- `x`:        `N` x `Nf` data matrix (`Nf` is number of features)
- `features`: length-`Nf` feature vector (including components of TL `A`, etc.)
- `N`:        (optional) number of samples (instances) to use for explanation
- `num_mc`:   (optional) number of Monte Carlo simulations

**Returns:**
- `df_shap`:       DataFrame of Shapley effects
- `baseline_shap`: intercept of Shapley effects
"""
function eval_shapley(m, x, features::Vector{Symbol},
                      N::Int      = min(10000,size(x,1)),
                      num_mc::Int = 10)

    seed!(2) # for reproducibility

    # put data matrix into DataFrame
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
|**Field**|**Type**|**Description**
|:--|:--|:--
`feature_name`|`Symbol`| feature name
`mean_effect` |`Real`  | mean Shapley effect
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
    eval_gsa(m, x, n::Int=min(10000,size(x,1)))

Global sensitivity analysis (GSA) with the Morris Method.

Reference: https://book.sciml.ai/notes/17-Global_Sensitivity_Analysis/

Reference: https://docs.sciml.ai/GlobalSensitivity/stable/methods/morris/

**Arguments:**
- `m`: neural network model
- `x`: `N` x `Nf` data matrix (`Nf` is number of features)
- `n`: (optional) number of samples (instances) to use for explanation

**Returns:**
- `means`: means of elementary effects
"""
function eval_gsa(m, x, n::Int=min(10000,size(x,1)))

    seed!(2) # for reproducibility

    method      = Morris(relative_scale=true,num_trajectory=n)
    param_range = vec(extrema(x,dims=1))
    means       = vec(gsa(m,method,param_range;samples=n).means)
    #* note: produces Float32 warning even if param_range is Float32

    return (means)
end # function eval_gsa

"""
    get_igrf(xyz::XYZ, ind=trues(xyz.traj.N);
             frame::Symbol   = :body,
             norm_igrf::Bool = false,
             check_xyz::Bool = true)

Returns the IGRF Earth vector in the body or navigation frame given an `XYZ`
flight data struct containing trajectory information, valid indices, a start
date in IGRF time (years since 0 CE), and reference frame.

**Arguments:**
- `xyz`:       `XYZ` flight data struct
- `ind`:       (optional) selected data indices
- `frame`:     (optional) desired reference frame {`:body`,`:nav`}
- `norm_igrf`: (optional) if true, normalize `igrf_vec`
- `check_xyz`: (optional) if true, cross-check with `igrf` field in `xyz`

**Returns:**
- `igrf_vec`: length-`N` stacked vector of `3` IGRF coordinates in `frame`
"""
function get_igrf(xyz::XYZ, ind=trues(xyz.traj.N);
                  frame::Symbol   = :body,
                  norm_igrf::Bool = false,
                  check_xyz::Bool = true)

    date_start = get_years.(xyz.year[ind],xyz.doy[ind].-1)
    seconds_in_year = 60 * 60 * 24 * get_days_in_year.(date_start)

    @assert frame in [:body,:nav] "must choose `:body` or `:nav` reference frame"
    @assert all(1900 .<= date_start .<= 2030)  "start date must be in valid IGRF date range"

    N   = length(xyz.traj.lat[ind])
    tt  = date_start + xyz.traj.tt[ind] ./ seconds_in_year # [yr]
    alt = xyz.traj.alt[ind]
    lat = xyz.traj.lat[ind]
    lon = xyz.traj.lon[ind]

    # get IGRF, igrf() outputs length N vector of [Bx, By, Bz] with
    # Bx: north component [nT]
    # By: east  component [nT]
    # Bz: down  component [nT]
    igrf_vec = [igrf(tt[i], alt[i], lat[i], lon[i], Val(:geodetic)) for i = 1:N]
    if check_xyz
        lim_igrf = 1
        err_igrf = round(rms(norm.(igrf_vec) - xyz.igrf[ind]),digits=2)
        @assert err_igrf < lim_igrf "IGRF discrepancy = $err_igrf > $lim_igrf nT, check `igrf`, `year`, & `doy` fields in `xyz`"
    end

    # normalize (or not) and rotate (or not)
    norm_igrf && (igrf_vec = normalize.(igrf_vec))
    if (frame == :nav)
        return (igrf_vec)
    else # convert to body frame
        Cnb = iszero(xyz.traj.Cnb[:,:,ind]) ? xyz.ins.Cnb[:,:,ind] : xyz.traj.Cnb[:,:,ind] # body to navigation
        # transpose is navigation to body
        igrf_vec_body = [Cnb[:,:,i]'*igrf_vec[i] for i=1:N]
        return (igrf_vec_body)
    end
end # function get_igrf

get_IGRF = get_igrf

"""
    project_vec_to_2d(vec_in, uvec_x, uvec_y)

Internal helper function to project a 3D vector into 2D given two orthogonal
3D unit vectors.

**Arguments:**
- `vec_in`: 3D vector desired to be projected onto a 2D plane
- `uvec_x`: 3D unit vector
- `uvec_y`: 3D unit vector orthogonal to `uvec_x`

**Returns:**
- `v_out`: 2D vector representing the 3D projection onto the plane
"""
function project_vec_to_2d(vec_in, uvec_x, uvec_y)
    @assert abs(dot(uvec_x, uvec_y)) <= 1e-7 "projected vectors must be orthogonal: ", dot(uvec_x, uvec_y)
    @assert norm(uvec_x) ≈ 1 "unit vector norm = $(norm(uvec_x)) ≠ 1"
    @assert norm(uvec_y) ≈ 1 "unit vector norm = $(norm(uvec_y)) ≠ 1"
    [dot(vec_in,uvec_x), dot(vec_in,uvec_y)]
end # function project_vec_to_2d

"""
    project_body_field_to_2d_igrf(vec_body, igrf_nav, Cnb)

Projects a body frame vector onto a 2D plane defined by the direction of the
IGRF and a tangent vector to the Earth ellipsoid, which is computed by taking
the cross product of the IGRF with the upward direction. Returns a 2D vector
whose components describe the amount of the body field that is in alignment
with the Earth field and an orthogonal direction to the Earth field (roughly
to the east).

**Arguments:**
- `vec_body`: vector in body frame (e.g., aircraft induced field)
- `igrf_nav`: IGRF unit vector in navigation frame
- `Cnb`:      `3` x `3` x `N` direction cosine matrix (body to navigation) [-]

**Returns:**
- `v_out`: 2D vector whose components illustrate projection onto the Earth field and an orthogonal component
"""
function project_body_field_to_2d_igrf(vec_body, igrf_nav, Cnb)
    # assume igrf_nav is "north" in navigation frame and rotate about Z axis to get "east"
    igrf_north     = igrf_nav  # components are [north, east, down]
    igrf_tan_earth = cross(igrf_north, [0.0, 0.0, -1.0]) # cross product with "up" direction
    igrf_east      = normalize(igrf_tan_earth)

    # transform aircraft vector from body to navigation frame
    vec_nav = Cnb*vec_body

    v_out = project_vec_to_2d(vec_nav,igrf_north,igrf_east)
    return (v_out)
end # function project_body_field_to_2d_igrf

"""
    get_optimal_rotation_matrix(v1s, v2s)

Returns the `3` x `3` rotation matrix rotating the directions of v1s into v2s.
Uses the Kabsch algorithm.

Reference: https://en.wikipedia.org/wiki/Kabsch_algorithm

**Arguments:**
- `v1s`: `N` x `3` matrix for first  set of 3D points
- `v2s`: `N` x `3` matrix for second set of 3D points

**Returns:**
- `R`: 3D matrix rotatating v1s into v2s' directions
"""
function get_optimal_rotation_matrix(v1s, v2s)
    @assert size(v1s,1) == size(v2s,1) "size(`v1s`,1) ≂̸ size(`v2s`,1)"
    @assert size(v1s,2) == 3 "size(`v1s`,2) ≂̸ 3"
    @assert size(v2s,2) == 3 "size(`v2s`,2) ≂̸ 3"

    # compute centroids and recenter point clouds
    v1_centroid   = mean(v1s,dims=1)
    v2_centroid   = mean(v2s,dims=1)
    v1_recentered = v1s .- v1_centroid
    v2_recentered = v2s .- v2_centroid

    # calculate cross-covariance matrix
    cov_matrix = v1_recentered'*v2_recentered

    # compute rotation matrix using least squares
    # R = sqrt(cov_matrix'*cov_matrix)*inv(cov_matrix)

    # compute rotation matrix using the Kabsch algorithm
    (U,_,V) = svd(cov_matrix)
    d = sign(det(V*transpose(U)))
    R = V*diagm([1.0,1.0,d])*transpose(U)

    @assert det(R) ≈ 1 "rotation matrix should not scale"
    @assert R*R' ≈ diagm([1.0,1.0,1.0]) "rotation matrix transpose should be its inverse"
    return (R)
end # function get_optimal_rotation_matrix

"""
    get_days_in_year(year)

Get days in year based on (rounded down) year.

**Arguments:**
- `year`: year (rounded down)

**Returns:**
- `days_in_year`: days in `year`
"""
function get_days_in_year(year)
    floor(Int,year) % 4 == 0 ? 366 : 365
end # function get_days_in_year

"""
    get_years(year, doy=0)

Get years from year and day of year.

**Arguments:**
- `year`: year
- `doy`:  day of year

**Returns:**
- `years`: `year` with fractional `doy`
"""
function get_years(year, doy)
    round(Int,year) + doy/get_days_in_year(year)
end # function get_years

"""
    get_lim(x, frac=0)

Internal helper function to get expanded limits (extrema) of data.

**Arguments:**
- `x`:    data
- `frac`: fraction of `x` limits (extrema) to expand

**Returns:**
- `lim`: limits (extrema) of `x` expanded by `frac` on each end
"""
function get_lim(x, frac=0)
    extrema(x) .+ (-frac,frac) .* (extrema(x)[2] - extrema(x)[1])
end # function get_lim

"""
    expand_range(x::Vector, xlim::Tuple = get_lim(x),
                 extra_step::Bool = false)

Internal helper function to expand range of data that has constant step size.

**Arguments:**
- `x`:          data
- `xlim`:       (optional) limits `(xmin,xmax)` to which `x` is expanded
- `extra_step`: (optional) if true, expand range by extra step on each end

**Returns:**
- `x`:     data, expanded
- `x_ind`: data indices within `x` that contain original data
"""
function expand_range(x::Vector, xlim::Tuple = get_lim(x),
                      extra_step::Bool = false)
    dx = get_step(x)
    imin = 0
    (xmin,xmax) = extrema(x)
    while minimum(xlim) < xmin
        imin += 1
        xmin -= dx
    end
    while xmax < maximum(xlim)
        xmax += dx
    end
    if extra_step
        imin += 1
        xmin -= dx
        xmax += dx
    end
    return ([xmin:dx:xmax;], imin.+(1:length(x)))
end # function expand_range

"""
    filter_events!(df_event::DataFrame, flight::Symbol, keyword::String = "";
                   tt_lim::Tuple = extrema(df_event.tt[df_event.flight .== flight]))

Filter a DataFrame of in-flight events to only contain relevant events.

**Arguments:**
- `df_event`: lookup table (DataFrame) of in-flight events
|**Field**|**Type**|**Description**
|:--|:--|:--
`flight`|`Symbol`| flight name (e.g., `:Flt1001`)
`tt`    |`Real`  | time of `event` [s]
`event` |`String`| event description
- `flight`:   flight name (e.g., `:Flt1001`)
- `keyword`:  (optional) keyword to search within events, case insensitive
- `tt_lim`:   (optional) 2-element (inclusive) start & end time limits. Defaults to use full time range [s]

**Returns:**
- `nothing`: `df_event` is filtered
"""
function filter_events!(df_event::DataFrame, flight::Symbol, keyword::String = "";
                        tt_lim::Tuple = extrema(df_event.tt[df_event.flight .== flight]))
    (t_start,t_end) = tt_lim
    filter!(:flight => f -> (f == flight), df_event)
    filter!(:tt     => t -> (t_start <= t <= t_end), df_event)
    filter!(:event  => e -> occursin(keyword,lowercase(e)), df_event)
end # function filter_events!

"""
    filter_events(df_event::DataFrame, flight::Symbol, keyword::String = "";
                  tt_lim::Tuple = extrema(df_event.tt[df_event.flight .== flight]))

Filter a DataFrame of in-flight events to only contain relevant events.

**Arguments:**
- `df_event`: lookup table (DataFrame) of in-flight events
|**Field**|**Type**|**Description**
|:--|:--|:--
`flight`|`Symbol`| flight name (e.g., `:Flt1001`)
`tt`    |`Real`  | time of `event` [s]
`event` |`String`| event description
- `flight`:   flight name (e.g., `:Flt1001`)
- `keyword`:  (optional) keyword to search within events, case insensitive
- `tt_lim`:   (optional) 2-element (inclusive) start & end time limits. Defaults to use full time range [s]

**Returns:**
- `df_event`: lookup table (DataFrame) of in-flight events, filtered
"""
function filter_events(df_event::DataFrame, flight::Symbol, keyword::String = "";
                       tt_lim::Tuple = extrema(df_event.tt[df_event.flight .== flight]))
    df_event = deepcopy(df_event)
    filter_events!(df_event,flight,keyword;tt_lim=tt_lim)
    return (df_event)
end # function filter_events

"""
    gif_animation_m3(TL_perm::AbstractMatrix, TL_induced::AbstractMatrix, TL_eddy::AbstractMatrix,
                     TL_aircraft::AbstractMatrix, B_unit::AbstractMatrix, y_nn::AbstractMatrix,
                     y::Vector, y_hat::Vector, xyz::XYZ,
                     filt_lat::Vector = [],
                     filt_lon::Vector = [];
                     ind              = trues(xyz.traj.N),
                     tt_lim::Tuple    = (0, (xyz.traj(ind).N-1)*xyz.traj.dt/60),
                     skip_every::Int  = 5,
                     save_plot::Bool  = false,
                     mag_gif::String  = "comp_xai.gif")

After calling `comp_m3_test` to generate the individual model components,
this function makes a GIF animation of the model components and the true and
predicted scalar magnetic field.

**Arguments**
- `TL_perm`:     `3` x `N` matrix of TL permanent vector field
- `TL_induced`:  `3` x `N` matrix of TL induced vector field
- `TL_eddy`:     `3` x `N` matrix of TL eddy current vector field
- `TL_aircraft`: `3` x `N` matrix of TL aircraft vector field
- `B_unit`:      `3` x `N` matrix of normalized vector magnetometer measurements
- `y_nn`:        `3` x `N` matrix of vector neural network correction (for scalar models, in direction of `Bt`)
- `y`:           length-`N` target vector
- `y_hat`:       length-`N` prediction vector
- `xyz`:         `XYZ` flight data struct
- `filt_lat`:    (optional) length-`N` filter output latitude  [rad]
- `filt_lon`:    (optional) length-`N` filter output longitude [rad]
- `ind`:         (optional) selected data indices
- `tt_lim`:      (optional) 2-element (inclusive) start & end time limits. Defaults to use full time range [min]
- `skip_every`:  (optional) number of time steps to skip between frames
- `save_plot`:   (optional) if true, `g1` will be saved as `mag_gif`
- `mag_gif`:     (optional) path/name of magnetic field GIF file to save (`.gif` extension optional)

**Returns**
- `g1`: magnetic field GIF animation

**Example**
```julia
gif_animation_m3(TL_perm, TL_induced, TL_eddy, TL_aircraft, B_unit,
                 y_nn, y, y_hat, xyz, filt_lat, filt_lon; ind=ind, tt_lim=(0.0,10.0),
                 skip_every=5, save_plot=false, mag_gif="comp_xai.gif")
```
"""
function gif_animation_m3(TL_perm::AbstractMatrix, TL_induced::AbstractMatrix, TL_eddy::AbstractMatrix,
                          TL_aircraft::AbstractMatrix, B_unit::AbstractMatrix, y_nn::AbstractMatrix,
                          y::Vector, y_hat::Vector, xyz::XYZ,
                          filt_lat::Vector = [],
                          filt_lon::Vector = [];
                          ind              = trues(xyz.traj.N),
                          tt_lim::Tuple    = (0, (xyz.traj(ind).N-1)*xyz.traj.dt/60),
                          skip_every::Int  = 5,
                          save_plot::Bool  = false,
                          mag_gif::String  = "comp_xai.gif")

    @assert size(TL_perm,2) == size(TL_induced,2) == size(TL_eddy,2) == size(TL_aircraft,2) == size(B_unit,2)
    @assert size(y_nn,2) == length(y) == length(y_hat) == xyz.traj(ind).N
    @assert 0 < skip_every < xyz.traj(ind).N

    show_ins  = false
    show_filt = length(filt_lat) == length(filt_lon) == xyz.traj(ind).N

    # get lat & lon for plotting
    traj     = xyz.traj(ind)
    ins      = xyz.ins(ind)
    filt_lat = rad2deg.(filt_lat)
    filt_lon = rad2deg.(filt_lon)
    gps_lat  = rad2deg.(traj.lat)
    gps_lon  = rad2deg.(traj.lon)

    xlim = get_lim(gps_lon,0.2)
    ylim = get_lim(gps_lat,0.2)
    ves  = traj.ve
    vns  = traj.vn
    dir  = atand.(vns,ves)

    # convert fields to 2D "compass" projections
    igrf_nav = get_igrf(xyz,ind;
                        frame     = :nav,
                        norm_igrf = true,
                        check_xyz = true)

    # compute dot product component of each field
    NN_component = vec(sum(y_nn        .* B_unit, dims=1))
    TL_component = vec(sum(TL_aircraft .* B_unit, dims=1))

    # note that we can project onto the 3D IGRF vector to get the full picture,
    # although most of the 3D vector may be in the "down" direction, which makes
    # the 2D plane tilt down towards the north pole. A remedy is to drop the
    # z-dimension and treat it like a compass, which is shown here
    igrf_nav_2D = reduce(hcat, igrf_nav)
    igrf_nav_2D[3,:] .= 0.0
    normalize!.(eachcol(igrf_nav_2D))

    Cnb = iszero(traj.Cnb) ? ins.Cnb : traj.Cnb # body to navigation
    aircraft_2D_NN = project_body_field_to_2d_igrf.(eachcol(y_nn       ),eachcol(igrf_nav_2D),eachslice(Cnb,dims=3))
    aircraft_2D_TL = project_body_field_to_2d_igrf.(eachcol(TL_aircraft),eachcol(igrf_nav_2D),eachslice(Cnb,dims=3))
    perm_field_2D  = project_body_field_to_2d_igrf.(eachcol(TL_perm    ),eachcol(igrf_nav_2D),eachslice(Cnb,dims=3))
    ind_field_2D   = project_body_field_to_2d_igrf.(eachcol(TL_induced ),eachcol(igrf_nav_2D),eachslice(Cnb,dims=3))
    eddy_field_2D  = project_body_field_to_2d_igrf.(eachcol(TL_eddy    ),eachcol(igrf_nav_2D),eachslice(Cnb,dims=3))

    aircraft_2D_NN = reduce(hcat,aircraft_2D_NN)
    aircraft_2D_TL = reduce(hcat,aircraft_2D_TL)
    perm_field_2D  = reduce(hcat,perm_field_2D)
    ind_field_2D   = reduce(hcat,ind_field_2D)
    eddy_field_2D  = reduce(hcat,eddy_field_2D)

    tt      = (traj.tt .- traj.tt[1]) / 60
    i_start = findfirst(tt .>= tt_lim[1])
    i_end   = findlast( tt .<= tt_lim[2])

    # create gif
    l  = @layout [ a{0.6w} [b;c] ]
    p1 = plot(layout=l, size=(800,500), margin=5*mm)
    a1 = Animation()

    for i in i_start:skip_every:i_end
        p1 = plot(layout=l, size=(800,500), margin=5*mm)

        # move a vertical line across the magnetic field data
        plot!(p1[1],xlab="time [min]",ylab=" magnetic field [nT]",xlim=tt_lim, legend=:bottomleft)
        plot!(p1[1],tt,y           ,lab="true compensation"   ,color=:gray, style=:dash)
        plot!(p1[1],tt,y_hat       ,lab="model 3 compensation",color=:black)
        plot!(p1[1],tt,TL_component,lab="TL component"        ,color=:blue)
        plot!(p1[1],tt,NN_component,lab="NN component"        ,color=:red)
        plot!(p1[1], [tt[i]], linetype=:vline, lab="", color=:black)

        # draw compass plot for each field
        plot!(p1[2],xlab="east [nT]",ylab=" north [nT]",
              xlim=(-2500,2500),ylim=(-2500,2500),legend=:topright)
        plot!(p1[2],[0.0,aircraft_2D_TL[2,i]],[0.0,aircraft_2D_TL[1,i]],arrow=true,lab="TL",color=:blue)
        plot!(p1[2],[0.0,aircraft_2D_NN[2,i]],[0.0,aircraft_2D_NN[1,i]],arrow=true,lab="NN",color=:red)
        plot!(p1[2],[0.0,perm_field_2D[ 2,i]],[0.0,perm_field_2D[ 1,i]],arrow=true,lab="perm.")
        plot!(p1[2],[0.0,ind_field_2D[  2,i]],[0.0,ind_field_2D[  1,i]],arrow=true,lab="ind.")
        plot!(p1[2],[0.0,eddy_field_2D[ 2,i]],[0.0,eddy_field_2D[ 1,i]],arrow=true,lab="eddy")

        # plot airplane on map
        plot!(p1[3],xlab="longitude [deg]",ylab="latitude [deg]")
        plot!(p1[3],gps_lon[1:i] ,gps_lat[1:i] ,xlim=xlim,ylim=ylim,lab="GPS")
        show_ins  && (plot!(p1[3],ins_lon[1:i] ,ins_lat[1:i] ,xlim=xlim,ylim=ylim,lab="INS"))
        show_filt && (plot!(p1[3],filt_lon[1:i],filt_lat[1:i],xlim=xlim,ylim=ylim,lab="EKF",xrotation=18))
        annotate!(gps_lon[i], gps_lat[i], Plots.text("✈", 20, rotation=dir[i]), subplot=3)

        frame(a1,p1)
    end

    # show or save gif
    mag_gif = add_extension(mag_gif,".gif")
    g1 = save_plot ? gif(a1,mag_gif;fps=15) : gif(a1;fps=15)

    return (g1)
end # function gif_animation_m3
