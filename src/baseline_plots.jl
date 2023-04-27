"""
    plot_basic(tt::Vector, y::Vector;
               ind          = trues(length(tt)),
               xlab::String = "time [min]",
               ylab::String = "",
               lab::String  = "")

**Arguments:**
- `tt`:   time [s]
- `y`:    observed data
- `ind`:  (optional) selected data indices
- `xlab`: (optional) x-axis label
- `ylab`: (optional) y-axis label
- `lab`:  (optional) data (legend) label

**Returns:**
- `p1`: plot with `y` vs `tt`
"""
function plot_basic(tt::Vector, y::Vector;
                    ind          = trues(length(tt)),
                    xlab::String = "time [min]",
                    ylab::String = "",
                    lab::String  = "")
    plot((tt[ind].-tt[ind][1])/60, y[ind], xlab=xlab, ylab=ylab, lab=lab)
end # function plot_basic

"""
    plot_activation(activation = [:relu,:σ,:swish,:tanh];
                    plot_deriv::Bool  = false,
                    save_plot::Bool   = false,
                    file_name::String = "act_func")

Plot activation function(s) or their derivative(s).

**Arguments:**
- `activation`: activation function(s) to plot
    - `relu`  = rectified linear unit
    - `σ`     = sigmoid (logistic function)
    - `swish` = self-gated
    - `tanh`  = hyperbolic tan
- `plot_deriv`: (optional) if true, plot activation function derivatives
- `save_plot`:  (optional) if true, plot will be saved
- `file_name`:  (optional) plot file name to save

**Returns:**
- `p1`: plot of activation function(s) or their derivative(s)
"""
function plot_activation(activation = [:relu,:σ,:swish,:tanh];
                         plot_deriv::Bool  = false,
                         save_plot::Bool   = false,
                         file_name::String = "act_func")

    save_plot ? dpi = 500 : dpi = 200

    # setup
    x = LinRange(-6,4,1000)
    p_relu  = relu.(x)
    p_σ     = sigmoid.(x)
    p_swish = swish.(x)
    p_tanh  = tanh.(x)

    d_relu  = ForwardDiff.derivative.(relu,x)
    d_σ     = ForwardDiff.derivative.(sigmoid,x)
    d_swish = ForwardDiff.derivative.(swish,x)
    d_tanh  = ForwardDiff.derivative.(tanh,x)

    if !plot_deriv # plot activation functions
        p1 = plot(xlab="z",ylab="f(z)",xlim=(-6,4),ylim=(-2,4),
                  legend=:topleft,margin=2*mm,dpi=dpi)
        :relu  in activation && plot!(p1,x,p_relu ,lab="ReLU"   ,lw=2,ls=:solid)
        :σ     in activation && plot!(p1,x,p_σ    ,lab="sigmoid",lw=2,ls=:solid)
        :swish in activation && plot!(p1,x,p_swish,lab="Swish"  ,lw=2,ls=:dash)
        :tanh  in activation && plot!(p1,x,p_tanh ,lab="tanh"   ,lw=2,ls=:dot)
    else # plot derivatives of activation functions
        p1 = plot(xlab="z",ylab="f'(z)",xlim=(-6,4),ylim=(-0.5,1.5),
                  legend=:topleft,margin=2*mm,dpi=dpi)
        :relu  in activation && plot!(p1,x,d_relu ,lab="ReLU"   ,lw=2,ls=:solid)
        :σ     in activation && plot!(p1,x,d_σ    ,lab="sigmoid",lw=2,ls=:solid)
        :swish in activation && plot!(p1,x,d_swish,lab="Swish"  ,lw=2,ls=:dash)
        :tanh  in activation && plot!(p1,x,d_tanh ,lab="tanh"   ,lw=2,ls=:dot)
    end

    save_plot && png(p1,file_name)

    return (p1)
end # function plot_activation

"""
    plot_mag(xyz::XYZ;
             ind                       = trues(xyz.traj.N),
             detrend_data::Bool        = false,
             use_mags::Vector{Symbol}  = [:all_mags],
             vec_terms::Vector{Symbol} = [:all],
             ylim                      = (),
             dpi::Int                  = 200,
             show_plot::Bool           = true,
             save_plot::Bool           = false,
             file_name::String         = "scalar_mags")

Plot scalar or vector (fluxgate) magnetometer data from a given flight test.

**Arguments:**
- `xyz`:          `XYZ` flight data struct
- `ind`:          (optional) selected data indices
- `detrend_data`: (optional) if true, plot data will be detrended
- `use_mags`:     (optional) scalar or vector (fluxgate) magnetometers to plot {`:all_mags`, `:comp_mags` or `:mag_1_c`, `:mag_1_uc`, `:flux_a`, etc.}
    - `:all_mags`  = all provided scalar magnetometer fields, e.g., `:mag_1_c`, `:mag_1_uc`, etc.
    - `:comp_mags` = provided compensation(s) between `:mag_1_uc` & `:mag_1_c`, etc.
- `vec_terms`:    (optional) vector magnetometer (fluxgate) terms to plot {`:all` or `:x`,`:y`,`:z`,`:t`}
- `ylim`:         (optional) 2-element y limits for plotting
- `dpi`:          (optional) dots per inch (image resolution)
- `show_plot`:    (optional) if true, plot will be shown
- `save_plot`:    (optional) if true, plot will be saved
- `file_name`:    (optional) plot file name to save

**Returns:**
- `p1`: plot of scalar or vector (fluxgate) magnetometer data
"""
function plot_mag(xyz::XYZ;
                  ind                       = trues(xyz.traj.N),
                  detrend_data::Bool        = false,
                  use_mags::Vector{Symbol}  = [:all_mags],
                  vec_terms::Vector{Symbol} = [:all],
                  ylim                      = (),
                  dpi::Int                  = 200,
                  show_plot::Bool           = true,
                  save_plot::Bool           = false,
                  file_name::String         = "scalar_mags")

    tt = (xyz.traj.tt[ind] .- xyz.traj.tt[ind][1]) / 60
    xlab = "time [min]"

    fields   = fieldnames(typeof(xyz))
    list_c   = [Symbol("mag_",i,"_c" ) for i = 1:num_mag_max]
    list_uc  = [Symbol("mag_",i,"_uc") for i = 1:num_mag_max]
    mags_c   = list_c[  list_c  .∈ (fields,)]
    mags_uc  = list_uc[ list_uc .∈ (fields,)]
    mags_c_  = findall((list_c  .∈ (fields,)) .& (list_uc .∈ (fields,)))
    mags_uc_ = findall((list_c  .∈ (fields,)) .& (list_uc .∈ (fields,)))
    mags_all = [mags_c; mags_uc]

    if :comp_mags in use_mags

        ylab = "magnetic field error [nT]"

        if isempty(ylim)
            p1 = plot(lw=2,dpi=dpi,xlab=xlab,ylab=ylab)
        else
            p1 = plot(lw=2,dpi=dpi,xlab=xlab,ylab=ylab,ylim=ylim)
        end

        for i in eachindex(mags_c_)
            val = (getfield(xyz,list_uc[mags_uc_[i]]) - 
                   getfield(xyz,list_c[ mags_c_[ i]]))[ind]
            detrend_data && (val = detrend(val))
            plot!(p1,tt,val,lab="mag_$i comp")
            println("==== mag_$i comp ====")
            println("avg comp = $(round(mean(val),digits=3)) nT")
            println("std dev  = $(round(std( val),digits=3)) nT")
        end

    elseif any(use_mags .∈ (field_check(xyz,MagV),))

        vec_terms = :all in vec_terms ? [:x,:y,:z,:t] : vec_terms

        ylab = "magnetic field [nT]"
        ylab = detrend_data ? "detrended $ylab" : ylab

        if isempty(ylim)
            p1 = plot(lw=2,dpi=dpi,xlab=xlab,ylab=ylab)
        else
            p1 = plot(lw=2,dpi=dpi,xlab=xlab,ylab=ylab,ylim=ylim)
        end

        for use_mag in use_mags[use_mags .∈ (field_check(xyz,MagV),)]

            flux = getfield(xyz,use_mag)(ind)
    
            for vec_term in vec_terms
                val = getfield(flux,vec_term)
                detrend_data && (val = detrend(val))
                plot!(p1,tt,val,lab="$use_mag $vec_term")
            end

        end

    elseif any(use_mags .∈ ([:all_mags; mags_all],))

        :all_mags in use_mags && (use_mags = mags_all)

        ylab = "magnetic field [nT]"
        ylab = detrend_data ? "detrended $ylab" : ylab

        if isempty(ylim)
            p1 = plot(lw=2,dpi=dpi,xlab=xlab,ylab=ylab)
        else
            p1 = plot(lw=2,dpi=dpi,xlab=xlab,ylab=ylab,ylim=ylim)
        end

        for mag in use_mags
            val = getfield(xyz,mag)[ind]
            detrend_data && (val = detrend(val))
            plot!(p1,tt,val,lab="$mag")
        end

    else

        ylab = ""

        if isempty(ylim)
            p1 = plot(lw=2,dpi=dpi,xlab=xlab,ylab=ylab)
        else
            p1 = plot(lw=2,dpi=dpi,xlab=xlab,ylab=ylab,ylim=ylim)
        end

        for mag in use_mags
            val = getfield(xyz,mag)[ind]
            detrend_data && (val = detrend(val))
            plot!(p1,tt,val,lab="$mag")
        end

    end

    show_plot && display(p1)
    save_plot && png(file_name)

    return (p1)
end # function plot_mag

"""
    plot_mag_c(xyz::XYZ,xyz_comp::XYZ;
               ind                      = trues(xyz.traj.N),
               ind_comp                 = trues(xyz_comp.traj.N),
               detrend_data::Bool       = true,
               λ                        = 0.025,
               terms                    = [:permanent,:induced,:eddy],
               pass1                    = 0.1,
               pass2                    = 0.9,
               fs                       = 10.0,
               use_mags::Vector{Symbol} = [:all_mags],
               use_vec::Symbol          = :flux_a,
               plot_diff::Bool          = false,
               plot_mag_1_uc::Bool      = true,
               plot_mag_1_c::Bool       = true,
               ylim                     = (),
               dpi::Int                 = 200,
               show_plot::Bool          = true,
               save_plot::Bool          = false,
               file_name::String        = "scalar_mags_comp")

Plot compensated magnetometer(s) data from a given flight test. Assumes `mag_1` 
(i.e., `:mag_1_uc` & `:mag_1_c`) is the best magnetometer (i.e., stinger).

**Arguments:**
- `xyz`:           `XYZ` flight data struct
- `xyz_comp`:      `XYZ` flight data struct to use for compensation
- `ind`:           (optional) selected data indices
- `ind_comp`:      (optional) selected data indices to use for compensation
- `detrend_data`:  (optional) if true, plot data will be detrended
- `λ`:             (optional) ridge parameter
- `terms`:         (optional) Tolles-Lawson terms to use {`:permanent`,`:induced`,`:eddy`,`:bias`}
- `pass1`:         (optional) filter first passband frequency [Hz]
- `pass2`:         (optional) filter second passband frequency [Hz]
- `fs`:            (optional) filter sampling frequency [Hz]
- `use_mags`:      (optional) scalar or vector (fluxgate) magnetometers to plot {`:all_mags`, or `:mag_1_uc`, etc.}
    - `:all_mags`  = all uncompensated scalar magnetometer fields, e.g., `:mag_1_uc`, etc.
- `use_vec`:       (optional) vector magnetometer (fluxgate) to use for Tolles-Lawson `A` matrix {`:flux_a`, etc.}
- `plot_diff`:     (optional) if true, plot difference between `provided` compensated data & compensated mags `as done here`
- `plot_mag_1_uc`: (optional) if true, plot mag_1_uc (uncompensated mag_1)
- `plot_mag_1_c`:  (optional) if true, plot mag_1_c (compensated mag_1)
- `ylim`:          (optional) 2-element y limits for plotting
- `dpi`:           (optional) dots per inch (image resolution)
- `show_plot`:     (optional) if true, plot will be shown
- `save_plot`:     (optional) if true, plot will be saved
- `file_name`:     (optional) plot file name to save

**Returns:**
- `p1`: plot of compensated magnetometer data
"""
function plot_mag_c(xyz::XYZ,xyz_comp::XYZ;
                    ind                      = trues(xyz.traj.N),
                    ind_comp                 = trues(xyz_comp.traj.N),
                    detrend_data::Bool       = true,
                    λ                        = 0.025,
                    terms                    = [:permanent,:induced,:eddy],
                    pass1                    = 0.1,
                    pass2                    = 0.9,
                    fs                       = 10.0,
                    use_mags::Vector{Symbol} = [:all_mags],
                    use_vec::Symbol          = :flux_a,
                    plot_diff::Bool          = false,
                    plot_mag_1_uc::Bool      = true,
                    plot_mag_1_c::Bool       = true,
                    dpi::Int                 = 200,
                    ylim                     = (),
                    show_plot::Bool          = true,
                    save_plot::Bool          = false,
                    file_name::String        = "scalar_mags_comp")

    field_check(xyz,use_vec,MagV)
    A = create_TL_A(getfield(xyz,use_vec),terms=terms)[ind,:]

    tt       = (xyz.traj.tt[ind] .- xyz.traj.tt[ind][1]) / 60
    mag_1_c  = detrend_data ? detrend(xyz.mag_1_c[ind ]) : xyz.mag_1_c[ind]
    mag_1_uc = detrend_data ? detrend(xyz.mag_1_uc[ind]) : xyz.mag_1_uc[ind]

    xlab = "time [min]"
    ylab = "magnetic field [nT]"

    if isempty(ylim)
        p1 = plot(lw=2,dpi=dpi,xlab=xlab,ylab=ylab)
    else
        p1 = plot(lw=2,dpi=dpi,xlab=xlab,ylab=ylab,ylim=ylim)
    end

    if plot_mag_1_uc .& ~plot_diff
        plot!(p1,tt,mag_1_uc,lab="mag_1_uc",color=:cyan)
    end

    fields  = fieldnames(typeof(xyz))
    list_uc = [Symbol("mag_",i,"_uc") for i = 1:num_mag_max]
    mags_uc = list_uc[ list_uc .∈ (fields,)]

    :all_mags in use_mags && (use_mags = mags_uc)

    for use_mag in use_mags

        if use_mag in mags_uc
            mag_uc  = getfield(xyz,use_mag)[ind]
            color   = :yellow
            use_mag == :mag_1_uc && (color = :red   )
            use_mag == :mag_2_uc && (color = :purple)
            use_mag == :mag_3_uc && (color = :green )
            use_mag == :mag_4_uc && (color = :black )
            use_mag == :mag_5_uc && (color = :orange)
            use_mag == :mag_6_uc && (color = :gray  )
            use_mag == :mag_7_uc && (color = :violet)
            use_mag == :mag_8_uc && (color = :brown )
            use_mag == :mag_9_uc && (color = :pink  )
        else
            error("$use_mag scalar magnetometer is not valid")
        end

        TL_coef = create_TL_coef(getfield(xyz_comp,use_vec),
                                 getfield(xyz_comp,use_mag),
                                 ind_comp;λ=λ,terms=terms,
                                 pass1=pass1,pass2=pass2,fs=fs)

        mag_c     = mag_uc - detrend(A*TL_coef;mean_only=true)
        mean_diff = mean(mag_c-xyz.mag_1_c[ind])
        detrend_data && (mag_c  = detrend(mag_c ))
        detrend_data && (mag_uc = detrend(mag_uc))

        lab = "$use_mag"[1:end-3]*"_c"
        lab = plot_diff ? "Δ $lab"      : lab
        val = plot_diff ? mag_c-mag_1_c : mag_c
        plot!(p1,tt[1:end-1],val[1:end-1],lab=lab,color=color)
        
        if plot_diff
            @info("=== $lab ===")
            println("avg diff = $(round(mean_diff,digits=3)) nT")
            println("std dev  = $(round( std(val),digits=3)) nT")
        end

    end

    if plot_mag_1_c .& ~plot_diff
        plot!(p1,tt[1:end-1],mag_1_c[1:end-1],
              lab="provided mag_1_c",ls= :dash,color=:blue)
    end

    show_plot && display(p1)
    save_plot && png(file_name)

    return (p1)
end # function plot_mag_c

"""
    plot_PSD(x::Vector, fs=10;
             window::Function  = hamming,
             dpi::Int          = 200,
             show_plot::Bool   = true,
             save_plot::Bool   = false,
             file_name::String = "PSD")

Plot the Welch power spectral density (PSD) for signal `x`.

**Arguments**
- `x`: input data
- `fs`:        (optional) sampling frequency [Hz]
- `window`:    (optional) type of window used
- `show_plot`: (optional) if true, plot will be shown
- `save_plot`: (optional) if true, plot will be saved
- `file_name`: (optional) plot file name to save

**Returns:**
- `p1`: plot of Welch power spectral density (PSD)
"""
function plot_PSD(x::Vector, fs=10;
                  window::Function  = hamming,
                  dpi::Int          = 200,
                  show_plot::Bool   = true,
                  save_plot::Bool   = false,
                  file_name::String = "PSD")

    psd_welch = welch_pgram(x,fs=fs,window=window)

    p1 = plot(psd_welch.freq,pow2db.(psd_welch.power),lab="",dpi=dpi,
              xlab="frequency [Hz]",ylab="power/frequency [dB/Hz]")

    show_plot && display(p1)
    save_plot && png(file_name)

    return (p1)
end # function plot_PSD

"""
    plot_spectrogram(x::Vector, fs=10;
                     window::Function  = hamming,
                     dpi::Int          = 200,
                     show_plot::Bool   = true,
                     save_plot::Bool   = false,
                     file_name::String = "spectrogram")

Create a spectrogram for signal `x`.

**Arguments**
- `x`: input data
- `fs`:        (optional) sampling frequency [Hz]
- `window`:    (optional) type of window used
- `show_plot`: (optional) if true, plot will be shown
- `save_plot`: (optional) if true, plot will be saved
- `file_name`: (optional) plot file name to save

**Returns:**
- `p1`: plot of spectrogram
"""
function plot_spectrogram(x::Vector, fs=10;
                          window::Function  = hamming,
                          dpi::Int          = 200,
                          show_plot::Bool   = true,
                          save_plot::Bool   = false,
                          file_name::String = "spectrogram")

    spec = spectrogram(x;fs=fs,window=window)

    p1 = heatmap(spec.time,spec.freq,pow2db.(spec.power),dpi=dpi,
                 xguide="time [s]",yguide="frequency [Hz]")

    show_plot && display(p1)
    save_plot && png(file_name)

    return (p1)
end # function plot_spectrogram

"""
    plot_frequency(xyz::XYZ;
                   ind                = trues(xyz.traj.N),
                   field::Symbol      = :mag_1_uc,
                   freq_type::Symbol  = :PSD,
                   detrend_data::Bool = true,
                   window::Function   = hamming,
                   dpi::Int           = 200,
                   show_plot::Bool    = true,
                   save_plot::Bool    = false,
                   file_name::String  = "PSD")

Plot frequency data, either Welch power spectral density (PSD) or spectrogram.

**Arguments:**
- `xyz`:          `XYZ` flight data struct
- `ind`:          (optional) selected data indices
- `field`:        (optional) data field in `xyz` to plot
- `freq_type`:    (optional) frequency plot type {`:PSD`,`:spec`}
- `detrend_data`: (optional) if true, plot data will be detrended
- `window`:       (optional) type of window used
- `show_plot`:    (optional) if true, plot will be shown
- `save_plot`:    (optional) if true, plot will be saved
- `file_name`:    (optional) plot file name to save

**Returns:**
- `p1`: plot of Welch power spectral density (PSD) or spectrogram
"""
function plot_frequency(xyz::XYZ;
                        ind                = trues(xyz.traj.N),
                        field::Symbol      = :mag_1_uc,
                        freq_type::Symbol  = :PSD,
                        detrend_data::Bool = true,
                        window::Function   = hamming,
                        dpi::Int           = 200,
                        show_plot::Bool    = true,
                        save_plot::Bool    = false,
                        file_name::String  = "PSD")

    x = getfield(xyz,field)[ind]

    detrend_data && (x = detrend(x))

    fs = 1/xyz.traj.dt

    f = freq_type == :PSD ? plot_PSD : plot_spectrogram

    p1 = f(x, fs;
           window    = window,
           dpi       = dpi,
           show_plot = show_plot,
           save_plot = save_plot,
           file_name = file_name)

    return (p1)
end # function plot_frequency

"""
    plot_correlation(xyz::XYZ,
                     xfield::Symbol  = :mag_1_c,
                     yfield::Symbol  = :mag_1_uc,
                     ind             = trues(xyz.traj.N);
                     lim             = 0,
                     dpi::Int        = 200,
                     show_plot::Bool = true,
                     save_plot::Bool = false,
                     silent::Bool    = true)

Plot the correlation between two features.

**Arguments:**
- `xyz`:       `XYZ` flight data struct
- `xfield`:    field name of x-axis feature
- `yfield`:    field name of y-axis feature
- `ind`:       (optional) selected data indices
- `lim`:       (optional) lower limit on the Pearson correlation coefficient (do not plot otherwise)
- `show_plot`: (optional) if true, plot will be shown
- `save_plot`: (optional) if true, plot will be saved
- `silent`:    (optional) if true, no print outs

**Returns:**
- `p1`: plot of correlation between two features
"""
function plot_correlation(xyz::XYZ,
                          xfield::Symbol  = :mag_1_c,
                          yfield::Symbol  = :mag_1_uc,
                          ind             = trues(xyz.traj.N);
                          lim             = 0,
                          dpi::Int        = 200,
                          show_plot::Bool = true,
                          save_plot::Bool = false,
                          silent::Bool    = true)

    x  = getfield(xyz,xfield)[ind]
    y  = getfield(xyz,yfield)[ind]
    p1 = plot_correlation(x,y,xfield,yfield;lim,dpi,show_plot,save_plot,silent)

    return (p1)
end # function plot_correlation

"""
    plot_correlation(x::Vector, y::Vector,
                     xfield::Symbol  = :feature_1,
                     yfield::Symbol  = :feature_2;
                     lim             = 0,
                     dpi::Int        = 200,
                     show_plot::Bool = true,
                     save_plot::Bool = false,
                     silent::Bool    = true)

Plot the correlation between two features.

**Arguments:**
- `x`: x-axis data
- `y`: y-axis data
- `xfield`: field name of x-axis feature
- `yfield`: field name of y-axis feature
"""
function plot_correlation(x::Vector, y::Vector,
                          xfield::Symbol  = :feature_1,
                          yfield::Symbol  = :feature_2;
                          lim             = 0,
                          dpi::Int        = 200,
                          show_plot::Bool = true,
                          save_plot::Bool = false,
                          silent::Bool    = true)

    xyc = cor(x,y)
    xys = linreg(y,x)

    if (abs(xyc) > lim) # & (abs(xys-1) < 1-lim)
        xlab  = "$xfield"
        ylab  = "$yfield"
        title = xlab*" & "*ylab
        p1 = scatter(x,y,lab=false,dpi=dpi,xlab=xlab,ylab=ylab,title=title;ms=2)
        show_plot && display(p1)
        save_plot && png(title)
        silent || println(xfield," & ",yfield)
        silent || println("correlation & slope: $(round.([xyc,xys],digits=5))")
        return (p1)
    end

end # function plot_correlation
