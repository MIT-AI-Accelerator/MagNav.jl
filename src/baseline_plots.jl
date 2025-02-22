"""
    plot_basic(tt::Vector, y::Vector, ind = trues(length(tt));
               lab::String      = "",
               xlab::String     = "time [min]",
               ylab::String     = "",
               show_plot::Bool  = true,
               save_plot::Bool  = false,
               plot_png::String = "data_vs_time.png")

Plot data vs time.

**Arguments:**
- `tt`:        length-`N` time vector [s]
- `y`:         length-`N` data vector
- `ind`:       (optional) selected data indices
- `lab`:       (optional) data (legend) label
- `xlab`:      (optional) x-axis label
- `ylab`:      (optional) y-axis label
- `show_plot`: (optional) if true, show `p1`
- `save_plot`: (optional) if true, save `p1` as `plot_png`
- `plot_png`:  (optional) plot file name to save (`.png` extension optional)

**Returns:**
- `p1`: plot of `y` vs `tt`
"""
function plot_basic(tt::Vector, y::Vector, ind = trues(length(tt));
                    lab::String      = "",
                    xlab::String     = "time [min]",
                    ylab::String     = "",
                    show_plot::Bool  = true,
                    save_plot::Bool  = false,
                    plot_png::String = "data_vs_time.png")

    p1 = plot((tt[ind] .- tt[ind][1]) / 60, y[ind],lab=lab,xlab=xlab,ylab=ylab)

    show_plot && display(p1)
    save_plot && png(p1,plot_png)

    return (p1)
end # function plot_basic

"""
    plot_activation(activation = [:relu,:σ,:swish,:tanh];
                    plot_deriv::Bool  = false,
                    show_plot::Bool   = true,
                    save_plot::Bool   = false,
                    plot_png::String  = "act_func.png")

Plot activation function(s) or their derivative(s).

**Arguments:**
- `activation`: activation function(s) to plot
    - `relu`  = rectified linear unit
    - `σ`     = sigmoid (logistic function)
    - `swish` = self-gated
    - `tanh`  = hyperbolic tan
- `plot_deriv`: (optional) if true, plot activation function(s) derivative(s)
- `show_plot`:  (optional) if true, show `p1`
- `save_plot`:  (optional) if true, save `p1` as `plot_png`
- `plot_png`:   (optional) plot file name to save (`.png` extension optional)

**Returns:**
- `p1`: plot of activation function(s) or their derivative(s)
"""
function plot_activation(activation = [:relu,:σ,:swish,:tanh];
                         plot_deriv::Bool  = false,
                         show_plot::Bool   = true,
                         save_plot::Bool   = false,
                         plot_png::String  = "act_func.png")

    save_plot ? dpi = 500 : dpi = 200

    # setup
    x = LinRange(-6,4,1000)
    p_relu  = relu.(x)
    p_σ     = σ.(x)
    p_swish = swish.(x)
    p_tanh  = tanh.(x)

    d_relu  = ForwardDiff.derivative.(relu,x)
    d_σ     = ForwardDiff.derivative.(σ,x)
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

    show_plot && display(p1)
    save_plot && png(p1,plot_png)

    return (p1)
end # function plot_activation

"""
    plot_mag(xyz::XYZ;
             ind                       = trues(xyz.traj.N),
             detrend_data::Bool        = false,
             use_mags::Vector{Symbol}  = [:all_mags],
             vec_terms::Vector{Symbol} = [:all],
             ylim::Tuple               = (),
             dpi::Int                  = 200,
             show_plot::Bool           = true,
             save_plot::Bool           = false,
             plot_png::String          = "scalar_mags.png")

Plot scalar or vector (fluxgate) magnetometer data from a given flight test.

**Arguments:**
- `xyz`:          `XYZ` flight data struct
- `ind`:          (optional) selected data indices
- `detrend_data`: (optional) if true, detrend plot data
- `use_mags`:     (optional) scalar or vector (fluxgate) magnetometers to plot {`:all_mags`, `:comp_mags` or `:mag_1_c`, `:mag_1_uc`, `:flux_a`, etc.}
    - `:all_mags`  = all provided scalar magnetometer fields (e.g., `:mag_1_c`, `:mag_1_uc`, etc.)
    - `:comp_mags` = provided compensation(s) between `:mag_1_uc` & `:mag_1_c`, etc.
- `vec_terms`:    (optional) vector magnetometer (fluxgate) terms to plot {`:all` or `:x`,`:y`,`:z`,`:t`}
- `ylim`:         (optional) length-`2` plot `y` limits (`ymin`,`ymax`) [nT]
- `dpi`:          (optional) dots per inch (image resolution)
- `show_plot`:    (optional) if true, show `p1`
- `save_plot`:    (optional) if true, save `p1` as `plot_png`
- `plot_png`:     (optional) plot file name to save (`.png` extension optional)

**Returns:**
- `p1`: plot of scalar or vector (fluxgate) magnetometer data
"""
function plot_mag(xyz::XYZ;
                  ind                       = trues(xyz.traj.N),
                  detrend_data::Bool        = false,
                  use_mags::Vector{Symbol}  = [:all_mags],
                  vec_terms::Vector{Symbol} = [:all],
                  ylim::Tuple               = (),
                  dpi::Int                  = 200,
                  show_plot::Bool           = true,
                  save_plot::Bool           = false,
                  plot_png::String          = "scalar_mags.png")

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
    save_plot && png(p1,plot_png)

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
               plot_png::String         = "scalar_mags_comp.png")

Plot compensated magnetometer(s) data from a given flight test. Assumes Mag 1
(i.e., `:mag_1_uc` & `:mag_1_c`) is the best magnetometer (i.e., stinger).

**Arguments:**
- `xyz`:           `XYZ` flight data struct
- `xyz_comp`:      `XYZ` flight data struct to use for compensation
- `ind`:           (optional) selected data indices
- `ind_comp`:      (optional) selected data indices to use for compensation
- `detrend_data`:  (optional) if true, detrend plot data
- `λ`:             (optional) ridge parameter
- `terms`:         (optional) Tolles-Lawson terms to use {`:permanent`,`:induced`,`:eddy`,`:bias`}
- `pass1`:         (optional) filter first passband frequency [Hz]
- `pass2`:         (optional) filter second passband frequency [Hz]
- `fs`:            (optional) filter sampling frequency [Hz]
- `use_mags`:      (optional) scalar or vector (fluxgate) magnetometers to plot {`:all_mags`, or `:mag_1_uc`, etc.}
    - `:all_mags` = all uncompensated scalar magnetometer fields (e.g., `:mag_1_uc`, etc.)
- `use_vec`:       (optional) vector magnetometer (fluxgate) to use for Tolles-Lawson `A` matrix {`:flux_a`, etc.}
- `plot_diff`:     (optional) if true, plot difference between `provided` compensated data & compensated mags `as performed here`
- `plot_mag_1_uc`: (optional) if true, plot mag_1_uc (uncompensated mag_1)
- `plot_mag_1_c`:  (optional) if true, plot mag_1_c (compensated mag_1)
- `ylim`:          (optional) length-`2` plot `y` limits (`ymin`,`ymax`) [nT]
- `dpi`:           (optional) dots per inch (image resolution)
- `show_plot`:     (optional) if true, show `p1`
- `save_plot`:     (optional) if true, save `p1` as `plot_png`
- `plot_png`:      (optional) plot file name to save (`.png` extension optional)

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
                    plot_png::String         = "scalar_mags_comp.png")

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
        plot!(p1,tt,mag_1_uc,lab="mag_1_uc",lc=:cyan)
    end

    fields  = fieldnames(typeof(xyz))
    list_uc = [Symbol("mag_",i,"_uc") for i = 1:num_mag_max]
    mags_uc = list_uc[ list_uc .∈ (fields,)]

    :all_mags in use_mags && (use_mags = mags_uc)

    for use_mag in use_mags

        if use_mag in mags_uc
            mag_uc  = getfield(xyz,use_mag)[ind]
            lc      = :yellow
            use_mag == :mag_1_uc && (lc = :red   )
            use_mag == :mag_2_uc && (lc = :purple)
            use_mag == :mag_3_uc && (lc = :green )
            use_mag == :mag_4_uc && (lc = :black )
            use_mag == :mag_5_uc && (lc = :orange)
            use_mag == :mag_6_uc && (lc = :gray  )
            use_mag == :mag_7_uc && (lc = :violet)
            use_mag == :mag_8_uc && (lc = :brown )
            use_mag == :mag_9_uc && (lc = :pink  )
        else
            error("$use_mag scalar magnetometer is invalid")
        end

        TL_coef = create_TL_coef(getfield(xyz_comp,use_vec),
                                 getfield(xyz_comp,use_mag),
                                 ind_comp;λ=λ,terms=terms,
                                 pass1=pass1,pass2=pass2,fs=fs)

        mag_c     = mag_uc - detrend(A*TL_coef;mean_only=true)
        mean_diff = mean(mag_c - xyz.mag_1_c[ind])
        detrend_data && (mag_c  = detrend(mag_c ))
        detrend_data && (mag_uc = detrend(mag_uc))

        lab = "$use_mag"[1:end-3]*"_c"
        lab = plot_diff ? "Δ $lab"        : lab
        val = plot_diff ? mag_c - mag_1_c : mag_c
        plot!(p1,tt[1:end-1],val[1:end-1],lab=lab,lc=lc)

        if plot_diff
            @info("=== $lab ===")
            println("avg diff = $(round(mean_diff,digits=3)) nT")
            println("std dev  = $(round( std(val),digits=3)) nT")
        end

    end

    if plot_mag_1_c .& ~plot_diff
        plot!(p1,tt[1:end-1],mag_1_c[1:end-1],
              lab="provided mag_1_c",lc=:blue,ls=:dash)
    end

    show_plot && display(p1)
    save_plot && png(p1,plot_png)

    return (p1)
end # function plot_mag_c

"""
    plot_PSD(x::Vector, fs=10;
             window::Function  = hamming,
             dpi::Int          = 200,
             show_plot::Bool   = true,
             save_plot::Bool   = false,
             plot_png::String  = "PSD.png")

Plot the Welch power spectral density (PSD) for signal `x`.

**Arguments:**
- `x`:         data vector
- `fs`:        (optional) sampling frequency [Hz]
- `window`:    (optional) type of window used
- `dpi`:       (optional) dots per inch (image resolution)
- `show_plot`: (optional) if true, show `p1`
- `save_plot`: (optional) if true, save `p1` as `plot_png`
- `plot_png`:  (optional) plot file name to save (`.png` extension optional)

**Returns:**
- `p1`: plot of Welch power spectral density (PSD)
"""
function plot_PSD(x::Vector, fs=10;
                  window::Function  = hamming,
                  dpi::Int          = 200,
                  show_plot::Bool   = true,
                  save_plot::Bool   = false,
                  plot_png::String  = "PSD.png")

    p = welch_pgram(x,fs=fs,window=window)

    p1 = plot(p.freq,pow2db.(p.power),lab="",dpi=dpi,
              xlab="frequency [Hz]",ylab="power/frequency [dB/Hz]")

    show_plot && display(p1)
    save_plot && png(p1,plot_png)

    return (p1)
end # function plot_PSD

"""
    plot_spectrogram(x::Vector, fs=10;
                     window::Function  = hamming,
                     dpi::Int          = 200,
                     show_plot::Bool   = true,
                     save_plot::Bool   = false,
                     plot_png::String  = "spectrogram.png")

Create a spectrogram for signal `x`.

**Arguments:**
- `x`:         data vector
- `fs`:        (optional) sampling frequency [Hz]
- `window`:    (optional) type of window used
- `dpi`:       (optional) dots per inch (image resolution)
- `show_plot`: (optional) if true, show `p1`
- `save_plot`: (optional) if true, save `p1` as `plot_png`
- `plot_png`:  (optional) plot file name to save (`.png` extension optional)

**Returns:**
- `p1`: plot of spectrogram
"""
function plot_spectrogram(x::Vector, fs=10;
                          window::Function  = hamming,
                          dpi::Int          = 200,
                          show_plot::Bool   = true,
                          save_plot::Bool   = false,
                          plot_png::String  = "spectrogram.png")

    s = spectrogram(x;fs=fs,window=window)

    p1 = heatmap(s.time,s.freq,pow2db.(s.power),dpi=dpi,
                 xguide="time [s]",yguide="frequency [Hz]")

    show_plot && display(p1)
    save_plot && png(p1,plot_png)

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
                   plot_png::String   = "PSD.png")

Plot frequency data, either Welch power spectral density (PSD) or spectrogram.

**Arguments:**
- `xyz`:          `XYZ` flight data struct
- `ind`:          (optional) selected data indices
- `field`:        (optional) data field in `xyz` to plot
- `freq_type`:    (optional) frequency plot type {`:PSD`,`:psd`,`:spectrogram`,`:spec`}
- `detrend_data`: (optional) if true, detrend plot data
- `window`:       (optional) type of window used
- `dpi`:          (optional) dots per inch (image resolution)
- `show_plot`:    (optional) if true, show `p1`
- `save_plot`:    (optional) if true, save `p1` as `plot_png`
- `plot_png`:     (optional) plot file name to save (`.png` extension optional)

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
                        plot_png::String   = "PSD.png")

    x = getfield(xyz,field)[ind]

    detrend_data && (x = detrend(x))

    fs = 1/xyz.traj.dt

    f = freq_type in [:PSD,:psd] ? plot_PSD : plot_spectrogram

    p1 = f(x,fs;
           window    = window,
           dpi       = dpi,
           show_plot = show_plot,
           save_plot = save_plot,
           plot_png  = plot_png)

    return (p1)
end # function plot_frequency

"""
    plot_correlation(x::Vector, y::Vector,
                     xfeature::Symbol = :feature_1,
                     yfeature::Symbol = :feature_2;
                     lim::Real        = 0,
                     dpi::Int         = 200,
                     show_plot::Bool  = true,
                     save_plot::Bool  = false,
                     plot_png::String = "\$xfeature-\$yfeature.png",
                     silent::Bool     = true)

Plot the correlation between 2 features.

**Arguments:**
- `x`:         x-axis data
- `y`:         y-axis data
- `xfeature`:  x-axis feature name
- `yfeature`:  y-axis feature name
- `lim`:       (optional) only plot if Pearson correlation coefficient > `lim`
- `dpi`:       (optional) dots per inch (image resolution)
- `show_plot`: (optional) if true, show `p1`
- `save_plot`: (optional) if true, save `p1` as `plot_png`
- `plot_png`:  (optional) plot file name to save (`.png` extension optional)
- `silent`:    (optional) if true, no print outs

**Returns:**
- `p1`: plot of `yfeature` vs `xfeature` correlation
"""
function plot_correlation(x::Vector, y::Vector,
                          xfeature::Symbol = :feature_1,
                          yfeature::Symbol = :feature_2;
                          lim::Real        = 0,
                          dpi::Int         = 200,
                          show_plot::Bool  = true,
                          save_plot::Bool  = false,
                          plot_png::String = "$xfeature-$yfeature.png",
                          silent::Bool     = true)

    xyc   = cor(x,y)
    xys   = linreg(y,x)
    xlab  = "$xfeature"
    ylab  = "$yfeature"
    title = "$yfeature vs $xfeature"
    silent || println("$title, correlation & slope: $(round.([xyc,xys],digits=5))")

    if abs(xyc) > lim
        p1 = scatter(x,y,lab=false,dpi=dpi,title=title,
                     xlab=xlab,ylab=ylab,mc=:black,ms=2)
        show_plot && display(p1)
        save_plot && png(p1,plot_png)
        return (p1)
    else
        return (nothing)
    end

end # function plot_correlation

"""
    plot_correlation(xyz::XYZ,
                     xfeature::Symbol = :mag_1_c,
                     yfeature::Symbol = :mag_1_uc,
                     ind              = trues(xyz.traj.N);
                     lim::Real        = 0,
                     dpi::Int         = 200,
                     show_plot::Bool  = true,
                     save_plot::Bool  = false,
                     plot_png::String = "\$xfeature-\$yfeature.png",
                     silent::Bool     = true)

Plot the correlation between 2 features.

**Arguments:**
- `xyz`:       `XYZ` flight data struct
- `xfeature`:  x-axis feature name
- `yfeature`:  y-axis feature name
- `ind`:       (optional) selected data indices
- `lim`:       (optional) only plot if Pearson correlation coefficient > `lim`
- `dpi`:       (optional) dots per inch (image resolution)
- `show_plot`: (optional) if true, show `p1`
- `save_plot`: (optional) if true, save `p1` as `plot_png`
- `plot_png`:  (optional) plot file name to save (`.png` extension optional)
- `silent`:    (optional) if true, no print outs

**Returns:**
- `p1`: plot of `yfeature` vs `xfeature` correlation
"""
function plot_correlation(xyz::XYZ,
                          xfeature::Symbol = :mag_1_c,
                          yfeature::Symbol = :mag_1_uc,
                          ind              = trues(xyz.traj.N);
                          lim::Real        = 0,
                          dpi::Int         = 200,
                          show_plot::Bool  = true,
                          save_plot::Bool  = false,
                          plot_png::String = "$xfeature-$yfeature.png",
                          silent::Bool     = true)
    x = getfield(xyz,xfeature)[ind]
    y = getfield(xyz,yfeature)[ind]
    plot_correlation(x,y,xfeature,yfeature;
                     lim       = lim,
                     dpi       = dpi,
                     show_plot = show_plot,
                     save_plot = save_plot,
                     plot_png  = plot_png,
                     silent    = silent)
end # function plot_correlation

"""
    plot_correlation_matrix(x::AbstractMatrix, features::Vector{Symbol};
                            dpi::Int         = 200,
                            Nmax::Int        = 1000,
                            show_plot::Bool  = true,
                            save_plot::Bool  = false,
                            plot_png::String = "correlation_matrix.png")

Plot the correlation matrix for `2-5` features.

**Arguments:**
- `x`:         `N` x `Nf` data matrix (`Nf` is number of features)
- `features`:  length-`Nf` feature vector (including components of TL `A`, etc.)
- `dpi`:       (optional) dots per inch (image resolution)
- `Nmax`:      (optional) maximum number of data points plotted
- `show_plot`: (optional) if true, show `p1`
- `save_plot`: (optional) if true, save `p1` as `plot_png`
- `plot_png`:  (optional) plot file name to save (`.png` extension optional)

**Returns:**
- `p1`: plot of correlation matrix between `features`
"""
function plot_correlation_matrix(x::AbstractMatrix, features::Vector{Symbol};
                                 dpi::Int         = 200,
                                 Nmax::Int        = 1000,
                                 show_plot::Bool  = true,
                                 save_plot::Bool  = false,
                                 plot_png::String = "correlation_matrix.png")

    # #* note: this could be modified to use StatsPlots.corrplot():
    # https://github.com/JuliaPlots/StatsPlots.jl

    Nf = length(features)
    (Nf_min,Nf_max) = (2,5)
    @assert Nf >= Nf_min "number of features = $Nf < $Nf_min"
    @assert Nf <= Nf_max "number of features = $Nf > $Nf_max"

    p_ = []
    for j = 2:Nf, i = 1:Nf-1
        if j > i
            xlab   = j == Nf ? features[i] : ""
            ylab   = i == 1  ? features[j] : ""
            x_     = x[:,i]
            y_     = x[:,j]
            xticks = j == Nf ? round.(mean(x_) .+ [-1,1]*std(x_),sigdigits=3) : []
            yticks = i == 1  ? round.(mean(y_) .+ [-1,1]*std(y_),sigdigits=3) : []
            push!(p_,scatter(downsample(x_,Nmax),downsample(y_,Nmax),
                  lab=false,dpi=dpi,mc=:black,ms=1,
                  xlab=xlab,ylab=ylab,xticks=xticks,yticks=yticks,
                  xguidefontsize=8,yguidefontsize=8))
        end
    end

    if Nf == 2
        l = @layout [p;]
    elseif Nf == 3
        l = @layout [p _; p p]
    elseif Nf == 4
        l = @layout [p _ _; p p _; p p p]
    elseif Nf == 5
        l = @layout [p _ _ _; p p _ _; p p p _; p p p p]
    end

    p1 = plot(p_...,layout=l,margin=2*mm)

    show_plot && display(p1)
    save_plot && png(p1,plot_png)

    return (p1)
end # function plot_correlation_matrix

"""
    plot_correlation_matrix(xyz::XYZ, ind = trues(xyz.traj.N),
                            features_setup::Vector{Symbol} = [:mag_1_uc,:TL_A_flux_a];
                            terms             = [:permanent],
                            sub_diurnal::Bool = false,
                            sub_igrf::Bool    = false,
                            bpf_mag::Bool     = false,
                            dpi::Int          = 200,
                            Nmax::Int         = 1000,
                            show_plot::Bool   = true,
                            save_plot::Bool   = false,
                            plot_png::String  = "correlation_matrix.png")

Plot the correlation matrix for `2-5` features.

**Arguments:**
- `xyz`:            `XYZ` flight data struct
- `ind`:            selected data indices
- `features_setup`: vector of features to include
- `terms`:          (optional) Tolles-Lawson terms to use {`:permanent`,`:induced`,`:eddy`,`:bias`}
- `sub_diurnal`:    (optional) if true, subtract diurnal from scalar magnetometer measurements
- `sub_igrf`:       (optional) if true, subtract IGRF from scalar magnetometer measurements
- `bpf_mag`:        (optional) if true, bpf scalar magnetometer measurements
- `dpi`:            (optional) dots per inch (image resolution)
- `Nmax`:           (optional) maximum number of data points plotted
- `show_plot`:      (optional) if true, show `p1`
- `save_plot`:      (optional) if true, save `p1` as `plot_png`
- `plot_png`:       (optional) plot file name to save (`.png` extension optional)

**Returns:**
- `p1`: plot of correlation matrix between `features` (created from `features_setup`)
"""
function plot_correlation_matrix(xyz::XYZ, ind = trues(xyz.traj.N),
                                 features_setup::Vector{Symbol} = [:mag_1_uc,:TL_A_flux_a];
                                 terms             = [:permanent],
                                 sub_diurnal::Bool = false,
                                 sub_igrf::Bool    = false,
                                 bpf_mag::Bool     = false,
                                 dpi::Int          = 200,
                                 Nmax::Int         = 1000,
                                 show_plot::Bool   = true,
                                 save_plot::Bool   = false,
                                 plot_png::String  = "correlation_matrix.png")

    (x,_,features,_) = get_x(xyz,ind,features_setup;
                             terms       = terms,
                             sub_diurnal = sub_diurnal,
                             sub_igrf    = sub_igrf,
                             bpf_mag     = bpf_mag)

    p1 = plot_correlation_matrix(x,features;
                                 dpi       = dpi,
                                 Nmax      = Nmax,
                                 show_plot = show_plot,
                                 save_plot = save_plot,
                                 plot_png  = plot_png)

    return (p1)
end # function plot_correlation_matrix
