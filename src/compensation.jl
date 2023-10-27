"""
    nn_comp_1_train(x, y, no_norm;
                    norm_type_x::Symbol  = :standardize,
                    norm_type_y::Symbol  = :standardize,
                    η_adam               = 0.001,
                    epoch_adam::Int      = 5,
                    epoch_lbfgs::Int     = 0,
                    hidden               = [8],
                    activation::Function = swish,
                    batchsize::Int       = 2048,
                    frac_train           = 14/17,
                    α_sgl                = 1,
                    λ_sgl                = 0,
                    k_pca::Int           = -1,
                    data_norms::Tuple    = (zeros(Float32,1,1),zeros(Float32,1,1),zeros(Float32,1,1),zeros(Float32,1,1),zeros(Float32,1,1),[0f0],[0f0]),
                    model::Chain         = Chain(),
                    l_segs::Vector       = [length(y)],
                    x_test::Matrix       = Matrix{eltype(x)}(undef,0,0),
                    y_test::Vector       = Vector{eltype(y)}(undef,0),
                    silent::Bool         = false)

Train neural network-based aeromagnetic compensation, model 1.
"""
function nn_comp_1_train(x, y, no_norm;
                         norm_type_x::Symbol  = :standardize,
                         norm_type_y::Symbol  = :standardize,
                         η_adam               = 0.001,
                         epoch_adam::Int      = 5,
                         epoch_lbfgs::Int     = 0,
                         hidden               = [8],
                         activation::Function = swish,
                         batchsize::Int       = 2048,
                         frac_train           = 14/17,
                         α_sgl                = 1,
                         λ_sgl                = 0,
                         k_pca::Int           = -1,
                         data_norms::Tuple    = (zeros(Float32,1,1),zeros(Float32,1,1),zeros(Float32,1,1),zeros(Float32,1,1),zeros(Float32,1,1),[0f0],[0f0]),
                         model::Chain         = Chain(),
                         l_segs::Vector       = [length(y)],
                         x_test::Matrix       = Matrix{eltype(x)}(undef,0,0),
                         y_test::Vector       = Vector{eltype(y)}(undef,0),
                         silent::Bool         = false)

    # convert to Float32 for ~50% speedup
    x      = Float32.(x)
    y      = Float32.(y)
    α      = Float32.(α_sgl)
    λ      = Float32.(λ_sgl)
    x_test = Float32.(x_test)
    y_test = Float32.(y_test)
    l_segs_test = [length(y_test)]

    Nf = size(x,2) # number of features

    if sum(data_norms[end]) == 0 # normalize data
        (x_bias,x_scale,x_norm) = norm_sets(x;norm_type=norm_type_x,no_norm=no_norm)
        (y_bias,y_scale,y_norm) = norm_sets(y;norm_type=norm_type_y)
        if k_pca > 0
            if k_pca > Nf
                silent || @info("reducing k_pca from $k_pca to $Nf")
                k_pca = Nf
            end
            (_,S,V) = svd(cov(x_norm))
            v_scale = V[:,1:k_pca]*inv(Diagonal(sqrt.(S[1:k_pca])))
            var_ret = round(sum(sqrt.(S[1:k_pca]))/sum(sqrt.(S))*100,digits=6)
            silent || @info("k_pca = $k_pca of $Nf, variance retained: $var_ret %")
        else
            v_scale = I(Nf)
        end
        x_norm = (x_norm * v_scale)'
        y_norm = (y_norm          )'
    else # unpack data normalizations
        (_,_,v_scale,x_bias,x_scale,y_bias,y_scale) = unpack_data_norms(data_norms)
        x_norm = (((x .- x_bias) ./ x_scale) * v_scale)'
        y_norm = ( (y .- y_bias) ./ y_scale           )'
    end

    # normalize test data if it was provided
    isempty(x_test) || (x_test_norm = (((x_test .- x_bias) ./ x_scale) * v_scale)')

    # separate into training and validation
    if frac_train < 1
        N = size(x_norm,2)
        (p_train,p_val) = get_split(N,frac_train,:none)
        x_train_norm    = x_norm[:,p_train]
        x_val_norm      = x_norm[:,p_val  ]
        y_train_norm    = y_norm[:,p_train]
        y_val_norm      = y_norm[:,p_val  ]
        data_train      = DataLoader((x_train_norm,y_train_norm),
                                     shuffle=true,batchsize=batchsize)
        data_val        = DataLoader((x_val_norm,y_val_norm),
                                     shuffle=true,batchsize=batchsize)
    else
        data_train      = DataLoader((x_norm,y_norm),
                                     shuffle=true,batchsize=batchsize)
        data_val        = data_train
    end

    # setup NN
    if model == Chain() # initialize model
        Ny = 1 # length of output
        m  = get_nn_m(Nf,Ny;hidden=hidden,activation=activation)
    else # re-train on known model
        m  = deepcopy(model)
    end

    # setup optimizer
    opt = Adam(η_adam)

    # setup loss function
    function loss_m1(x_norm, y_norm)
        y_hat_norm = nn_comp_1_fwd(x_norm,y_bias,y_scale,m;
                                   denorm   = false,
                                   testmode = false)
        return mse(y_hat_norm, vec(y_norm))
    end # function loss_m1

    loss_m1_λ(xl,yl) = loss_m1(xl,yl) + λ*sum(sparse_group_lasso(m,α))
    loss = λ > 0 ? loss_m1_λ : loss_m1

    function loss_all(data_l)
        l = 0f0
        for (x_l,y_l) in data_l
            l += loss(x_l,y_l)
        end
        return (l/length(data_l))
    end # function loss_all

    # train NN with Adam optimizer
    m_store   = deepcopy(m)
    best_loss = loss_all(data_val)
    isempty(x_test) || (best_test_error = std(nn_comp_1_test(x_test_norm,y_test,y_bias,y_scale,m;
                                                             l_segs = l_segs_test,
                                                             silent = silent_debug)[2]))

    silent || @info("epoch 0: loss = $best_loss")
    for i = 1:epoch_adam
        Flux.train!(loss,Flux.params(m),data_train,opt)
        current_loss = loss_all(data_val)
        # update NN model weights for lowest validation loss or lowest test error
        if isempty(x_test)
            if current_loss < best_loss
                best_loss = current_loss
                m_store   = deepcopy(m)
            end
            mod(i,5) == 0 && !silent && @info("epoch $i: loss = $best_loss")
        else
            test_error = std(nn_comp_1_test(x_test_norm,y_test,y_bias,y_scale,m;
                                            l_segs = l_segs_test,
                                            silent = silent_debug)[2])
            if test_error < best_test_error
                best_test_error = test_error
                m_store         = deepcopy(m)
            end
            mod(i,5) == 0 && !silent && @info("epoch $i: loss = $current_loss, test error = $(round(best_test_error,digits=2)) nT")
        end
        if (mod(i,10) == 0) & !silent
            err = nn_comp_1_test(x_norm,y,y_bias,y_scale,m;l_segs=l_segs,silent=true)[2]
            @info("$i train error: $(round(std(err),digits=2)) nT")
            isempty(x_test) || @info("$i test  error: $(round((test_error),digits=2)) nT")
        end
    end

    Flux.loadmodel!(m,m_store)

    if epoch_lbfgs > 0 # LBFGS, may overfit depending on iterations
        data = DataLoader((x_norm,y_norm),shuffle=true,batchsize=batchsize)

        function lbfgs_train!(m_l,data_l,iter,silent)
            (x_l,y_l) = data_l.data
            loss_() = loss_m1(x_l,y_l)
            refresh()
            params = Flux.params(m_l)
            opt = LBFGS()
            (_,_,fg!,p0) = optfuns(loss_,params)
            options = Options(iterations=iter,show_every=5,show_trace=!silent)
            optimize(only_fg!(fg!),p0,opt,options)
        end # function lbfgs_train!

        # train NN with LBFGS optimizer
        lbfgs_train!(m,data,epoch_lbfgs,silent)
    end

    # get results
    (y_hat,err) = nn_comp_1_test(x_norm,y,y_bias,y_scale,m;l_segs=l_segs,silent=true)
    silent || @info("train error: $(round(std(err),digits=2)) nT")

    if !isempty(x_test)
        nn_comp_1_test(x_test_norm,y_test,y_bias,y_scale,m;
                       l_segs = l_segs_test,
                       silent = silent)
    end

    model = m

    # pack data normalizations
    data_norms = (zeros(1,1),zeros(1,1),v_scale,x_bias,x_scale,y_bias,y_scale)

    return (model, data_norms, y_hat, err)
end # function nn_comp_1_train

"""
    nn_comp_1_fwd(x_norm::AbstractMatrix, y_bias, y_scale, model::Chain;
                  denorm::Bool   = true,
                  testmode::Bool = true)

Forward pass of neural network-based aeromagnetic compensation, model 1.
"""
function nn_comp_1_fwd(x_norm::AbstractMatrix, y_bias, y_scale, model::Chain;
                       denorm::Bool   = true,
                       testmode::Bool = true)

    # set to test mode in case model uses batchnorm or dropout
    m = model
    testmode && Flux.testmode!(m)

    # get results
    y_hat = vec(m(x_norm))

    denorm && (y_hat .= denorm_sets(y_bias,y_scale,y_hat))

    return (y_hat)
end # function nn_comp_1_fwd

"""
    nn_comp_1_fwd(x::Matrix, data_norms::Tuple, model::Chain)

Forward pass of neural network-based aeromagnetic compensation, model 1.
"""
function nn_comp_1_fwd(x::Matrix, data_norms::Tuple, model::Chain)

    # convert to Float32 for consistency with nn_comp_1_train
    x = Float32.(x)

    # unpack data normalizations
    (_,_,v_scale,x_bias,x_scale,y_bias,y_scale) = unpack_data_norms(data_norms)
    x_norm = (((x .- x_bias) ./ x_scale) * v_scale)'

    # get results
    y_hat = nn_comp_1_fwd(x_norm,y_bias,y_scale,model)

    return (y_hat)
end # function nn_comp_1_fwd

"""
    nn_comp_1_test(x_norm::AbstractMatrix, y, y_bias, y_scale, model::Chain;
                   l_segs::Vector = [length(y)],
                   silent::Bool   = false)

Evaluate performance of neural network-based aeromagnetic compensation, model 1.
"""
function nn_comp_1_test(x_norm::AbstractMatrix, y, y_bias, y_scale, model::Chain;
                        l_segs::Vector = [length(y)],
                        silent::Bool   = false)

    # get results
    y_hat = nn_comp_1_fwd(x_norm,y_bias,y_scale,model)
    err   = err_segs(y_hat,y,l_segs;silent=silent_debug)
    silent || @info("test  error: $(round(std(err),digits=2)) nT")

    return (y_hat, err)
end # function nn_comp_1_test

"""
    nn_comp_1_test(x::Matrix, y, data_norms::Tuple, model::Chain;
                   l_segs::Vector = [length(y)],
                   silent::Bool   = false)

Evaluate performance of neural network-based aeromagnetic compensation, model 1.
"""
function nn_comp_1_test(x::Matrix, y, data_norms::Tuple, model::Chain;
                        l_segs::Vector = [length(y)],
                        silent::Bool   = false)

    # convert to Float32 for consistency with nn_comp_1_train
    y = Float32.(y)

    # get results
    y_hat = nn_comp_1_fwd(x,data_norms,model)
    err   = err_segs(y_hat,y,l_segs;silent=silent_debug)
    silent || @info("test  error: $(round(std(err),digits=2)) nT")

    return (y_hat, err)
end # function nn_comp_1_test

"""
    nn_comp_2_train(A, x, y, no_norm;
                    model_type::Symbol   = :m2a,
                    norm_type_A::Symbol  = :none,
                    norm_type_x::Symbol  = :standardize,
                    norm_type_y::Symbol  = :standardize,
                    TL_coef::Vector      = zeros(Float32,18),
                    η_adam               = 0.001,
                    epoch_adam::Int      = 5,
                    epoch_lbfgs::Int     = 0,
                    hidden               = [8],
                    activation::Function = swish,
                    batchsize::Int       = 2048,
                    frac_train           = 14/17,
                    α_sgl                = 1,
                    λ_sgl                = 0,
                    k_pca::Int           = -1,
                    data_norms::Tuple    = (zeros(Float32,1,1),zeros(Float32,1,1),zeros(Float32,1,1),zeros(Float32,1,1),zeros(Float32,1,1),[0f0],[0f0]),
                    model::Chain         = Chain(),
                    l_segs::Vector       = [length(y)],
                    A_test::Matrix       = Matrix{eltype(A)}(undef,0,0),
                    x_test::Matrix       = Matrix{eltype(x)}(undef,0,0),
                    y_test::Vector       = Vector{eltype(y)}(undef,0),
                    silent::Bool         = false)

Train neural network-based aeromagnetic compensation, model 2.
"""
function nn_comp_2_train(A, x, y, no_norm;
                         model_type::Symbol   = :m2a,
                         norm_type_A::Symbol  = :none,
                         norm_type_x::Symbol  = :standardize,
                         norm_type_y::Symbol  = :none,
                         TL_coef::Vector      = zeros(Float32,18),
                         η_adam               = 0.001,
                         epoch_adam::Int      = 5,
                         epoch_lbfgs::Int     = 0,
                         hidden               = [8],
                         activation::Function = swish,
                         batchsize::Int       = 2048,
                         frac_train           = 14/17,
                         α_sgl                = 1,
                         λ_sgl                = 0,
                         k_pca::Int           = -1,
                         data_norms::Tuple    = (zeros(Float32,1,1),zeros(Float32,1,1),zeros(Float32,1,1),zeros(Float32,1,1),zeros(Float32,1,1),[0f0],[0f0]),
                         model::Chain         = Chain(),
                         l_segs::Vector       = [length(y)],
                         A_test::Matrix       = Matrix{eltype(A)}(undef,0,0),
                         x_test::Matrix       = Matrix{eltype(x)}(undef,0,0),
                         y_test::Vector       = Vector{eltype(y)}(undef,0),
                         silent::Bool         = false)

    # convert to Float32 for ~50% speedup
    A       = Float32.(A)
    x       = Float32.(x)
    y       = Float32.(y)
    α       = Float32.(α_sgl)
    λ       = Float32.(λ_sgl)
    A_test  = Float32.(A_test)
    x_test  = Float32.(x_test)
    y_test  = Float32.(y_test)
    TL_coef = Float32.(TL_coef)
    l_segs_test = [length(y_test)]

    Nf = size(x,2) # number of features

    if sum(data_norms[end]) == 0 # normalize data
        (A_bias,A_scale,A_norm) = norm_sets(A;norm_type=norm_type_A)
        (x_bias,x_scale,x_norm) = norm_sets(x;norm_type=norm_type_x,no_norm=no_norm)
        (y_bias,y_scale,y_norm) = norm_sets(y;norm_type=norm_type_y)
        if k_pca > 0
            if k_pca > Nf
                silent || @info("reducing k_pca from $k_pca to $Nf")
                k_pca = Nf
            end
            (_,S,V) = svd(cov(x_norm))
            v_scale = V[:,1:k_pca]*inv(Diagonal(sqrt.(S[1:k_pca])))
            var_ret = round(sum(sqrt.(S[1:k_pca]))/sum(sqrt.(S))*100,digits=6)
            silent || @info("k_pca = $k_pca of $Nf, variance retained: $var_ret %")
        else
            v_scale = I(Nf)
        end
        A_norm = (A_norm          )'
        x_norm = (x_norm * v_scale)'
        y_norm = (y_norm          )'
    else # unpack data normalizations
        (A_bias,A_scale,v_scale,x_bias,x_scale,y_bias,y_scale) =
            unpack_data_norms(data_norms)
        A_norm = ( (A .- A_bias) ./ A_scale           )'
        x_norm = (((x .- x_bias) ./ x_scale) * v_scale)'
        y_norm = ( (y .- y_bias) ./ y_scale           )'
    end

    TL_coef_norm = TL_coef ./ y_scale

    # normalize test data if it was provided
    isempty(A_test) || (A_test_norm = ( (A_test .- A_bias) ./ A_scale           )')
    isempty(x_test) || (x_test_norm = (((x_test .- x_bias) ./ x_scale) * v_scale)')

    # separate into training and validation
    if frac_train < 1
        N = size(x_norm,2)
        (p_train,p_val) = get_split(N,frac_train,:none)
        A_train_norm    = A_norm[:,p_train]
        A_val_norm      = A_norm[:,p_val  ]
        x_train_norm    = x_norm[:,p_train]
        x_val_norm      = x_norm[:,p_val  ]
        y_train_norm    = y_norm[:,p_train]
        y_val_norm      = y_norm[:,p_val  ]
        data_train      = DataLoader((A_train_norm,x_train_norm,y_train_norm),
                                     shuffle=true,batchsize=batchsize)
        data_val        = DataLoader((A_val_norm,x_val_norm,y_val_norm),
                                     shuffle=true,batchsize=batchsize)
    else
        data_train      = DataLoader((A_norm,x_norm,y_norm),
                                     shuffle=true,batchsize=batchsize)
        data_val        = data_train
    end

    # setup NN
    if model == Chain() # initialize model
        Ny = model_type in [:m2a,:m2d] ? size(A_norm,1) : 1 # length of output
        m  = get_nn_m(Nf,Ny;hidden=hidden,activation=activation)
    else # re-train on known model
        m  = deepcopy(model)
    end

    # setup optimizer
    opt = Adam(η_adam)

    # setup loss function
    function loss_m2(A_norm, x_norm, y_norm, model_type::Symbol, TL_coef_norm)
        y_hat_norm = nn_comp_2_fwd(A_norm,x_norm,y_bias,y_scale,m;
                                   model_type   = model_type,
                                   TL_coef_norm = TL_coef_norm,
                                   denorm       = false,
                                   testmode     = false)
        return mse(y_hat_norm, vec(y_norm))
    end # function loss_m2

    loss_m2a(Al,xl,yl) = loss_m2(Al,xl,yl,:m2a,TL_coef_norm)
    loss_m2b(Al,xl,yl) = loss_m2(Al,xl,yl,:m2b,TL_coef_norm)
    loss_m2c(Al,xl,yl) = loss_m2(Al,xl,yl,:m2c,TL_coef_norm)
    loss_m2d(Al,xl,yl) = loss_m2(Al,xl,yl,:m2d,TL_coef_norm)

    loss_m2a_λ(Al,xl,yl) = loss_m2a(Al,xl,yl) + λ*sum(sparse_group_lasso(m,α))
    loss_m2b_λ(Al,xl,yl) = loss_m2b(Al,xl,yl) + λ*sum(sparse_group_lasso(m,α))
    loss_m2c_λ(Al,xl,yl) = loss_m2c(Al,xl,yl) + λ*sum(sparse_group_lasso(m,α))
    loss_m2d_λ(Al,xl,yl) = loss_m2d(Al,xl,yl) + λ*sum(sparse_group_lasso(m,α))

    model_type == :m2a && (loss = λ > 0 ? loss_m2a_λ : loss_m2a)
    model_type == :m2b && (loss = λ > 0 ? loss_m2b_λ : loss_m2b)
    model_type == :m2c && (loss = λ > 0 ? loss_m2c_λ : loss_m2c)
    model_type == :m2d && (loss = λ > 0 ? loss_m2d_λ : loss_m2d)

    function loss_all(data_l)
        l = 0f0
        for (A_l,x_l,y_l) in data_l
            l += loss(A_l,x_l,y_l)
        end
        return (l/length(data_l))
    end # function loss_all

    # train NN with Adam optimizer
    TL_coef_store = deepcopy(TL_coef_norm)
    m_store       = deepcopy(m)
    best_loss     = loss_all(data_val)
    isempty(x_test) || (best_test_error = std(nn_comp_2_test(A_test_norm,x_test_norm,y_test,y_bias,y_scale,m;
                                                             model_type   = model_type,
                                                             TL_coef_norm = TL_coef_norm,
                                                             l_segs       = l_segs_test,
                                                             silent       = silent_debug)[2]))

    silent || @info("epoch 0: loss = $best_loss")
    for i = 1:epoch_adam
        if model_type in [:m2a,:m2b,:m2d] # train on NN model weights only
            Flux.train!(loss,Flux.params(m),data_train,opt)
        elseif model_type in [:m2c] # train on NN model weights + TL coef
            Flux.train!(loss,Flux.params(m,TL_coef_norm),data_train,opt)
        end
        current_loss = loss_all(data_val)
        if isempty(x_test)
            if current_loss < best_loss
                best_loss     = current_loss
                m_store       = deepcopy(m)
                TL_coef_store = deepcopy(TL_coef_norm)
            end
            mod(i,5) == 0 && !silent && @info("epoch $i: loss = $best_loss")
        else
            test_error = std(nn_comp_2_test(A_test_norm,x_test_norm,y_test,y_bias,y_scale,m;
                                            model_type   = model_type,
                                            TL_coef_norm = TL_coef_norm,
                                            l_segs       = l_segs_test,
                                            silent       = silent_debug)[2])
            if test_error < best_test_error
                best_test_error = test_error
                m_store         = deepcopy(m)
                TL_coef_store   = deepcopy(TL_coef_norm)
            end
            mod(i,5) == 0 && !silent && @info("epoch $i: loss = $current_loss, test error = $(round(best_test_error,digits=2)) nT")
        end
        if (mod(i,10) == 0) & !silent
            err = nn_comp_2_test(A_norm,x_norm,y,y_bias,y_scale,m;
                                 model_type   = model_type,
                                 TL_coef_norm = TL_coef_norm,
                                 l_segs       = l_segs,
                                 silent       = true)[2]
            @info("$i train error: $(round(std(err),digits=2)) nT")
            isempty(x_test) || @info("$i test  error: $(round((test_error),digits=2)) nT")
        end
    end

    Flux.loadmodel!(m,m_store)
    TL_coef_norm = TL_coef_store

    if epoch_lbfgs > 0 # LBFGS, may overfit depending on iterations
        model_type == :m2c && !silent && @info("LBFGS will only update NN model weights (not TL coef)")
        data = DataLoader((A_norm,x_norm,y_norm),shuffle=true,batchsize=batchsize)

        function lbfgs_train!(m_l,data_l,iter,t_l,TL_coef_l,silent)
            (A_l,x_l,y_l) = data_l.data
            loss_() = loss_m2(A_l,x_l,y_l,t_l,TL_coef_l)
            refresh()
            if t_l in [:m2a,:m2b,:m2d] # train on NN model weights only
                params = Flux.params(m_l)
            elseif t_l in [:m2c] # train on NN model weights + TL coef
                params = Flux.params(m_l,TL_coef_l)
            end
            opt = LBFGS()
            (_,_,fg!,p0) = optfuns(loss_,params)
            options = Options(iterations=iter,show_every=5,show_trace=!silent)
            optimize(only_fg!(fg!),p0,opt,options)
        end # function lbfgs_train!

        # train NN with LBFGS optimizer
        lbfgs_train!(m,data,epoch_lbfgs,model_type,TL_coef_norm,silent)
    end

    # get results
    (y_hat,err) = nn_comp_2_test(A_norm,x_norm,y,y_bias,y_scale,m;
                                 model_type   = model_type,
                                 TL_coef_norm = TL_coef_norm,
                                 l_segs       = l_segs,
                                 silent       = true)
    silent || @info("train error: $(round(std(err),digits=2)) nT")

    if !isempty(x_test)
        nn_comp_2_test(A_test_norm,x_test_norm,y_test,y_bias,y_scale,m;
                       model_type   = model_type,
                       TL_coef_norm = TL_coef_norm,
                       l_segs       = l_segs_test,
                       silent       = silent)
    end

    model   = m
    TL_coef = TL_coef_norm .* y_scale

    # pack data normalizations
    data_norms = (A_bias,A_scale,v_scale,x_bias,x_scale,y_bias,y_scale)

    return (model, TL_coef, data_norms, y_hat, err)
end # function nn_comp_2_train

"""
    nn_comp_2_fwd(A_norm::AbstractMatrix, x_norm::AbstractMatrix, y_bias, y_scale, model::Chain;
                  model_type::Symbol   = :m2a,
                  TL_coef_norm::Vector = zeros(Float32,18),
                  denorm::Bool         = true,
                  testmode::Bool       = true)

Forward pass of neural network-based aeromagnetic compensation, model 2.
"""
function nn_comp_2_fwd(A_norm::AbstractMatrix, x_norm::AbstractMatrix, y_bias, y_scale, model::Chain;
                       model_type::Symbol   = :m2a,
                       TL_coef_norm::Vector = zeros(Float32,18),
                       denorm::Bool         = true,
                       testmode::Bool       = true)

    # set to test mode in case model uses batchnorm or dropout
    m = model
    testmode && Flux.testmode!(m)

    # get results
    if model_type in [:m2a]
        y_hat = vec(sum(A_norm.*m(x_norm), dims=1))
    elseif model_type in [:m2b,:m2c]
        y_hat = vec(m(x_norm)) + A_norm'*TL_coef_norm
    elseif model_type in [:m2d]
        y_hat = vec(sum(A_norm.*(m(x_norm) .+ TL_coef_norm), dims=1))
    end

    denorm && (y_hat .= denorm_sets(y_bias,y_scale,y_hat))

    return (y_hat)
end # function nn_comp_2_fwd

"""
    nn_comp_2_fwd(A::Matrix, x::Matrix, data_norms::Tuple, model::Chain;
                  model_type::Symbol = :m2a,
                  TL_coef::Vector    = zeros(Float32,18))

Forward pass of neural network-based aeromagnetic compensation, model 2.
"""
function nn_comp_2_fwd(A::Matrix, x::Matrix, data_norms::Tuple, model::Chain;
                       model_type::Symbol = :m2a,
                       TL_coef::Vector    = zeros(Float32,18))

    # convert to Float32 for consistency with nn_comp_2_train
    A       = Float32.(A)
    x       = Float32.(x)
    TL_coef = Float32.(TL_coef)

    # unpack data normalizations
    (A_bias,A_scale,v_scale,x_bias,x_scale,y_bias,y_scale) =
        unpack_data_norms(data_norms)
    A_norm = ( (A .- A_bias) ./ A_scale           )'
    x_norm = (((x .- x_bias) ./ x_scale) * v_scale)'

    TL_coef_norm = TL_coef ./ y_scale

    # get results
    y_hat = nn_comp_2_fwd(A_norm,x_norm,y_bias,y_scale,model;
                          model_type   = model_type,
                          TL_coef_norm = TL_coef_norm)

    return (y_hat)
end # function nn_comp_2_fwd

"""
    nn_comp_2_test(A_norm::AbstractMatrix, x_norm::AbstractMatrix, y, y_bias, y_scale, model::Chain;
                   model_type::Symbol   = :m2a,
                   TL_coef_norm::Vector = zeros(Float32,18),
                   l_segs::Vector       = [length(y)],
                   silent::Bool         = false)

Evaluate performance of neural network-based aeromagnetic compensation, model 2.
"""
function nn_comp_2_test(A_norm::AbstractMatrix, x_norm::AbstractMatrix, y, y_bias, y_scale, model::Chain;
                        model_type::Symbol   = :m2a,
                        TL_coef_norm::Vector = zeros(Float32,18),
                        l_segs::Vector       = [length(y)],
                        silent::Bool         = false)

    # get results
    y_hat = nn_comp_2_fwd(A_norm,x_norm,y_bias,y_scale,model;
                          model_type   = model_type,
                          TL_coef_norm = TL_coef_norm)
    err   = err_segs(y_hat,y,l_segs;silent=silent_debug)
    silent || @info("test  error: $(round(std(err),digits=2)) nT")

    return (y_hat, err)
end # function nn_comp_2_test

"""
    nn_comp_2_test(A::Matrix, x::Matrix, y, data_norms::Tuple, model::Chain;
                   model_type::Symbol = :m2a,
                   TL_coef::Vector    = zeros(Float32,18),
                   l_segs::Vector     = [length(y)],
                   silent::Bool       = false)

Evaluate performance of neural network-based aeromagnetic compensation, model 2.
"""
function nn_comp_2_test(A::Matrix, x::Matrix, y, data_norms::Tuple, model::Chain;
                        model_type::Symbol = :m2a,
                        TL_coef::Vector    = zeros(Float32,18),
                        l_segs::Vector     = [length(y)],
                        silent::Bool       = false)

    # convert to Float32 for consistency with nn_comp_2_train
    y = Float32.(y)

    # get results
    y_hat = nn_comp_2_fwd(A,x,data_norms,model;
                          model_type = model_type,
                          TL_coef    = TL_coef)
    err   = err_segs(y_hat,y,l_segs;silent=silent_debug)
    silent || @info("test  error: $(round(std(err),digits=2)) nT")

    return (y_hat, err)
end # function nn_comp_2_test

"""
    get_curriculum_ind(TL_diff::Vector, N_sigma::Real=1)

Internal helper function to get indices for curriculum learning (to train
the Tolles-Lawson part of the loss) and indices to train the neural network.
Curriculum learning (Tolles-Lawson) indices are those within `N_sigma` of the
mean and neural network indices are those outside of it (i.e., outliers).

**Arguments:**
- `TL_diff`: difference of TL model to ground truth
- `N_sigma`: (optional) standard deviation threshold

**Returns:**
- `ind_cur`: indices for curriculum learning (within `N_sigma`)
- `ind_nn`:  indices for training the neural network (outside `N_sigma`)
"""
function get_curriculum_ind(TL_diff::Vector, N_sigma::Real=1)
    TL_diff = detrend(TL_diff;mean_only=true)
    cutoff  = N_sigma*std(TL_diff)
    ind_cur = -cutoff .<= TL_diff .<= cutoff
    ind_nn  = .!ind_cur
    return (ind_cur, ind_nn)
end # function get_curriculum_ind

"""
    TL_vec2mat(TL_coef::Vector, terms; Bt_scale=50000f0)

Internal helper function to extract the matrix form of Tolles-Lawson
coefficients from the vector form.

**Arguments:**
- `TL_coef`:  Tolles-Lawson coefficients (must include `:permanent` & `:induced`)
- `terms`:    Tolles-Lawson terms used {`:permanent`,`:induced`,`:eddy`}
- `Bt_scale`: (optional) scaling factor for induced and eddy current terms [nT]

**Returns:**
- `TL_coef_p`: length `3` vector of permanent field coefficients
- `TL_coef_i`: `3` x `3`  symmetric matrix of induced field coefficients, denormalized
- `TL_coef_e`: `3` x `3`  matrix of eddy current coefficients, denormalized
"""
function TL_vec2mat(TL_coef::Vector, terms; Bt_scale=50000f0)
    @assert any([:permanent,:p,:permanent3,:p3] .∈ (terms,)) "permanent terms are required"
    @assert any([:induced,:i,:induced6,:i6,:induced5,:i5,:induced3,:i3] .∈ (terms,)) "induced terms are required"
    @assert !any([:fdm,:f,:fdm3,:f3,:bias,:b] .∈ (terms,)) "derivative and bias terms may not be used"

    N = length(TL_coef)
    A_test = create_TL_A([1.0],[1.0],[1.0];terms=terms)
    @assert N == length(A_test) "TL_coef does not agree with specified terms"

    TL_coef_p = TL_coef[1:3]

    if any([:induced,:i,:induced6,:i6] .∈ (terms,))
        TL_coef_i = Symmetric([TL_coef[4] TL_coef[5]/2 TL_coef[6]/2
                               0f0        TL_coef[7]   TL_coef[8]/2
                               0f0        0f0          TL_coef[9]  ] / Bt_scale, :U)
    elseif any([:induced5,:i5] .∈ (terms,))
        TL_coef_i = Symmetric([TL_coef[4] TL_coef[5]/2 TL_coef[6]/2
                               0f0        TL_coef[7]   TL_coef[8]/2
                               0f0        0f0          0f0         ] / Bt_scale, :U)
    elseif any([:induced3,:i3] .∈ (terms,))
        TL_coef_i = Symmetric([TL_coef[4] 0f0          0f0
                               0f0        TL_coef[5]   0f0
                               0f0        0f0          TL_coef[6]  ] / Bt_scale, :U)
    end

    if any([:eddy,:e,:eddy9,:e9] .∈ (terms,))
        TL_coef_e = [TL_coef[N-8] TL_coef[N-7] TL_coef[N-6]
                     TL_coef[N-5] TL_coef[N-4] TL_coef[N-3]
                     TL_coef[N-2] TL_coef[N-1] TL_coef[N-0]] / Bt_scale
    elseif any([:eddy8,:e8] .∈ (terms,))
        TL_coef_e = [TL_coef[N-7] TL_coef[N-6] TL_coef[N-5]
                     TL_coef[N-4] TL_coef[N-3] TL_coef[N-2]
                     TL_coef[N-1] TL_coef[N-0] 0f0         ] / Bt_scale
    elseif any([:eddy3,:e3] .∈ (terms,))
        TL_coef_e = [TL_coef[N-2] 0f0          0f0
                     0f0          TL_coef[N-1] 0f0
                     0f0          0f0          TL_coef[N-0]] / Bt_scale
    else
        TL_coef_e = []
    end

    return (TL_coef_p, TL_coef_i, TL_coef_e)
end # function TL_vec2mat

"""
    TL_mat2vec(TL_coef_p, TL_coef_i, TL_coef_e, terms; Bt_scale=50000f0)

Internal helper function to extract the vector form of Tolles-Lawson
coefficients from the matrix form.

**Arguments:**
- `TL_coef_p`: length `3` vector of permanent field coefficients
- `TL_coef_i`: `3` x `3`  symmetric matrix of induced field coefficients, denormalized
- `TL_coef_e`: `3` x `3`  matrix of eddy current coefficients, denormalized
- `terms`:     Tolles-Lawson terms used {`:permanent`,`:induced`,`:eddy`}
- `Bt_scale`:  (optional) scaling factor for induced and eddy current terms [nT]

**Returns:**
- `TL_coef`: Tolles-Lawson coefficients
"""
function TL_mat2vec(TL_coef_p, TL_coef_i, TL_coef_e, terms; Bt_scale=50000f0)

    if any([:induced,:i,:induced6,:i6] .∈ (terms,))
        TL_coef_i = [TL_coef_i[1,1:3]; TL_coef_i[2,2:3]; TL_coef_i[3,3]] * Bt_scale
        TL_coef_i[[2,3,5]] .*= 2
    elseif any([:induced5,:i5] .∈ (terms,))
        TL_coef_i = [TL_coef_i[1,1:3]; TL_coef_i[2,2:3]                ] * Bt_scale
        TL_coef_i[[2,3,5]] .*= 2
    elseif any([:induced3,:i3] .∈ (terms,))
        TL_coef_i = [TL_coef_i[1,1]  ; TL_coef_i[2,2]  ; TL_coef_i[3,3]] * Bt_scale
    end

    if any([:eddy,:e,:eddy9,:e9] .∈ (terms,))
        TL_coef_e = vec(TL_coef_e')          * Bt_scale
    elseif any([:eddy8,:e8] .∈ (terms,))
        TL_coef_e = vec(TL_coef_e')[1:8]     * Bt_scale
    elseif any([:eddy3,:e3] .∈ (terms,))
        TL_coef_e = vec(TL_coef_e')[[1,5,9]] * Bt_scale
    else
        TL_coef_e = []
    end

    TL_coef = [TL_coef_p; TL_coef_i; TL_coef_e]

    return (TL_coef)
end # function TL_mat2vec

"""
    get_TL_aircraft_vec(B_vec, B_vec_dot, TL_coef_p, TL_coef_i, TL_coef_e;
                        return_parts::Bool=false)

**Arguments:**
- `B_vec`:        `3` x `N`  matrix of vector magnetometer measurements
- `B_vec_dot`:    `3` x `N`  matrix of vector magnetometer measurement derivatives
- `TL_coef_p`:    length `3` vector of permanent field coefficients
- `TL_coef_i`:    `3` x `3`  symmetric matrix of induced field coefficients, denormalized
- `TL_coef_e`:    `3` x `3`  matrix of eddy current coefficients, denormalized
- `return_parts`: (optional) if true, also return `TL_perm`, `TL_induced`, & `TL_eddy`

**Returns:**
- `TL_aircraft`: `3` x `N` matrix of TL aircraft vector field
- `TL_perm`:     if `return_parts = true`, `3` x `N` matrix of TL permanent vector field
- `TL_induced`:  if `return_parts = true`, `3` x `N` matrix of TL induced vector field
- `TL_eddy`:     if `return_parts = true`, `3` x `N` matrix of TL eddy current vector field
"""
function get_TL_aircraft_vec(B_vec, B_vec_dot, TL_coef_p, TL_coef_i, TL_coef_e;
                             return_parts::Bool=false)

    TL_perm    = TL_coef_p .* one.(B_vec)
    TL_induced = TL_coef_i * B_vec

    if length(TL_coef_e) > 0
        TL_eddy     = TL_coef_e * B_vec_dot
        TL_aircraft = TL_perm + TL_induced + TL_eddy
    else
        TL_eddy     = []
        TL_aircraft = TL_perm + TL_induced
    end

    if return_parts
        return (TL_aircraft, TL_perm, TL_induced, TL_eddy)
    else
        return (TL_aircraft)
    end
end # function get_TL_aircraft_vec

"""
    get_temporal_data(x_norm::AbstractMatrix, l_window::Int)

Internal helper function to create windowed sequence temporal data from
original data. Adds padding to the beginning of the data.

**Arguments:**
- `x_norm`:   `Nf` x `N` normalized data matrix (`Nf` is number of features)
- `l_window`: temporal window length

**Returns:**
- `x_w`: `Nf` x `l_window` x `N` normalized data matrix, windowed
"""
function get_temporal_data(x_norm::AbstractMatrix, l_window::Int)
    (Nf,N) = size(x_norm)
    x_w    = zeros(Float32, Nf, l_window, N)
    for i = 1:N
        i1 = max(1, i - l_window + 1)
        x_w[:, i1+l_window-i:l_window, i] = x_norm[:, i1:i]
    end
    return (x_w)
end # function get_temporal_data

"""
    get_split(N::Int, frac_train::Real, window_type::Symbol = :none;
              l_window::Int = 0)

Internal helper function to get training & validation data indices.

**Arguments:**
- `N`:           number of samples (instances)
- `frac_train`:  fraction of training data used for training (remainder for validation)
- `window_type`: (optional) type of windowing, `:none` for none, :sliding` for overlapping, or `:contiguous` for non-overlapping
- `l_window`:    (optional) temporal window length

**Returns:**
- `p_train`: training   indices
- `p_val`:   validation indices
"""
function get_split(N::Int, frac_train::Real, window_type::Symbol = :none;
                   l_window::Int = 0)

    @assert 0 <= frac_train <= 1 "frac_train must be between 0 and 1"
    @assert l_window < N "window length of $l_window is too large for $N samples"

    if window_type == :none # no windows
        if frac_train < 1
            p       = randperm(N)
            N_train = floor(Int,N*frac_train)
            p_train = p[1:N_train]
            p_val   = p[N_train+1:end]
        else
            p       = 1:N
            p_train = p
            p_val   = p
        end
    elseif window_type == :sliding # sliding windows
        p = 1:N
        if frac_train < 1
            N_train = floor(Int,N*frac_train)
            @assert all(l_window .<= (N_train,N-N_train)) "window length of $l_window is too large for $N samples with frac_train = $frac_train"
            p_train = p[1:N_train]
            p_val   = p[N_train+l_window:end]
        else
            p_train = p
            p_val   = p
        end
    elseif window_type == :contiguous # contiguous windows
        ind = 1:l_window:N
        N_  = length(ind)
        p_  = randperm(N_)
        if frac_train < 1
            N_train = floor(Int,frac_train*N_)
            @assert all(1 .<= (N_train,N_-N_train)) "window length of $l_window is too large for $N samples with frac_train = $frac_train"
            p_train = ind[p_[1:N_train]]
            p_val   = ind[p_[N_train+1:end]]
        else
            p_train = ind[p_]
            p_val   = ind[p_]
        end
    else
        error("$window_type window_type is invalid, choose :none, :sliding, or :contiguous")
    end

    return (p_train, p_val)
end # function get_split

"""
    nn_comp_3_train(A, Bt, B_dot, x, y, no_norm;
                    model_type::Symbol    = :m3s,
                    norm_type_x::Symbol   = :standardize,
                    norm_type_y::Symbol   = :standardize,
                    TL_coef::Vector       = zeros(Float32,18),
                    terms_A               = [:permanent,:induced,:eddy],
                    y_type::Symbol        = :d,
                    η_adam                = 0.001,
                    epoch_adam::Int       = 5,
                    epoch_lbfgs::Int      = 0,
                    hidden                = [8],
                    activation::Function  = swish,
                    batchsize::Int        = 2048,
                    frac_train            = 14/17,
                    α_sgl                 = 1,
                    λ_sgl                 = 0,
                    k_pca::Int            = -1,
                    σ_curriculum          = 1.0,
                    l_window::Int         = 5,
                    window_type::Symbol   = :sliding,
                    tf_layer_type::Symbol = :postlayer,
                    tf_norm_type::Symbol  = :batch,
                    dropout_prob          = 0.2,
                    N_tf_head::Int        = 8,
                    tf_gain               = 1.0,
                    data_norms::Tuple     = (zeros(Float32,1,1),zeros(Float32,1,1),zeros(Float32,1,1),zeros(Float32,1,1),zeros(Float32,1,1),[0f0],[0f0]),
                    model::Chain          = Chain(),
                    l_segs::Vector        = [length(y)],
                    A_test::Matrix        = Matrix{eltype(A)}(undef,0,0),
                    Bt_test::Vector       = Vector{eltype(Bt)}(undef,0),
                    B_dot_test::Matrix    = Matrix{eltype(B_dot)}(undef,0,0),
                    x_test::Matrix        = Matrix{eltype(x)}(undef,0,0),
                    y_test::Vector        = Vector{eltype(y)}(undef,0),
                    silent::Bool          = false)

Train neural network-based aeromagnetic compensation, model 3. Model 3
architectures retain the Tolles-Lawson (TL) terms in vector form, making it
possible to remove the Taylor expansion approximation used for predicting
the Earth field in the loss function, as well as creating vector-based neural
network corrections.

Note that this model is subject to change, and not all options are supported
(e.g., sparse group Lasso, and generally the IGRF and diurnal fields should not
be subtracted out from the `y` target vector)

Currently, it is recommended to use a low-pass (not a bandpass) filter to
initialize the Tolles-Lawson coefficients, as a bandpass filter removes the
Earth field and leaves a large bias in the aircraft field prediction, e.g.,
`TL_coef = create_TL_coef(getfield(xyz,use_vec), getfield(xyz,use_mag) - xyz.mag_1_c, TL_ind;
                          terms=terms, pass1=0.0, pass2=0.9)`

- `model_type`:
    - `:m3tl` = no NN, TL coefficients fine-tuned via SGD, without Taylor expansion for `y_type` :b and :c (for testing)
    - `:m3s`  = NN determines scalar correction to TL, using expanded TL vector terms for explainability
    - `:m3v`  = NN determines vector correction to TL, using expanded TL vector terms for explainability
    - `:m3sc` = `:m3s` with curriculum learning based on TL error
    - `:m3vc` = `:m3v` with curriculum learning based on TL error
    - `:m3w`  = `:m3s` with window NN for temporal dataset
    - `:m3tf` = `:m3s` with time series transformer for temporal dataset
"""
function nn_comp_3_train(A, Bt, B_dot, x, y, no_norm;
                         model_type::Symbol    = :m3s,
                         norm_type_x::Symbol   = :standardize,
                         norm_type_y::Symbol   = :standardize,
                         TL_coef::Vector       = zeros(Float32,18),
                         terms_A               = [:permanent,:induced,:eddy],
                         y_type::Symbol        = :d,
                         η_adam                = 0.001,
                         epoch_adam::Int       = 5,
                         epoch_lbfgs::Int      = 0,
                         hidden                = [8],
                         activation::Function  = swish,
                         batchsize::Int        = 2048,
                         frac_train            = 14/17,
                         α_sgl                 = 1,
                         λ_sgl                 = 0,
                         k_pca::Int            = -1,
                         σ_curriculum          = 1.0,
                         l_window::Int         = 5,
                         window_type::Symbol   = :sliding,
                         tf_layer_type::Symbol = :postlayer,
                         tf_norm_type::Symbol  = :batch,
                         dropout_prob          = 0.2,
                         N_tf_head::Int        = 8,
                         tf_gain               = 1.0,
                         data_norms::Tuple     = (zeros(Float32,1,1),zeros(Float32,1,1),zeros(Float32,1,1),zeros(Float32,1,1),zeros(Float32,1,1),[0f0],[0f0]),
                         model::Chain          = Chain(),
                         l_segs::Vector        = [length(y)],
                         A_test::Matrix        = Matrix{eltype(A)}(undef,0,0),
                         Bt_test::Vector       = Vector{eltype(Bt)}(undef,0),
                         B_dot_test::Matrix    = Matrix{eltype(B_dot)}(undef,0,0),
                         x_test::Matrix        = Matrix{eltype(x)}(undef,0,0),
                         y_test::Vector        = Vector{eltype(y)}(undef,0),
                         silent::Bool          = false)

    @assert (α_sgl,λ_sgl) == (1,0)  "sparse group Lasso not implemented in nn_comp_3"
    @assert y_type in [:a,:b,:c,:d] "unsupported y_type = $y_type for nn_comp_3"

    # convert to Float32 for ~50% speedup
    A          = Float32.(A)
    Bt         = Float32.(Bt)    # magnitude of total field measurements
    B_dot      = Float32.(B_dot) # finite differences of total field vector
    x          = Float32.(x)
    y          = Float32.(y)
    A_test     = Float32.(A_test)
    Bt_test    = Float32.(Bt_test)
    B_dot_test = Float32.(B_dot_test)
    x_test     = Float32.(x_test)
    y_test     = Float32.(y_test)
    TL_coef    = Float32.(TL_coef)
    l_segs_test = [length(y_test)]

    Nf = size(x,2) # number of features

    # assume all terms are stored, but they may be zero if not trained
    Bt_scale = 50000f0
    (TL_coef_p,TL_coef_i,TL_coef_e) = TL_vec2mat(TL_coef,terms_A;Bt_scale=Bt_scale)

    B_unit    = A[:,1:3]'     # normalized vector magnetometer reading
    B_vec     = B_unit .* Bt' # vector magnetometer to be used in TL
    B_vec_dot = B_dot'        # not exactly true, but internally consistent
    isempty(A_test)     || (B_unit_test    = A_test[:,1:3]')
    isempty(Bt_test)    || (B_vec_test     = B_unit_test .* Bt_test')
    isempty(B_dot_test) || (B_vec_dot_test = B_dot_test')

    if sum(data_norms[end]) == 0 # normalize data
        (x_bias,x_scale,x_norm) = norm_sets(x;norm_type=norm_type_x,no_norm=no_norm)
        (y_bias,y_scale,y_norm) = norm_sets(y;norm_type=norm_type_y)
        if k_pca > 0
            if k_pca > Nf
                silent || @info("reducing k_pca from $k_pca to $Nf")
                k_pca = Nf
            end
            (_,S,V) = svd(cov(x_norm))
            v_scale = V[:,1:k_pca]*inv(Diagonal(sqrt.(S[1:k_pca])))
            var_ret = round(sum(sqrt.(S[1:k_pca]))/sum(sqrt.(S))*100,digits=6)
            silent || @info("k_pca = $k_pca of $Nf, variance retained: $var_ret %")
        else
            v_scale = I(Nf)
        end
        x_norm = (x_norm * v_scale)'
        y_norm = (y_norm          )'
    else # unpack data normalizations
        (_,_,v_scale,x_bias,x_scale,y_bias,y_scale) = unpack_data_norms(data_norms)
        x_norm = (((x .- x_bias) ./ x_scale) * v_scale)'
        y_norm = ( (y .- y_bias) ./ y_scale           )'
    end

    # normalize test data if it was provided
    isempty(x_test) || (x_test_norm = (((x_test .- x_bias) ./ x_scale) * v_scale)')

    # get temporal data for temporal models
    if model_type in [:m3w,:m3tf]
        x_norm = get_temporal_data(x_norm,l_window)
        isempty(x_test) || (x_test_norm = get_temporal_data(x_test_norm,l_window))
    end

    # separate into training and validation
    if frac_train < 1
        if model_type in [:m3w,:m3tf]
            N = size(x_norm,3)
            (p_train,p_val) = get_split(N,frac_train,window_type;l_window=l_window)
            x_train_norm    = x_norm[:,:,p_train]
            x_val_norm      = x_norm[:,:,p_val  ]
        else
            N = size(x_norm,2)
            (p_train,p_val) = get_split(N,frac_train,:none)
            x_train_norm    = x_norm[:,p_train]
            x_val_norm      = x_norm[:,p_val  ]
        end

        B_unit_train    = B_unit[   :,p_train]
        B_unit_val      = B_unit[   :,p_val  ]
        B_vec_train     = B_vec[    :,p_train]
        B_vec_val       = B_vec[    :,p_val  ]
        B_vec_dot_train = B_vec_dot[:,p_train]
        B_vec_dot_val   = B_vec_dot[:,p_val  ]
        y_train_norm    = y_norm[   :,p_train]
        y_val_norm      = y_norm[   :,p_val  ]

        if model_type in [:m3sc,:m3vc]
            silent || @info("making curriculum")
            # calculate TL estimate in training data
            y_TL = nn_comp_3_fwd(B_unit_train,B_vec_train,B_vec_dot_train,
                                 x_train_norm,y_bias,y_scale,model,
                                 TL_coef_p,TL_coef_i,TL_coef_e;
                                 model_type = :m3tl,
                                 y_type     = y_type,
                                 use_nn     = false,
                                 denorm     = true,
                                 testmode   = true)
            TL_diff = vec(y[p_train,:]) - y_TL
            (ind_cur,_)  = get_curriculum_ind(TL_diff,σ_curriculum)
            data_train   = DataLoader((B_unit_train[   :,ind_cur],
                                       B_vec_train[    :,ind_cur],
                                       B_vec_dot_train[:,ind_cur],
                                       x_train_norm[   :,ind_cur],
                                       y_train_norm[   :,ind_cur]),
                                      shuffle=true,batchsize=batchsize)
            data_train_2 = DataLoader((B_unit_train,B_vec_train,B_vec_dot_train,
                                       x_train_norm,y_train_norm),
                                      shuffle=true,batchsize=batchsize)
        else
            data_train   = DataLoader((B_unit_train,B_vec_train,B_vec_dot_train,
                                       x_train_norm,y_train_norm),
                                      shuffle=true,batchsize=batchsize)
        end
        data_val     = DataLoader((B_unit_val,B_vec_val,B_vec_dot_val,
                                   x_val_norm,y_val_norm),
                                  shuffle=true,batchsize=batchsize)
    else
        data_train   = DataLoader((B_unit,B_vec,B_vec_dot,
                                   x_norm,y_norm),
                                  shuffle=true,batchsize=batchsize)
        data_train_2 = data_train
        data_val     = data_train
    end

    # setup NN
    if model == Chain() # initialize model
        Ny = model_type in [:m3v,:m3vc] ? 3 : 1 # length of output
        m  = get_nn_m(Nf,Ny;hidden=hidden,activation=activation,
                      model_type    = model_type,
                      l_window      = l_window,
                      tf_layer_type = tf_layer_type,
                      tf_norm_type  = tf_norm_type,
                      dropout_prob  = dropout_prob,
                      N_tf_head     = N_tf_head,
                      tf_gain       = tf_gain)
    else # re-train on known model
        m  = deepcopy(model)
    end

    # setup optimizer
    opt = Adam(η_adam)

    # setup loss function
    function loss_m3(B_unit, B_vec, B_vec_dot, x_norm, y_norm, model_type::Symbol,
                     TL_coef_p, TL_coef_i, TL_coef_e, use_nn::Bool)
        y_hat_norm = nn_comp_3_fwd(B_unit,B_vec,B_vec_dot,
                                   x_norm,y_bias,y_scale,m,
                                   TL_coef_p,TL_coef_i,TL_coef_e;
                                   model_type = model_type,
                                   y_type     = y_type,
                                   use_nn     = use_nn,
                                   denorm     = false,
                                   testmode   = false)
        return mse(y_hat_norm, vec(y_norm))
    end # function loss_m3

    use_nn = model_type in [:m3s,:m3v,:m3w,:m3tf] ? true : false # multiplier to include/exclude NN contribution to y_hat
    loss_m3tl(Bul,Bvl,Bvdl,xl,yl) = loss_m3(Bul,Bvl,Bvdl,xl,yl,:m3tl,TL_coef_p,TL_coef_i,TL_coef_e,use_nn) # :m3tl does not use NN
    loss_m3s( Bul,Bvl,Bvdl,xl,yl) = loss_m3(Bul,Bvl,Bvdl,xl,yl,:m3s ,TL_coef_p,TL_coef_i,TL_coef_e,use_nn) # use NN
    loss_m3v( Bul,Bvl,Bvdl,xl,yl) = loss_m3(Bul,Bvl,Bvdl,xl,yl,:m3v ,TL_coef_p,TL_coef_i,TL_coef_e,use_nn) # use NN
    loss_m3sc(Bul,Bvl,Bvdl,xl,yl) = loss_m3(Bul,Bvl,Bvdl,xl,yl,:m3tl,TL_coef_p,TL_coef_i,TL_coef_e,use_nn) # :m3sc starts without NN
    loss_m3vc(Bul,Bvl,Bvdl,xl,yl) = loss_m3(Bul,Bvl,Bvdl,xl,yl,:m3v ,TL_coef_p,TL_coef_i,TL_coef_e,use_nn) # :m3vc starts without NN
    loss_m3w( Bul,Bvl,Bvdl,xl,yl) = loss_m3(Bul,Bvl,Bvdl,xl,yl,:m3w ,TL_coef_p,TL_coef_i,TL_coef_e,use_nn) # use NN
    loss_m3tf(Bul,Bvl,Bvdl,xl,yl) = loss_m3(Bul,Bvl,Bvdl,xl,yl,:m3tf,TL_coef_p,TL_coef_i,TL_coef_e,use_nn) # use NN

    model_type == :m3tl && (loss = loss_m3tl)
    model_type == :m3s  && (loss = loss_m3s)
    model_type == :m3v  && (loss = loss_m3v)
    model_type == :m3sc && (loss = loss_m3sc)
    model_type == :m3vc && (loss = loss_m3vc)
    model_type == :m3w  && (loss = loss_m3w)
    model_type == :m3tf && (loss = loss_m3tf)

    function loss_all(data_l)
        l = 0f0
        for (B_unit_l,B_vec_l,B_vec_dot_l,x_l,y_l) in data_l
            l += loss(B_unit_l,B_vec_l,B_vec_dot_l,x_l,y_l)
        end
        return (l/length(data_l))
    end # function loss_all

    # train NN with ADAM optimizer
    TL_coef_store = TL_coef
    m_store       = deepcopy(m)
    best_loss     = loss_all(data_val)
    isempty(x_test) || (best_test_error = std(nn_comp_3_test(B_unit_test,B_vec_test,B_vec_dot_test,
                                                             x_test_norm,y_test,y_bias,y_scale,m,
                                                             TL_coef_p,TL_coef_i,TL_coef_e;
                                                             model_type = model_type,
                                                             y_type     = y_type,
                                                             l_segs     = l_segs_test,
                                                             use_nn     = use_nn,
                                                             denorm     = true,
                                                             testmode   = true,
                                                             silent     = silent_debug)[2]))

    silent || @info("epoch 0: loss = $best_loss")
    for i = 1:epoch_adam
        if model_type in [:m3tl,:m3s,:m3v,:m3w,:m3tf] # train on NN model weights + TL coef
            Flux.train!(loss,Flux.params(m,TL_coef_p,TL_coef_i.data,TL_coef_e),data_train,opt)
        elseif model_type in [:m3sc,:m3vc]
            if     i < epoch_adam * (1 / 10)
                params = Flux.params(TL_coef_p)
            elseif i < epoch_adam * (2 / 10)
                params = Flux.params(TL_coef_p,TL_coef_i.data)
            elseif i < epoch_adam * (3 / 10)
                params = Flux.params(TL_coef_p,TL_coef_i.data,TL_coef_e)
            elseif i < epoch_adam * (6 / 10)
                use_nn = true
                params = Flux.params(m)
                data_train = data_train_2
                if model_type == :m3sc # :m3sc finishes with NN
                    loss = loss_m3s
                elseif model_type == :m3vc # :m3vc finishes with NN
                    loss = loss_m3v
                end
            else
                use_nn = true
                params = Flux.params(m,TL_coef_p,TL_coef_i.data,TL_coef_e)
            end
            Flux.train!(loss,params,data_train,opt)
        end
        TL_coef = TL_mat2vec(TL_coef_p,TL_coef_i,TL_coef_e,terms_A;Bt_scale=Bt_scale)
        current_loss = loss_all(data_val)
        if isempty(x_test)
            if current_loss < best_loss
                best_loss     = current_loss
                m_store       = deepcopy(m)
                TL_coef_store = TL_coef
            end
            mod(i,5) == 0 && !silent && @info("epoch $i: loss = $best_loss")
        else
            test_error = std(nn_comp_3_test(B_unit_test,B_vec_test,B_vec_dot_test,
                                            x_test_norm,y_test,y_bias,y_scale,m,
                                            TL_coef_p,TL_coef_i,TL_coef_e;
                                            model_type = model_type,
                                            y_type     = y_type,
                                            l_segs     = l_segs_test,
                                            use_nn     = use_nn,
                                            denorm     = true,
                                            testmode   = true,
                                            silent     = silent_debug)[2])
            if test_error < best_test_error
                best_test_error = test_error
                m_store         = deepcopy(m)
                TL_coef_store   = TL_coef
            end
            mod(i,5) == 0 && !silent && @info("epoch $i: loss = $current_loss, test error = $(round(best_test_error,digits=2)) nT")
        end
        if (mod(i,10) == 0) & !silent
            err = nn_comp_3_test(B_unit,B_vec,B_vec_dot,
                                 x_norm,y,y_bias,y_scale,m,
                                 TL_coef_p,TL_coef_i,TL_coef_e;
                                 model_type = model_type,
                                 y_type     = y_type,
                                 l_segs     = l_segs,
                                 use_nn     = use_nn,
                                 denorm     = true,
                                 testmode   = true,
                                 silent     = true)[2]
            @info("$i train error: $(round(std(err),digits=2)) nT")
            isempty(x_test) || @info("$i test  error: $(round((test_error),digits=2)) nT")
        end
    end

    Flux.loadmodel!(m,m_store)
    TL_coef = TL_coef_store
    (TL_coef_p,TL_coef_i,TL_coef_e) = TL_vec2mat(TL_coef,terms_A;Bt_scale=Bt_scale)

    if (epoch_lbfgs > 0) & (model_type == :m3tl)
        silent || @info("LBFGS not supported with $model_type model type")
        epoch_lbfgs = 0
    end

    if epoch_lbfgs > 0 # LBFGS, may overfit depending on iterations
        silent || @info("LBFGS will only update NN model weights (not TL coef)")
        data = DataLoader((B_unit,B_vec,B_vec_dot,x_norm,y_norm),shuffle=true,batchsize=batchsize)

        function lbfgs_train!(m_l,data_l,iter,t_l,p_l,i_l,e_l,silent)
            (Bu_l,Bv_l,Bvd_l,x_l,y_l) = data_l.data
            # loss_() = loss(Bu_l,Bv_l,Bvd_l,x_l,y_l)
            loss_() = loss_m3(Bu_l,Bv_l,Bvd_l,x_l,y_l,t_l,p_l,i_l,e_l,true)
            refresh()
            params = Flux.params(m_l,p_l,i_l,e_l)
            opt = LBFGS()
            (_,_,fg!,p0) = optfuns(loss_,params)
            options = Options(iterations=iter,show_every=5,show_trace=!silent)
            optimize(only_fg!(fg!),p0,opt,options)
        end # function lbfgs_train!

        # train NN with LBFGS optimizer
        lbfgs_train!(m,data,epoch_lbfgs,model_type,TL_coef_p,TL_coef_i.data,TL_coef_e,silent)
    end

    TL_coef = TL_mat2vec(TL_coef_p,TL_coef_i,TL_coef_e,terms_A;Bt_scale=Bt_scale)

    # get results
    (y_hat,err) = nn_comp_3_test(B_unit,B_vec,B_vec_dot,
                                 x_norm,y,y_bias,y_scale,m,
                                 TL_coef_p,TL_coef_i,TL_coef_e;
                                 model_type = model_type,
                                 y_type     = y_type,
                                 l_segs     = l_segs,
                                 use_nn     = use_nn,
                                 denorm     = true,
                                 testmode   = true,
                                 silent     = true)
    silent || @info("train error: $(round(std(err),digits=2)) nT")

    if !isempty(x_test)
        nn_comp_3_test(B_unit_test,B_vec_test,B_vec_dot_test,
                       x_test_norm,y_test,y_bias,y_scale,m,
                       TL_coef_p,TL_coef_i,TL_coef_e;
                       model_type = model_type,
                       y_type     = y_type,
                       l_segs     = l_segs_test,
                       use_nn     = use_nn,
                       denorm     = true,
                       testmode   = true,
                       silent     = silent)
    end

    model = m

    # pack data normalizations
    data_norms = (zeros(1,1),zeros(1,1),v_scale,x_bias,x_scale,y_bias,y_scale)

    return (model, TL_coef, data_norms, y_hat, err)
end # function nn_comp_3_train

"""
    nn_comp_3_fwd(B_unit, B_vec, B_vec_dot,
                  x_norm, y_bias, y_scale, model::Chain,
                  TL_coef_p, TL_coef_i, TL_coef_e;
                  model_type::Symbol = :m3s,
                  y_type::Symbol     = :d,
                  use_nn::Bool       = true,
                  denorm::Bool       = true,
                  testmode::Bool     = true)

Evaluate performance of neural network-based aeromagnetic compensation, model 3.
Computes the scalar aircraft field correction or Earth field target. Computes
the Tolles-Lawson vector, optionally adds a vector neural network correction
for model `:m3v` or `:m3vc`, gets the scalar magnitude of the result, and
optionally adds a scalar neural network correction for model `:m3s` or `:m3sc`.

**Arguments:**
- `B_unit`:     `3`  x `N` matrix of normalized vector magnetometer measurements
- `B_vec`:      `3`  x `N` matrix of vector magnetometer measurements
- `B_vec_dot`:  `3`  x `N` matrix of vector magnetometer measurement derivatives
- `x_norm`:     `Nf` x `N` normalized data matrix (`Nf` is number of features)
- `y_bias`:     `y` target vector bias (mean, min, or zero)
- `y_scale`:    `y` target vector scaling factor (std dev, max-min, or one)
- `model`:      neural network model
- `TL_coef_p`:  length `3` vector of permanent field coefficients
- `TL_coef_i`:  `3` x `3`  symmetric matrix of induced field coefficients, denormalized
- `TL_coef_e`:  `3` x `3`  matrix of eddy current coefficients, denormalized
- `model_type`: (optional) aeromagnetic compensation model type
- `y_type`:     (optional) `y` target type
- `use_nn`:     (optional) if true, include neural network contribution to `y_hat`
- `denorm`:     (optional) if true, denormalize `y_hat`
- `testmode`:   (optional) if true, turn on `Flux.testmode!`

**Returns:**
- `y_hat`: length `N` prediction vector
"""
function nn_comp_3_fwd(B_unit, B_vec, B_vec_dot,
                       x_norm, y_bias, y_scale, model::Chain,
                       TL_coef_p, TL_coef_i, TL_coef_e;
                       model_type::Symbol = :m3s,
                       y_type::Symbol     = :d,
                       use_nn::Bool       = true,
                       denorm::Bool       = true,
                       testmode::Bool     = true)

    @assert y_type in [:a,:b,:c,:d] "unsupported y_type = $y_type for nn_comp_3"

    # set to test mode in case model uses batchnorm or dropout
    m = model
    testmode && Flux.testmode!(m)

    # get results
    TL_aircraft  = get_TL_aircraft_vec(B_vec,B_vec_dot,TL_coef_p,TL_coef_i,TL_coef_e)
    vec_aircraft = TL_aircraft

    if (model_type in [:m3v,:m3vc]) & use_nn # vector NN correction to TL
        vec_aircraft += m(x_norm) .* y_scale;
    end

    if y_type in [:c,:d] # aircraft field to subtract from scalar mag
        # vec_aircraft += y_bias .* B_unit # This was worse, overall
        y_hat = vec(sum(vec_aircraft .* B_unit, dims=1)) # dot product
        # println("Aircraft correction = ", y_hat)
    elseif y_type in [:a,:b] # magnitude of scalar Earth field
        B_e   = B_vec - vec_aircraft
        y_hat = vec(sqrt.(sum(B_e.^2,dims=1)))
        # println("Aircraft correction = ", sum(vec_aircraft .* B_unit, dims=1))
        # println("|B_e| = ", y_hat)
    end

    if (model_type in [:m3s,:m3sc,:m3w,:m3tf]) & use_nn # scalar NN correction to TL
        y_hat += vec(m(x_norm)) .* y_scale
    end

    denorm || (y_hat = (y_hat .- y_bias) ./ y_scale)

    return (y_hat)
end # function nn_comp_3_fwd

"""
    nn_comp_3_fwd(A, Bt, B_dot, x, data_norms::Tuple, model::Chain;
                  model_type::Symbol = :m3s,
                  y_type::Symbol     = :d,
                  TL_coef::Vector    = zeros(Float32,18),
                  terms_A            = [:permanent,:induced,:eddy],
                  l_window::Int      = 5)

Evaluate performance of neural network-based aeromagnetic compensation, model 3.
"""
function nn_comp_3_fwd(A, Bt, B_dot, x, data_norms::Tuple, model::Chain;
                       model_type::Symbol = :m3s,
                       y_type::Symbol     = :d,
                       TL_coef::Vector    = zeros(Float32,18),
                       terms_A            = [:permanent,:induced,:eddy],
                       l_window::Int      = 5)

    @assert y_type in [:a,:b,:c,:d] "unsupported y_type = $y_type for nn_comp_3"

    # convert to Float32 for consistency with nn_comp_3_train
    A       = Float32.(A)
    Bt      = Float32.(Bt)    # magnitude of total field measurements
    B_dot   = Float32.(B_dot) # finite differences of total field vector
    x       = Float32.(x)
    TL_coef = Float32.(TL_coef)

    # assume all terms are stored, but they may be zero if not trained
    Bt_scale = 50000f0
    (TL_coef_p,TL_coef_i,TL_coef_e) = TL_vec2mat(TL_coef,terms_A;Bt_scale=Bt_scale)

    B_unit    = A[:,1:3]'     # normalized vector magnetometer reading
    B_vec     = B_unit .* Bt' # vector magnetometer to be used in TL
    B_vec_dot = B_dot'        # not exactly true, but internally consistent

    # unpack data normalizations
    (_,_,v_scale,x_bias,x_scale,y_bias,y_scale) = unpack_data_norms(data_norms)
    x_norm = (((x .- x_bias) ./ x_scale) * v_scale)'

    model_type in [:m3w,:m3tf] && (x_norm = get_temporal_data(x_norm,l_window))

    # get results
    y_hat = nn_comp_3_fwd(B_unit,B_vec,B_vec_dot,
                          x_norm,y_bias,y_scale,model,
                          TL_coef_p,TL_coef_i,TL_coef_e;
                          model_type = model_type,
                          y_type     = y_type,
                          use_nn     = true,
                          denorm     = true,
                          testmode   = true)

    return (y_hat)
end # function nn_comp_3_fwd

"""
    nn_comp_3_test(B_unit, B_vec, B_vec_dot,
                   x_norm, y, y_bias, y_scale, model::Chain,
                   TL_coef_p,TL_coef_i,TL_coef_e;
                   model_type::Symbol = :m3s,
                   y_type::Symbol     = :d,
                   l_segs::Vector     = [length(y)],
                   use_nn::Bool       = true,
                   denorm::Bool       = true,
                   testmode::Bool     = true,
                   silent::Bool       = false)

Evaluate performance of neural network-based aeromagnetic compensation, model 3.
"""
function nn_comp_3_test(B_unit, B_vec, B_vec_dot,
                        x_norm, y, y_bias, y_scale, model::Chain,
                        TL_coef_p,TL_coef_i,TL_coef_e;
                        model_type::Symbol = :m3s,
                        y_type::Symbol      = :d,
                        l_segs::Vector     = [length(y)],
                        use_nn::Bool       = true,
                        denorm::Bool       = true,
                        testmode::Bool     = true,
                        silent::Bool       = false)

    @assert y_type in [:a,:b,:c,:d] "unsupported y_type = $y_type for nn_comp_3"

    # get results
    y_hat = nn_comp_3_fwd(B_unit,B_vec,B_vec_dot,
                          x_norm,y_bias,y_scale,model,
                          TL_coef_p,TL_coef_i,TL_coef_e;
                          model_type = model_type,
                          y_type     = y_type,
                          use_nn     = use_nn,
                          denorm     = denorm,
                          testmode   = testmode)
    err   = err_segs(y_hat,y,l_segs;silent=silent_debug)
    silent || @info("test  error: $(round(std(err),digits=2)) nT")

    return (y_hat, err)
end # function nn_comp_3_test

"""
    nn_comp_3_test(A, Bt, B_dot, x, y, data_norms::Tuple, model::Chain;
                   model_type::Symbol = :m3s,
                   y_type::Symbol     = :d,
                   TL_coef::Vector    = zeros(Float32,18),
                   terms_A            = [:permanent,:induced,:eddy],
                   l_segs::Vector     = [length(y)],
                   l_window::Int      = 5,
                   silent::Bool       = false)

Evaluate performance of neural network-based aeromagnetic compensation, model 3.
"""
function nn_comp_3_test(A, Bt, B_dot, x, y, data_norms::Tuple, model::Chain;
                        model_type::Symbol = :m3s,
                        y_type::Symbol     = :d,
                        TL_coef::Vector    = zeros(Float32,18),
                        terms_A            = [:permanent,:induced,:eddy],
                        l_segs::Vector     = [length(y)],
                        l_window::Int      = 5,
                        silent::Bool       = false)

    @assert y_type in [:a,:b,:c,:d] "unsupported y_type = $y_type for nn_comp_3"

    # convert to Float32 for consistency with nn_comp_3_train
    y = Float32.(y)

    # get results
    y_hat = nn_comp_3_fwd(A,Bt,B_dot,x,data_norms,model;
                          model_type = model_type,
                          y_type     = y_type,
                          TL_coef    = TL_coef,
                          terms_A    = terms_A,
                          l_window   = l_window)
    err   = err_segs(y_hat,y,l_segs;silent=silent_debug)
    silent || @info("test  error: $(round(std(err),digits=2)) nT")

    return (y_hat, err)
end # function nn_comp_3_test

"""
    plsr_fit(x, y, k::Int=size(x,2);
             l_segs::Vector   = [length(y)],
             return_set::Bool = false,
             silent::Bool     = false)

Fit a multi-input, multi-output (MIMO) partial least squares regression (PLSR)
model to data with a specified output dimension. PLSR is a type of regularized
linear regression where the number of components controls the strength of the
regularization.

**Arguments:**
- `x`:          `N` x `Nf` data matrix (`Nf` is number of features)
- `y`:          length `N` target vector
- `k`:          (optional) number of components
- `l_segs`:     (optional) length `N_lines` vector of lengths of `lines`, sum(l_segs) = `N`
- `return_set`: (optional) if true, return `coef_set` instead of other outputs
- `silent`:     (optional) if true, no print outs

**Returns:**
- `model`:      tuple of PLSR-based model `(coefficients, bias=0)`
- `data_norms`: tuple of data normalizations, e.g., `(x_bias,x_scale,y_bias,y_scale)`
- `y_hat`:      length `N` prediction vector
- `err`:        length `N` mean-corrected (per line) error
- `coef_set`:   if `return_set = true`, set of coefficients (size `Nf` x `Ny` x `k`)
"""
function plsr_fit(x, y, k::Int=size(x,2);   # N x Nf , N x Ny , k
                  l_segs::Vector   = [length(y)],
                  return_set::Bool = false,
                  silent::Bool     = false)

	Nf = size(x,2)
    Ny = size(y,2)

    if k > Nf
        silent || @info("reducing k from $k to $Nf")
        k = Nf
    end

    # standardize data
    (x_bias,x_scale,x_norm) = norm_sets(x;norm_type=:standardize)
    (y_bias,y_scale,y_norm) = norm_sets(y;norm_type=:standardize)

    x_temp   = sum(x_norm.^2)           # scalar
    y_temp   = sum(y_norm.^2)           # scalar
    p_out    = zeros(eltype(x),Nf,k)    # Nf x k
    q_out    = zeros(eltype(x),Ny,k)    # Ny x k
    u_out    = zeros(eltype(x),Nf,k)    # Nf x k
    coef_set = zeros(eltype(x),Nf,Ny,k) # Nf x Ny x k

    # covariance & cross-covariance matrices
	Cxx = cov(x_norm)                   # Nf x Nf
	Cyx = collect(cov(y_norm,x_norm))   # Ny x Nf

    for i = 1:k

        # unit vectors that maximize correlation between input & output scores
        (U,_,V) = svd(Cyx')      # Nf x Ny , _ , Ny x Ny
        u = U[:,1:1]             # Nf                      # x'*y ./ norm(x'*y)
        v = V[:,1:1]             # Ny

        # input & output scores, input & output loading vectors
        z = x_norm*u             # N
        r = y_norm*v             # N
        p = (Cxx*u) / (u'*Cxx*u) # Nf                      # x'*z ./ norm(z)^2
        q = (Cyx*u) / (u'*Cxx*u) # Ny                      # y'*z ./ norm(z)^2

        # deflated covariance & cross-covariance matrices
        Cxx = (I(Nf) - p*u')*Cxx # Nf x Nf
        Cyx = Cyx*(I(Nf) - u*p') # Ny x Nf

        # deflated input & output data
        x_norm = x_norm - z*p'   # N  x Nf                 # x*(u*p')
        y_norm = y_norm - z*q'   # N  x Ny                 # x*(u*q')

        p_out[:,i] = p           # Nf x k
        q_out[:,i] = q           # Ny x k
        u_out[:,i] = u           # Nf x k

        if return_set
            coef_set[:,:,i] = u_out[:,1:i]*inv(p_out[:,1:i]'*u_out[:,1:i])*q_out[:,1:i]' # Nf x Ny x k
        end

    end

    return_set && return (coef_set)

    coef = vec(u_out*inv(p_out'*u_out)*q_out') # Nf x Ny
    bias = zero(eltype(coef))

    # input & output residue variance %
    silent || @info("input  residue variance = $(round(sum(x_norm.^2)/x_temp*100,digits=2)) %")
    silent || @info("output residue variance = $(round(sum(y_norm.^2)/y_temp*100,digits=2)) %")

    # get results
    y_hat_norm = norm_sets(x;norm_type=:standardize)[3]*coef .+ bias
    y_hat      = denorm_sets(y_bias,y_scale,y_hat_norm)
    err        = err_segs(y_hat,y,l_segs;silent=silent_debug)
    silent || @info("fit  error: $(round(std(err),digits=2)) nT")

    # pack data normalizations
    data_norms = (x_bias,x_scale,y_bias,y_scale)

    return ((coef, bias), data_norms, y_hat, err)
end # function plsr_fit

# #* note: the existing PLSR Julia package (PartialLeastSquaresRegressor.jl)
# #* gives the exact same result, but takes 3x longer, requires more
# #* dependencies, has issues working in src, and only provides output for a
# #* single k per evaluation
# regressor = PLSRegressor(n_factors=k) # create model
# plsr_p = @pipeline Standardizer regressor target=Standardizer # build pipeline
# plsr_m = machine(plsr_p,DataFrame(x,features),y) # create machine
# fit!(plsr_m) # fit model

"""
    elasticnet_fit(x, y, α=0.99;  λ = -1,
                   l_segs::Vector = [length(y)],
                   silent::Bool   = false)

Fit an elastic net (ridge regression and/or Lasso) model to data.

**Arguments:**
- `x`:      `N` x `Nf` data matrix (`Nf` is number of features)
- `y`:      length `N` target vector
- `α`:      (optional) ridge regression (`α=0`) vs Lasso (`α=1`) balancing parameter {0:1}
- `λ`:      (optional) elastic net parameter (otherwise determined with cross-validation), `-1` to ignore
- `l_segs`: (optional) length `N_lines` vector of lengths of `lines`, sum(l_segs) = `N`
- `silent`: (optional) if true, no print outs

**Returns:**
- `model`:      tuple of elastic net-based model `(coefficients,bias)`
- `data_norms`: tuple of data normalizations, e.g., `(x_bias,x_scale,y_bias,y_scale)`
- `y_hat`:      length `N` prediction vector
- `err`:        length `N` mean-corrected (per line) error
"""
function elasticnet_fit(x, y, α=0.99;  λ = -1,
                        l_segs::Vector = [length(y)],
                        silent::Bool   = false)

    # standardize data
    (x_bias,x_scale,x_norm) = norm_sets(x;norm_type=:standardize)
    (y_bias,y_scale,y_norm) = norm_sets(y;norm_type=:standardize)

    if λ < 0
        # determine best λ with cross-validation with GLMNet
        cv_glm  = glmnetcv(x_norm,y_norm;standardize=false,alpha=α)
        λr      = (1-α)*cv_glm.lambda[end] # ridge parameter
        λl      =    α *cv_glm.lambda[end] # Lasso parameter
    else
        λr      = (1-α)*λ
        λl      =    α *λ
    end

    # get coefficients and bias with MLJLinearModels
    glr_mlj = ElasticNetRegression(λr,λl;scale_penalty_with_samples=false)
    fit_mlj = fit(glr_mlj,x_norm,y_norm)
    coef    = fit_mlj[1:end-1]
    bias    = fit_mlj[end]

    # get results
    y_hat_norm = x_norm*coef .+ bias
    y_hat      = denorm_sets(y_bias,y_scale,y_hat_norm)
    err        = err_segs(y_hat,y,l_segs;silent=silent_debug)
    silent || @info("fit  error: $(round(std(err),digits=2)) nT")

    # pack data normalizations
    data_norms = (x_bias,x_scale,y_bias,y_scale)

    return ((coef, bias), data_norms, y_hat, err)
end # function elasticnet_fit

"""
    linear_fit(x, y;
               trim::Int=0, λ=0,
               norm_type_x::Symbol = :none,
               norm_type_y::Symbol = :none,
               l_segs::Vector      = [length(y)],
               silent::Bool        = false)

Fit a linear regression model to data.

**Arguments:**
- `x`:           `N` x `Nf` data matrix (`Nf` is number of features)
- `y`:           length `N` target vector
- `trim`:        (optional) number of elements to trim (e.g., due to bpf)
- `λ`:           (optional) ridge parameter
- `norm_type_x`: (optional) normalization for `x` data matrix
- `norm_type_y`: (optional) normalization for `y` target vector
- `l_segs`:      (optional) length `N_lines` vector of lengths of `lines`, sum(l_segs) = `N`
- `silent`:      (optional) if true, no print outs

**Returns:**
- `model`:      tuple of linear regression model `(coefficients, bias=0)`
- `data_norms`: tuple of data normalizations, e.g., `(x_bias,x_scale,y_bias,y_scale)`
- `y_hat`:      length `N` prediction vector
- `err`:        length `N` mean-corrected (per line) error
"""
function linear_fit(x, y;
                    trim::Int=0, λ=0,
                    norm_type_x::Symbol = :none,
                    norm_type_y::Symbol = :none,
                    l_segs::Vector      = [length(y)],
                    silent::Bool        = false)

    # standardize data
    (x_bias,x_scale,x_norm) = norm_sets(x;norm_type=norm_type_x)
    (y_bias,y_scale,y_norm) = norm_sets(y;norm_type=norm_type_y)

    # trim each line
    ind = []
    for i in eachindex(l_segs)
        (i1,i2) = cumsum(l_segs)[i] .- (l_segs[i]-1-trim,trim)
        ind = [ind;i1:i2]
    end

    # linear regression to get coefficients
    coef = vec(linreg(y_norm[ind,:],x_norm[ind,:];λ=λ))
    bias = zero(eltype(coef))

    y_hat_norm = x_norm*coef .+ bias
    y_hat      = denorm_sets(y_bias,y_scale,y_hat_norm)
    err        = err_segs(y_hat,y,l_segs;silent=silent_debug)
    silent || @info("fit  error: $(round(std(err),digits=2)) nT")
    silent || @info("note that fit error may be misleading if using bpf")

    # pack data normalizations
    data_norms = (x_bias,x_scale,y_bias,y_scale)

    return ((coef, bias), data_norms, y_hat, err)
end # function linear_fit

"""
    linear_fwd(x_norm, y_bias, y_scale, model)

Evaluate performance of linear model.

**Arguments:**
- `x_norm`:  `N` x `Nf` normalized data matrix (`Nf` is number of features)
- `y_bias`:  `y` target vector bias (mean, min, or zero)
- `y_scale`: `y` target vector scaling factor (std dev, max-min, or one)
- `model`:   tuple of model `(coefficients,bias)`

**Returns:**
- `y_hat`: length `N` prediction vector
"""
function linear_fwd(x_norm, y_bias, y_scale, model)

    # unpack linear model weights
    (coef,bias) = model

    # get results
    y_hat_norm = x_norm*coef .+ bias
    y_hat      = denorm_sets(y_bias,y_scale,y_hat_norm)

    return (y_hat)
end # function linear_fwd

"""
    linear_fwd(x, data_norms::Tuple, model)

Evaluate performance of linear model.

**Arguments:**
- `x`:          `N` x `Nf` data matrix (`Nf` is number of features)
- `data_norms`: tuple of data normalizations, e.g., `(x_bias,x_scale,y_bias,y_scale)`
- `model`:      tuple of model `(coefficients,bias)`

**Returns:**
- `y_hat`: length `N` prediction vector
"""
function linear_fwd(x, data_norms::Tuple, model)

    # unpack data normalizations
    (x_bias,x_scale,y_bias,y_scale) = data_norms
    x_norm = (x .- x_bias) ./ x_scale

    # get results
    y_hat = linear_fwd(x_norm,y_bias,y_scale,model)

    return (y_hat)
end # function linear_fwd

"""
    linear_test(x_norm, y, y_bias, y_scale, model;
                l_segs::Vector = [length(y)],
                silent::Bool   = false)

Evaluate performance of linear model.

**Arguments:**
- `x_norm`:  `N` x `Nf` normalized data matrix (`Nf` is number of features)
- `y`:       length `N` target vector
- `y_bias`:  `y` target vector bias bias (mean, min, or zero)
- `y_scale`: `y` target vector bias scaling factor (std dev, max-min, or one)
- `model`:   tuple of model `(coefficients,bias)`
- `l_segs`:  (optional) length `N_lines` vector of lengths of `lines`, sum(l_segs) = `N`
- `silent`:  (optional) if true, no print outs

**Returns:**
- `y_hat`: length `N` prediction vector
- `err`:   length `N` mean-corrected (per line) error
"""
function linear_test(x_norm, y, y_bias, y_scale, model;
                     l_segs::Vector = [length(y)],
                     silent::Bool   = false)

    # get results
    y_hat = linear_fwd(x_norm,y_bias,y_scale,model)
    err   = err_segs(y_hat,y,l_segs;silent=silent_debug)
    silent || @info("test  error: $(round(std(err),digits=2)) nT")

    return (y_hat, err)
end # function linear_test

"""
    linear_test(x, y, data_norms::Tuple, model;
                l_segs::Vector = [length(y)],
                silent::Bool   = false)

Evaluate performance of linear model.

**Arguments:**
- `x`:          `N` x `Nf` data matrix (`Nf` is number of features)
- `y`:          length `N` target vector
- `data_norms`: tuple of data normalizations, e.g., `(x_bias,x_scale,y_bias,y_scale)`
- `model`:      tuple of model `(coefficients,bias)`
- `l_segs`:     (optional) length `N_lines` vector of lengths of `lines`, sum(l_segs) = `N`
- `silent`:     (optional) if true, no print outs

**Returns:**
- `y_hat`: length `N` prediction vector
- `err`:   length `N` mean-corrected (per line) error
"""
function linear_test(x, y, data_norms::Tuple, model;
                     l_segs::Vector = [length(y)],
                     silent::Bool   = false)

    # get results
    y_hat = linear_fwd(x,data_norms,model)
    err   = err_segs(y_hat,y,l_segs;silent=silent_debug)
    silent || @info("test  error: $(round(std(err),digits=2)) nT")

    return (y_hat, err)
end # function linear_test

"""
    comp_train(comp_params::CompParams, xyz::XYZ, ind,
               mapS::Union{MapS,MapSd,MapS3D} = mapS_null;
               xyz_test::XYZ         = xyz,
               ind_test              = BitVector(),
               σ_curriculum          = 1.0,
               l_window::Int         = 5,
               window_type::Symbol   = :sliding,
               tf_layer_type::Symbol = :postlayer,
               tf_norm_type::Symbol  = :batch,
               dropout_prob          = 0.2,
               N_tf_head::Int        = 8,
               tf_gain               = 1.0,
               silent::Bool          = false)

Train an aeromagnetic compensation model.

**Arguments:**
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct, either:
    - `NNCompParams`:  neural network-based aeromagnetic compensation parameters struct
    - `LinCompParams`: linear aeromagnetic compensation parameters struct
- `xyz`:           `XYZ` flight data struct
- `ind`:           selected data indices
- `mapS`:          (optional) `MapS`, `MapSd`, or `MapS3D` scalar magnetic anomaly map struct, only used for `y_type = :b, :c`
- `xyz_test`:      (optional) `XYZ` held-out test data struct
- `ind_test`:      (optional) indices for test data struct
- `σ_curriculum`:  (optional) standard deviation threshold, only used for `model_type = :m3sc, :m3vc`
- `l_window`:      (optional) temporal window length, only used for `model_type = :m3w, :m3tf`
- `window_type`:   (optional) type of windowing, `:sliding` for overlapping or `:contiguous` for non-overlapping, only used for `model_type = :m3w, :m3tf`
- `tf_layer_type`: (optional) transformer normalization layer before or after skip connection {`:prelayer`,`:postlayer`}, only used for `model_type = :m3tf`
- `tf_norm_type`:  (optional) normalization for transformer encoder {`:batch`,`:layer`,`:none`}, only used for `model_type = :m3tf`
- `dropout_prob`:  (optional) dropout rate, only used for `model_type = :m3w, :m3tf`
- `N_tf_head`:     (optional) number of attention heads, only used for `model_type = :m3tf`
- `tf_gain`:       (optional) weight initialization parameter, only used for `model_type = :m3tf`
- `silent`:        (optional) if true, no print outs

**Returns:**
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct
- `y`:           length `N` target vector
- `y_hat`:       length `N` prediction vector
- `err`:         length `N` compensation error
- `features`:    length `Nf` feature vector (including components of TL `A`, etc.)
"""
function comp_train(comp_params::CompParams, xyz::XYZ, ind,
                    mapS::Union{MapS,MapSd,MapS3D} = mapS_null;
                    xyz_test::XYZ         = xyz,
                    ind_test              = BitVector(),
                    σ_curriculum          = 1.0,
                    l_window::Int         = 5,
                    window_type::Symbol   = :sliding,
                    tf_layer_type::Symbol = :postlayer,
                    tf_norm_type::Symbol  = :batch,
                    dropout_prob          = 0.2,
                    N_tf_head::Int        = 8,
                    tf_gain               = 1.0,
                    silent::Bool          = false)

    seed!(2) # for reproducibility
    t0 = time()

    # unpack parameters
    if typeof(comp_params) <: NNCompParams
        @unpack version, features_setup, features_no_norm, model_type, y_type,
        use_mag, use_vec, data_norms, model, terms, terms_A, sub_diurnal,
        sub_igrf, bpf_mag, reorient_vec, norm_type_A, norm_type_x, norm_type_y,
        TL_coef, η_adam, epoch_adam, epoch_lbfgs, hidden, activation,
        batchsize, frac_train, α_sgl, λ_sgl, k_pca,
        drop_fi, drop_fi_bson, drop_fi_csv, perm_fi, perm_fi_csv = comp_params
    elseif typeof(comp_params) <: LinCompParams
        @unpack version, features_setup, features_no_norm, model_type, y_type,
        use_mag, use_vec, data_norms, model, terms, terms_A, sub_diurnal,
        sub_igrf, bpf_mag, reorient_vec, norm_type_A, norm_type_x, norm_type_y,
        k_plsr, λ_TL = comp_params
        drop_fi = false
        perm_fi = false
    end

    if (model_type in [:TL,:mod_TL]) & (y_type != :e)
        silent || @info("forcing y_type $(y_type) => :e (BPF'd total field)")
        y_type = :e
    end
    if (model_type in [:map_TL]) & (y_type != :c)
        silent || @info("forcing y_type $(y_type) => :c (aircraft field #1, using map)")
        y_type = :c
    end
    if model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc,:m3w,:m3tf]
        for term in terms_A
            if term in [:fdm,:f,:fdm3,:f3,:bias,:b]
                silent || @info("removing $term term from terms_A")
                A_test = create_TL_A([1.0],[1.0],[1.0];terms=terms_A)
                if length(A_test) == length(TL_coef)
                    ind_term = get_TL_term_ind(term,terms_A)
                    silent || @info("removing $(sum(ind_term)) value(s) from TL_coef")
                    TL_coef = TL_coef[.!ind_term]
                end
                terms_A = terms_A[terms_A .!= term]
            end
        end
    end

    # map values along trajectory (if needed)
    map_val = y_type in [:b,:c] ? get_map_val(mapS,xyz.traj,ind;α=200) : -1

    # `A` matrix for selected vector magnetometer
    field_check(xyz,use_vec,MagV)
    if model_type == :mod_TL
        A = create_TL_A(getfield(xyz,use_vec),ind;
                        Bt=getfield(xyz,use_mag),terms=terms_A)
    elseif model_type == :map_TL
        A = create_TL_A(getfield(xyz,use_vec),ind;
                        Bt=map_val,terms=terms_A)
    else
        (A,Bt,B_dot) = create_TL_A(getfield(xyz,use_vec),ind;
                                   terms=terms_A,return_B=true)
    end
    fs = 1 / xyz.traj.dt
    model_type in [:TL,:mod_TL] && (A_no_bpf = deepcopy(A))
    y_type == :e && bpf_data!(A;bpf=get_bpf(;fs=fs))

    # load data
    (x,no_norm,features) = get_x(xyz,ind,features_setup;
                                 features_no_norm = features_no_norm,
                                 terms            = terms,
                                 sub_diurnal      = sub_diurnal,
                                 sub_igrf         = sub_igrf,
                                 bpf_mag          = bpf_mag)

    y = get_y(xyz,ind,map_val;
              y_type      = y_type,
              use_mag     = use_mag,
              sub_diurnal = sub_diurnal,
              sub_igrf    = sub_igrf)

    model_type in [:TL,:mod_TL] && (y_no_bpf = get_y(xyz,ind,map_val;
                                                     y_type      = :d,
                                                     use_mag     = use_mag,
                                                     sub_diurnal = sub_diurnal,
                                                     sub_igrf    = sub_igrf))

    # set held-out test line
    A_test     = Matrix{eltype(A)}(undef,0,0)
    Bt_test    = Vector{eltype(x)}(undef,0)
    B_dot_test = Matrix{eltype(x)}(undef,0,0)
    x_test     = Matrix{eltype(x)}(undef,0,0)
    y_test     = Vector{eltype(y)}(undef,0)
    if !isempty(ind_test)
        (x_test,_,_) = get_x(xyz_test,ind_test,features_setup;
                             features_no_norm = features_no_norm,
                             terms            = terms,
                             sub_diurnal      = sub_diurnal,
                             sub_igrf         = sub_igrf,
                             bpf_mag          = bpf_mag)
        y_test = get_y(xyz_test,ind_test,map_val;
                       y_type      = y_type,
                       use_mag     = use_mag,
                       sub_diurnal = sub_diurnal,
                       sub_igrf    = sub_igrf)
        (A_test,Bt_test,B_dot_test) =
            create_TL_A(getfield(xyz_test,use_vec),ind_test;
                        terms=terms_A,return_B=true)
    end

    y_hat = zero(y) # initialize
    err   = 10*y    # initialize

    if drop_fi

        drop_fi_bson = remove_extension(drop_fi_bson,".bson")

        for i in axes(x,2)

            x_fi = x[:, axes(x,2) .!= i]
            x_test_fi = isempty(x_test) ? x_test : x_test[:,axes(x_test,2) .!=i]

            # train model
            if model_type in [:m1]
                (model,data_norms,y_hat_fi,err_fi) =
                    nn_comp_1_train(x_fi,y,no_norm;
                                    norm_type_x = norm_type_x,
                                    norm_type_y = norm_type_y,
                                    η_adam      = η_adam,
                                    epoch_adam  = epoch_adam,
                                    epoch_lbfgs = epoch_lbfgs,
                                    hidden      = hidden,
                                    activation  = activation,
                                    batchsize   = batchsize,
                                    frac_train  = frac_train,
                                    α_sgl       = α_sgl,
                                    λ_sgl       = λ_sgl,
                                    k_pca       = k_pca,
                                    x_test      = x_test_fi,
                                    y_test      = y_test,
                                    silent      = silent)
            elseif model_type in [:m2a,:m2b,:m2c,:m2d]
                (model,TL_coef,data_norms,y_hat_fi,err_fi) =
                    nn_comp_2_train(A,x_fi,y,no_norm;
                                    model_type  = model_type,
                                    norm_type_A = norm_type_A,
                                    norm_type_x = norm_type_x,
                                    norm_type_y = norm_type_y,
                                    TL_coef     = TL_coef,
                                    η_adam      = η_adam,
                                    epoch_adam  = epoch_adam,
                                    epoch_lbfgs = epoch_lbfgs,
                                    hidden      = hidden,
                                    activation  = activation,
                                    batchsize   = batchsize,
                                    frac_train  = frac_train,
                                    α_sgl       = α_sgl,
                                    λ_sgl       = λ_sgl,
                                    k_pca       = k_pca,
                                    A_test      = A_test,
                                    x_test      = x_test_fi,
                                    y_test      = y_test,
                                    silent      = silent)
            elseif model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc,:m3w,:m3tf]
                (model,TL_coef,data_norms,y_hat_fi,err_fi) =
                    nn_comp_3_train(A,Bt,B_dot,x_fi,y,no_norm;
                                    model_type    = model_type,
                                    norm_type_x   = norm_type_x,
                                    norm_type_y   = norm_type_y,
                                    TL_coef       = TL_coef,
                                    terms_A       = terms_A,
                                    y_type        = y_type,
                                    η_adam        = η_adam,
                                    epoch_adam    = epoch_adam,
                                    epoch_lbfgs   = epoch_lbfgs,
                                    hidden        = hidden,
                                    activation    = activation,
                                    batchsize     = batchsize,
                                    frac_train    = frac_train,
                                    α_sgl         = α_sgl,
                                    λ_sgl         = λ_sgl,
                                    k_pca         = k_pca,
                                    σ_curriculum  = σ_curriculum,
                                    l_window      = l_window,
                                    window_type   = window_type,
                                    tf_layer_type = tf_layer_type,
                                    tf_norm_type  = tf_norm_type,
                                    dropout_prob  = dropout_prob,
                                    N_tf_head     = N_tf_head,
                                    tf_gain       = tf_gain,
                                    A_test        = A_test,
                                    Bt_test       = Bt_test,
                                    B_dot_test    = B_dot_test,
                                    x_test        = x_test_fi,
                                    y_test        = y_test,
                                    silent        = silent)
            else
                error("$model_type model type not defined")
            end

            if std(err_fi) < std(err)
                y_hat = y_hat_fi
                err   = err_fi
            end

            comp_params = NNCompParams(comp_params,
                                       data_norms = data_norms,
                                       model      = model,
                                       terms_A    = terms_A,
                                       TL_coef    = TL_coef)
            save_comp_params(comp_params,drop_fi_bson*"_$i.bson")

        end

    else

        # train model
        if model_type in [:m1]
            (model,data_norms,y_hat,err) =
                nn_comp_1_train(x,y,no_norm;
                                norm_type_x = norm_type_x,
                                norm_type_y = norm_type_y,
                                η_adam      = η_adam,
                                epoch_adam  = epoch_adam,
                                epoch_lbfgs = epoch_lbfgs,
                                hidden      = hidden,
                                activation  = activation,
                                batchsize   = batchsize,
                                frac_train  = frac_train,
                                α_sgl       = α_sgl,
                                λ_sgl       = λ_sgl,
                                k_pca       = k_pca,
                                data_norms  = data_norms,
                                model       = model,
                                x_test      = x_test,
                                y_test      = y_test,
                                silent      = silent)
        elseif model_type in [:m2a,:m2b,:m2c,:m2d]
            (model,TL_coef,data_norms,y_hat,err) =
                nn_comp_2_train(A,x,y,no_norm;
                                model_type  = model_type,
                                norm_type_A = norm_type_A,
                                norm_type_x = norm_type_x,
                                norm_type_y = norm_type_y,
                                TL_coef     = TL_coef,
                                η_adam      = η_adam,
                                epoch_adam  = epoch_adam,
                                epoch_lbfgs = epoch_lbfgs,
                                hidden      = hidden,
                                activation  = activation,
                                batchsize   = batchsize,
                                frac_train  = frac_train,
                                α_sgl       = α_sgl,
                                λ_sgl       = λ_sgl,
                                k_pca       = k_pca,
                                data_norms  = data_norms,
                                model       = model,
                                A_test      = A_test,
                                x_test      = x_test,
                                y_test      = y_test,
                                silent      = silent)
        elseif model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc,:m3w,:m3tf]
            (model,TL_coef,data_norms,y_hat,err) =
                nn_comp_3_train(A,Bt,B_dot,x,y,no_norm;
                                model_type    = model_type,
                                norm_type_x   = norm_type_x,
                                norm_type_y   = norm_type_y,
                                TL_coef       = TL_coef,
                                terms_A       = terms_A,
                                y_type        = y_type,
                                η_adam        = η_adam,
                                epoch_adam    = epoch_adam,
                                epoch_lbfgs   = epoch_lbfgs,
                                hidden        = hidden,
                                activation    = activation,
                                batchsize     = batchsize,
                                frac_train    = frac_train,
                                α_sgl         = α_sgl,
                                λ_sgl         = λ_sgl,
                                k_pca         = k_pca,
                                σ_curriculum  = σ_curriculum,
                                l_window      = l_window,
                                window_type   = window_type,
                                tf_layer_type = tf_layer_type,
                                tf_norm_type  = tf_norm_type,
                                dropout_prob  = dropout_prob,
                                N_tf_head     = N_tf_head,
                                tf_gain       = tf_gain,
                                A_test        = A_test,
                                Bt_test       = Bt_test,
                                B_dot_test    = B_dot_test,
                                x_test        = x_test,
                                y_test        = y_test,
                                silent        = silent)
        elseif model_type in [:TL,:mod_TL,:map_TL]
            trim = model_type in [:TL,:mod_TL] ? 20 : 0
            (model,data_norms,y_hat,err) =
                linear_fit(A,y;
                           trim=trim, λ=λ_TL,
                           norm_type_x = norm_type_A,
                           norm_type_y = norm_type_y,
                           silent      = silent)
            if model_type in [:TL,:mod_TL]
                (y_hat,err) = linear_test(A_no_bpf,y_no_bpf,data_norms,model;
                                          silent=silent)
            end
        elseif model_type in [:elasticnet]
            (model,data_norms,y_hat,err) =
                elasticnet_fit(x,y; silent=silent)
        elseif model_type in [:plsr]
            (model,data_norms,y_hat,err) =
                plsr_fit(x,y,k_plsr; silent=silent)
        else
            error("$model_type model type not defined")
        end

    end

    if typeof(comp_params) <: NNCompParams
        comp_params = NNCompParams(comp_params,
                                   data_norms = data_norms,
                                   model      = model,
                                   terms_A    = terms_A,
                                   TL_coef    = TL_coef)
    elseif typeof(comp_params) <: LinCompParams
        comp_params = LinCompParams(comp_params,
                                    data_norms = data_norms,
                                    model      = model)
    end

    silent || print_time(time()-t0,1)

    return (comp_params, y, y_hat, err, features)
end # function comp_train

function comp_train(xyz::XYZ, ind, mapS::Union{MapS,MapSd,MapS3D} = mapS_null;
                    comp_params::CompParams = NNCompParams(),
                    xyz_test::XYZ           = xyz,
                    ind_test                = BitVector(),
                    σ_curriculum            = 1.0,
                    l_window::Int           = 5,
                    window_type::Symbol     = :sliding,
                    tf_layer_type::Symbol   = :postlayer,
                    tf_norm_type::Symbol    = :batch,
                    dropout_prob            = 0.2,
                    N_tf_head::Int          = 8,
                    tf_gain                 = 1.0,
                    silent::Bool            = false)
    @warn("this version of comp_train() is deprecated & will be removed in MagNav.jl v1.2.0, see docstring for argument order")
    comp_train(comp_params,xyz,ind,mapS;
               xyz_test      = xyz_test,
               ind_test      = ind_test,
               σ_curriculum  = σ_curriculum,
               l_window      = l_window,
               window_type   = window_type,
               tf_layer_type = tf_layer_type,
               tf_norm_type  = tf_norm_type,
               dropout_prob  = dropout_prob,
               N_tf_head     = N_tf_head,
               tf_gain       = tf_gain,
               silent        = silent)
end # function comp_train

"""
    comp_train(comp_params::CompParams, xyz_vec::Vector, ind_vec::Vector,
               mapS::Union{MapS,MapSd,MapS3D} = mapS_null;
               xyz_test::XYZ         = xyz_vec[1],
               ind_test              = BitVector(),
               σ_curriculum          = 1.0,
               l_window::Int         = 5,
               window_type::Symbol   = :sliding,
               tf_layer_type::Symbol = :postlayer,
               tf_norm_type::Symbol  = :batch,
               dropout_prob          = 0.2,
               N_tf_head::Int        = 8,
               tf_gain               = 1.0,
               silent::Bool          = false)

Train an aeromagnetic compensation model.

**Arguments:**
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct, either:
    - `NNCompParams`:  neural network-based aeromagnetic compensation parameters struct
    - `LinCompParams`: linear aeromagnetic compensation parameters struct
- `xyz_vec`:       vector of `XYZ` flight data structs
- `ind_vec`:       vector of selected data indices
- `mapS`:          (optional) `MapS`, `MapSd`, or `MapS3D` scalar magnetic anomaly map struct, only used for `y_type = :b, :c`
- `xyz_test`:      (optional) `XYZ` held-out test data struct
- `ind_test`:      (optional) indices for test data struct
- `σ_curriculum`:  (optional) standard deviation threshold, only used for `model_type = :m3sc, :m3vc`
- `l_window`:      (optional) temporal window length, only used for `model_type = :m3w, :m3tf`
- `window_type`:   (optional) type of windowing, `:sliding` for overlapping or `:contiguous` for non-overlapping, only used for `model_type = :m3w, :m3tf`
- `tf_layer_type`: (optional) transformer normalization layer before or after skip connection {`:prelayer`,`:postlayer`}, only used for `model_type = :m3tf`
- `tf_norm_type`:  (optional) normalization for transformer encoder {`:batch`,`:layer`,`:none`}, only used for `model_type = :m3tf`
- `dropout_prob`:  (optional) dropout rate, only used for `model_type = :m3w, :m3tf`
- `N_tf_head`:     (optional) number of attention heads, only used for `model_type = :m3tf`
- `tf_gain`:       (optional) weight initialization parameter, only used for `model_type = :m3tf`
- `silent`:        (optional) if true, no print outs

**Returns:**
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct
- `y`:           length `N` target vector
- `y_hat`:       length `N` prediction vector
- `err`:         length `N` compensation error
- `features`:    length `Nf` feature vector (including components of TL `A`, etc.)
"""
function comp_train(comp_params::CompParams, xyz_vec::Vector, ind_vec::Vector,
                    mapS::Union{MapS,MapSd,MapS3D} = mapS_null;
                    xyz_test::XYZ         = xyz_vec[1],
                    ind_test              = BitVector(),
                    σ_curriculum          = 1.0,
                    l_window::Int         = 5,
                    window_type::Symbol   = :sliding,
                    tf_layer_type::Symbol = :postlayer,
                    tf_norm_type::Symbol  = :batch,
                    dropout_prob          = 0.2,
                    N_tf_head::Int        = 8,
                    tf_gain               = 1.0,
                    silent::Bool          = false)

    seed!(2) # for reproducibility
    t0 = time()

    # unpack parameters
    if typeof(comp_params) <: NNCompParams
        @unpack version, features_setup, features_no_norm, model_type, y_type,
        use_mag, use_vec, data_norms, model, terms, terms_A, sub_diurnal,
        sub_igrf, bpf_mag, reorient_vec, norm_type_A, norm_type_x, norm_type_y,
        TL_coef, η_adam, epoch_adam, epoch_lbfgs, hidden, activation,
        batchsize, frac_train, α_sgl, λ_sgl, k_pca,
        drop_fi, drop_fi_bson, drop_fi_csv, perm_fi, perm_fi_csv = comp_params
    elseif typeof(comp_params) <: LinCompParams
        @unpack version, features_setup, features_no_norm, model_type, y_type,
        use_mag, use_vec, data_norms, model, terms, terms_A, sub_diurnal,
        sub_igrf, bpf_mag, reorient_vec, norm_type_A, norm_type_x, norm_type_y,
        k_plsr, λ_TL = comp_params
        drop_fi = false
        perm_fi = false
    end

    if (model_type in [:TL,:mod_TL]) & (y_type != :e)
        silent || @info("forcing y_type $(y_type) => :e (BPF'd total field)")
        y_type = :e
    end
    if (model_type in [:map_TL]) & (y_type != :c)
        silent || @info("forcing y_type $(y_type) => :c (aircraft field #1, using map)")
        y_type = :c
    end
    if model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc,:m3w,:m3tf]
        for term in terms_A
            if term in [:fdm,:f,:fdm3,:f3,:bias,:b]
                silent || @info("removing $term term from terms_A")
                A_test = create_TL_A([1.0],[1.0],[1.0];terms=terms_A)
                if length(A_test) == length(TL_coef)
                    ind_term = get_TL_term_ind(term,terms_A)
                    silent || @info("removing $(sum(ind_term)) value(s) from TL_coef")
                    TL_coef = TL_coef[.!ind_term]
                end
                terms_A = terms_A[terms_A .!= term]
            end
        end
    end

    # initialize loop over XYZ structs and indices
    xyz = xyz_vec[1]
    ind = ind_vec[1]

    # map values along trajectory (if needed)
    map_val = y_type in [:b,:c] ? get_map_val(mapS,xyz.traj,ind;α=200) : -1

    # `A` matrix for selected vector magnetometer
    field_check(xyz,use_vec,MagV)
    if model_type == :mod_TL
        A = create_TL_A(getfield(xyz,use_vec),ind;
                        Bt=getfield(xyz,use_mag),terms=terms_A)
    elseif model_type == :map_TL
        A = create_TL_A(getfield(xyz,use_vec),ind;
                        Bt=map_val,terms=terms_A)
    else
        (A,Bt,B_dot) = create_TL_A(getfield(xyz_vec[1],use_vec),ind;
                                   terms=terms_A,return_B=true)
    end
    fs = 1 / xyz.traj.dt
    model_type in [:TL,:mod_TL] && (A_no_bpf = deepcopy(A))
    y_type == :e && bpf_data!(A;bpf=get_bpf(;fs=fs))

    # load data
    (x,no_norm,features) = get_x(xyz,ind,features_setup;
                                 features_no_norm = features_no_norm,
                                 terms            = terms,
                                 sub_diurnal      = sub_diurnal,
                                 sub_igrf         = sub_igrf,
                                 bpf_mag          = bpf_mag)

    y = get_y(xyz,ind,map_val;
              y_type      = y_type,
              use_mag     = use_mag,
              sub_diurnal = sub_diurnal,
              sub_igrf    = sub_igrf)

    model_type in [:TL,:mod_TL] && (y_no_bpf = get_y(xyz,ind,map_val;
                                                     y_type      = :d,
                                                     use_mag     = use_mag,
                                                     sub_diurnal = sub_diurnal,
                                                     sub_igrf    = sub_igrf))

    for (xyz,ind) in zip(xyz_vec[2:end],ind_vec[2:end])

        # map values along trajectory (if needed)
        map_val = y_type in [:b,:c] ? get_map_val(mapS,xyz.traj,ind;α=200) : -1

        if model_type == :mod_TL
            A_ = create_TL_A(getfield(xyz,use_vec),ind;
                             Bt=getfield(xyz,use_mag),terms=terms_A)
        elseif model_type == :map_TL
            A_ = create_TL_A(getfield(xyz,use_vec),ind;
                             Bt=map_val,terms=terms_A)
        else
            (A_,Bt_,B_dot_) = create_TL_A(getfield(xyz,use_vec),ind;
                                          terms=terms_A,return_B=true)
            Bt    = vcat(Bt,Bt_)
            B_dot = vcat(B_dot,B_dot_)
        end

        fs = 1 / xyz.traj.dt
        model_type in [:TL,:mod_TL] && (A_no_bpf = vcat(A_no_bpf,A_))
        y_type == :e && bpf_data!(A_;bpf=get_bpf(;fs=fs))
        A = vcat(A,A_)

        x = vcat(x,get_x(xyz,ind,features_setup;
                         features_no_norm = features_no_norm,
                         terms            = terms,
                         sub_diurnal      = sub_diurnal,
                         sub_igrf         = sub_igrf,
                         bpf_mag          = bpf_mag)[1])

        y = vcat(y,get_y(xyz,ind,map_val;
                         y_type      = y_type,
                         use_mag     = use_mag,
                         sub_diurnal = sub_diurnal,
                         sub_igrf    = sub_igrf))

        model_type in [:TL,:mod_TL] && (y_no_bpf = vcat(y_no_bpf,get_y(xyz,ind,map_val;
                                                                       y_type      = :d,
                                                                       use_mag     = use_mag,
                                                                       sub_diurnal = sub_diurnal,
                                                                       sub_igrf    = sub_igrf)))

    end

    y_hat = zero(y) # initialize
    err   = 10*y    # initialize

    # set held-out test line
    A_test     = Matrix{eltype(A)}(undef,0,0)
    Bt_test    = Vector{eltype(x)}(undef,0)
    B_dot_test = Matrix{eltype(x)}(undef,0,0)
    x_test     = Matrix{eltype(x)}(undef,0,0)
    y_test     = Vector{eltype(y)}(undef,0)
    if !isempty(ind_test)
        (x_test,_,_) = get_x(xyz_test,ind_test,features_setup;
                             features_no_norm = features_no_norm,
                             terms            = terms,
                             sub_diurnal      = sub_diurnal,
                             sub_igrf         = sub_igrf,
                             bpf_mag          = bpf_mag)
        y_test = get_y(xyz_test,ind_test,map_val;
                       y_type      = y_type,
                       use_mag     = use_mag,
                       sub_diurnal = sub_diurnal,
                       sub_igrf    = sub_igrf)
        (A_test,Bt_test,B_dot_test) =
            create_TL_A(getfield(xyz_test,use_vec),ind_test;
                        terms=terms_A,return_B=true)
    end

    if drop_fi

        drop_fi_bson = remove_extension(drop_fi_bson,".bson")

        for i in axes(x,2)

            x_fi = x[:, axes(x,2) .!= i]
            x_test_fi = isempty(x_test) ? x_test : x_test[:,axes(x_test,2) .!=i]

            # train model
            if model_type in [:m1]
                (model,data_norms,y_hat_fi,err_fi) =
                    nn_comp_1_train(x_fi,y,no_norm;
                                    norm_type_x = norm_type_x,
                                    norm_type_y = norm_type_y,
                                    η_adam      = η_adam,
                                    epoch_adam  = epoch_adam,
                                    epoch_lbfgs = epoch_lbfgs,
                                    hidden      = hidden,
                                    activation  = activation,
                                    batchsize   = batchsize,
                                    frac_train  = frac_train,
                                    α_sgl       = α_sgl,
                                    λ_sgl       = λ_sgl,
                                    k_pca       = k_pca,
                                    x_test      = x_test_fi,
                                    y_test      = y_test,
                                    silent      = silent)
            elseif model_type in [:m2a,:m2b,:m2c,:m2d]
                (model,TL_coef,data_norms,y_hat_fi,err_fi) =
                    nn_comp_2_train(A,x_fi,y,no_norm;
                                    model_type  = model_type,
                                    norm_type_A = norm_type_A,
                                    norm_type_x = norm_type_x,
                                    norm_type_y = norm_type_y,
                                    TL_coef     = TL_coef,
                                    η_adam      = η_adam,
                                    epoch_adam  = epoch_adam,
                                    epoch_lbfgs = epoch_lbfgs,
                                    hidden      = hidden,
                                    activation  = activation,
                                    batchsize   = batchsize,
                                    frac_train  = frac_train,
                                    α_sgl       = α_sgl,
                                    λ_sgl       = λ_sgl,
                                    k_pca       = k_pca,
                                    A_test      = A_test,
                                    x_test      = x_test_fi,
                                    y_test      = y_test,
                                    silent      = silent)
            elseif model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc,:m3w,:m3tf]
                (model,TL_coef,data_norms,y_hat_fi,err_fi) =
                    nn_comp_3_train(A,Bt,B_dot,x_fi,y,no_norm;
                                    model_type    = model_type,
                                    norm_type_x   = norm_type_x,
                                    norm_type_y   = norm_type_y,
                                    TL_coef       = TL_coef,
                                    terms_A       = terms_A,
                                    y_type        = y_type,
                                    η_adam        = η_adam,
                                    epoch_adam    = epoch_adam,
                                    epoch_lbfgs   = epoch_lbfgs,
                                    hidden        = hidden,
                                    activation    = activation,
                                    batchsize     = batchsize,
                                    frac_train    = frac_train,
                                    α_sgl         = α_sgl,
                                    λ_sgl         = λ_sgl,
                                    k_pca         = k_pca,
                                    σ_curriculum  = σ_curriculum,
                                    l_window      = l_window,
                                    window_type   = window_type,
                                    tf_layer_type = tf_layer_type,
                                    tf_norm_type  = tf_norm_type,
                                    dropout_prob  = dropout_prob,
                                    N_tf_head     = N_tf_head,
                                    tf_gain       = tf_gain,
                                    A_test        = A_test,
                                    Bt_test       = Bt_test,
                                    B_dot_test    = B_dot_test,
                                    x_test        = x_test_fi,
                                    y_test        = y_test,
                                    silent        = silent)
            else
                error("$model_type model type not defined")
            end

            if std(err_fi) < std(err)
                y_hat = y_hat_fi
                err   = err_fi
            end

            comp_params = NNCompParams(comp_params,
                                       data_norms = data_norms,
                                       model      = model,
                                       terms_A    = terms_A,
                                       TL_coef    = TL_coef)
            save_comp_params(comp_params,drop_fi_bson*"_$i.bson")

        end

    else

        # train model
        if model_type in [:m1]
            (model,data_norms,y_hat,err) =
                nn_comp_1_train(x,y,no_norm;
                                norm_type_x = norm_type_x,
                                norm_type_y = norm_type_y,
                                η_adam      = η_adam,
                                epoch_adam  = epoch_adam,
                                epoch_lbfgs = epoch_lbfgs,
                                hidden      = hidden,
                                activation  = activation,
                                batchsize   = batchsize,
                                frac_train  = frac_train,
                                α_sgl       = α_sgl,
                                λ_sgl       = λ_sgl,
                                k_pca       = k_pca,
                                data_norms  = data_norms,
                                model       = model,
                                x_test      = x_test,
                                y_test      = y_test,
                                silent      = silent)
        elseif model_type in [:m2a,:m2b,:m2c,:m2d]
            (model,TL_coef,data_norms,y_hat,err) =
                nn_comp_2_train(A,x,y,no_norm;
                                model_type  = model_type,
                                norm_type_A = norm_type_A,
                                norm_type_x = norm_type_x,
                                norm_type_y = norm_type_y,
                                TL_coef     = TL_coef,
                                η_adam      = η_adam,
                                epoch_adam  = epoch_adam,
                                epoch_lbfgs = epoch_lbfgs,
                                hidden      = hidden,
                                activation  = activation,
                                batchsize   = batchsize,
                                frac_train  = frac_train,
                                α_sgl       = α_sgl,
                                λ_sgl       = λ_sgl,
                                k_pca       = k_pca,
                                data_norms  = data_norms,
                                model       = model,
                                A_test      = A_test,
                                x_test      = x_test,
                                y_test      = y_test,
                                silent      = silent)
        elseif model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc,:m3w,:m3tf]
            (model,TL_coef,data_norms,y_hat,err) =
                nn_comp_3_train(A,Bt,B_dot,x,y,no_norm;
                                model_type    = model_type,
                                norm_type_x   = norm_type_x,
                                norm_type_y   = norm_type_y,
                                TL_coef       = TL_coef,
                                terms_A       = terms_A,
                                y_type        = y_type,
                                η_adam        = η_adam,
                                epoch_adam    = epoch_adam,
                                epoch_lbfgs   = epoch_lbfgs,
                                hidden        = hidden,
                                activation    = activation,
                                batchsize     = batchsize,
                                frac_train    = frac_train,
                                α_sgl         = α_sgl,
                                λ_sgl         = λ_sgl,
                                k_pca         = k_pca,
                                σ_curriculum  = σ_curriculum,
                                l_window      = l_window,
                                window_type   = window_type,
                                tf_layer_type = tf_layer_type,
                                tf_norm_type  = tf_norm_type,
                                dropout_prob  = dropout_prob,
                                N_tf_head     = N_tf_head,
                                tf_gain       = tf_gain,
                                A_test        = A_test,
                                Bt_test       = Bt_test,
                                B_dot_test    = B_dot_test,
                                x_test        = x_test,
                                y_test        = y_test,
                                silent        = silent)
        elseif model_type in [:TL,:mod_TL,:map_TL]
            trim = model_type in [:TL,:mod_TL] ? 20 : 0
            (model,data_norms,y_hat,err) =
                linear_fit(A,y;
                           trim=trim, λ=λ_TL,
                           norm_type_x = norm_type_A,
                           norm_type_y = norm_type_y,
                           silent      = silent)
            if model_type in [:TL,:mod_TL]
                (y_hat,err) = linear_test(A_no_bpf,y_no_bpf,data_norms,model;
                                          silent=silent)
            end
        elseif model_type in [:elasticnet]
            (model,data_norms,y_hat,err) =
                elasticnet_fit(x,y; silent=silent)
        elseif model_type in [:plsr]
            (model,data_norms,y_hat,err) =
                plsr_fit(x,y,k_plsr; silent=silent)
        else
            error("$model_type model type not defined")
        end

    end

    if typeof(comp_params) <: NNCompParams
        comp_params = NNCompParams(comp_params,
                                   data_norms = data_norms,
                                   model      = model,
                                   terms_A    = terms_A,
                                   TL_coef    = TL_coef)
    elseif typeof(comp_params) <: LinCompParams
        comp_params = LinCompParams(comp_params,
                                    data_norms = data_norms,
                                    model      = model)
    end

    silent || print_time(time()-t0,1)

    return (comp_params, y, y_hat, err, features)
end # function comp_train

function comp_train(xyz_vec::Vector{XYZ20{Int64,Float64}},
                    ind_vec::Vector{BitVector},
                    mapS::Union{MapS,MapSd,MapS3D} = mapS_null;
                    comp_params::CompParams = NNCompParams(),
                    xyz_test::XYZ           = xyz_vec[1],
                    ind_test                = BitVector(),
                    σ_curriculum            = 1.0,
                    l_window::Int           = 5,
                    window_type::Symbol     = :sliding,
                    tf_layer_type::Symbol   = :postlayer,
                    tf_norm_type::Symbol    = :batch,
                    dropout_prob            = 0.2,
                    N_tf_head::Int          = 8,
                    tf_gain                 = 1.0,
                    silent::Bool            = false)
    @warn("this version of comp_train() is deprecated & will be removed in MagNav.jl v1.2.0, see docstring for argument order")
    comp_train(comp_params,xyz_vec,ind_vec,mapS;
               xyz_test      = xyz_test,
               ind_test      = ind_test,
               σ_curriculum  = σ_curriculum,
               l_window      = l_window,
               window_type   = window_type,
               tf_layer_type = tf_layer_type,
               tf_norm_type  = tf_norm_type,
               dropout_prob  = dropout_prob,
               N_tf_head     = N_tf_head,
               tf_gain       = tf_gain,
               silent        = silent)
end # function comp_train

"""
    comp_train(comp_params::CompParams, lines,
               df_line::DataFrame, df_flight::DataFrame, df_map::DataFrame;
               σ_curriculum          = 1.0,
               l_window::Int         = 5,
               window_type::Symbol   = :sliding,
               tf_layer_type::Symbol = :postlayer,
               tf_norm_type::Symbol  = :batch,
               dropout_prob          = 0.2,
               N_tf_head::Int        = 8,
               tf_gain               = 1.0,
               silent::Bool          = false)

Train an aeromagnetic compensation model.

**Arguments:**
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct, either:
    - `NNCompParams`:  neural network-based aeromagnetic compensation parameters struct
    - `LinCompParams`: linear aeromagnetic compensation parameters struct
- `lines`:   selected line number(s)
- `df_line`: lookup table (DataFrame) of `lines`
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
- `df_flight`: lookup table (DataFrame) of flight data HDF5 files
|**Field**|**Type**|**Description**
|:--|:--|:--
`flight`  |`Symbol`| flight name (e.g., `:Flt1001`)
`xyz_type`|`Symbol`| subtype of `XYZ` to use for flight data {`:XYZ0`,`:XYZ1`,`:XYZ20`,`:XYZ21`}
`xyz_set` |`Real`  | flight dataset number (used to prevent inproper mixing of datasets, such as different magnetometer locations)
`xyz_h5`  |`String`| path/name of flight data HDF5 file (`.h5` extension optional)
- `df_map`: lookup table (DataFrame) of map data HDF5 files
|**Field**|**Type**|**Description**
|:--|:--|:--
`map_name`|`Symbol`| name of magnetic anomaly map
`map_h5`  |`String`| path/name of map data HDF5 file (`.h5` extension optional)
`map_type`|`Symbol`| (optional) type of magnetic anomaly map {`drape`,`HAE`}
`map_alt` |`Real`  | (optional) map altitude, -1 for drape map [m]
- `σ_curriculum`:  (optional) standard deviation threshold, only used for `model_type = :m3sc, :m3vc`
- `l_window`:      (optional) temporal window length, only used for `model_type = :m3w, :m3tf`
- `window_type`:   (optional) type of windowing, `:sliding` for overlapping or `:contiguous` for non-overlapping, only used for `model_type = :m3w, :m3tf`
- `tf_layer_type`: (optional) transformer normalization layer before or after skip connection {`:prelayer`,`:postlayer`}, only used for `model_type = :m3tf`
- `tf_norm_type`:  (optional) normalization for transformer encoder {`:batch`,`:layer`,`:none`}, only used for `model_type = :m3tf`
- `dropout_prob`:  (optional) dropout rate, only used for `model_type = :m3w, :m3tf`
- `N_tf_head`:     (optional) number of attention heads, only used for `model_type = :m3tf`
- `tf_gain`:       (optional) weight initialization parameter, only used for `model_type = :m3tf`
- `silent`:        (optional) if true, no print outs

**Returns:**
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct
- `y`:           length `N` target vector
- `y_hat`:       length `N` prediction vector
- `err`:         length `N` mean-corrected (per line) compensation error
- `features`:    length `Nf` feature vector (including components of TL `A`, etc.)
"""
function comp_train(comp_params::CompParams, lines,
                    df_line::DataFrame, df_flight::DataFrame, df_map::DataFrame;
                    σ_curriculum          = 1.0,
                    l_window::Int         = 5,
                    window_type::Symbol   = :sliding,
                    tf_layer_type::Symbol = :postlayer,
                    tf_norm_type::Symbol  = :batch,
                    dropout_prob          = 0.2,
                    N_tf_head::Int        = 8,
                    tf_gain               = 1.0,
                    silent::Bool          = false)

    seed!(2) # for reproducibility
    t0 = time()

    # unpack parameters
    if typeof(comp_params) <: NNCompParams
        @unpack version, features_setup, features_no_norm, model_type, y_type,
        use_mag, use_vec, data_norms, model, terms, terms_A, sub_diurnal,
        sub_igrf, bpf_mag, reorient_vec, norm_type_A, norm_type_x, norm_type_y,
        TL_coef, η_adam, epoch_adam, epoch_lbfgs, hidden, activation,
        batchsize, frac_train, α_sgl, λ_sgl, k_pca,
        drop_fi, drop_fi_bson, drop_fi_csv, perm_fi, perm_fi_csv = comp_params
    elseif typeof(comp_params) <: LinCompParams
        @unpack version, features_setup, features_no_norm, model_type, y_type,
        use_mag, use_vec, data_norms, model, terms, terms_A, sub_diurnal,
        sub_igrf, bpf_mag, reorient_vec, norm_type_A, norm_type_x, norm_type_y,
        k_plsr, λ_TL = comp_params
        drop_fi = false
        perm_fi = false
    end

    if (model_type in [:TL,:mod_TL]) & (y_type != :e)
        silent || @info("forcing y_type $(y_type) => :e (BPF'd total field)")
        y_type = :e
    end
    if (model_type in [:map_TL]) & (y_type != :c)
        silent || @info("forcing y_type $(y_type) => :c (aircraft field #1, using map)")
        y_type = :c
    end
    if model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc,:m3w,:m3tf]
        for term in terms_A
            if term in [:fdm,:f,:fdm3,:f3,:bias,:b]
                silent || @info("removing $term term from terms_A")
                A_test = create_TL_A([1.0],[1.0],[1.0];terms=terms_A)
                if length(A_test) == length(TL_coef)
                    ind_term = get_TL_term_ind(term,terms_A)
                    silent || @info("removing $(sum(ind_term)) value(s) from TL_coef")
                    TL_coef = TL_coef[.!ind_term]
                end
                terms_A = terms_A[terms_A .!= term]
            end
        end
    end

    mod_TL = model_type == :mod_TL ? true : false
    map_TL = model_type == :map_TL ? true : false

    # load data
    if model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc,:m3w,:m3tf]
        (A,Bt,B_dot,x,y,no_norm,features,l_segs) = get_Axy(lines,df_line,df_flight,df_map,
                                                           features_setup;
                                                           features_no_norm = features_no_norm,
                                                           y_type           = y_type,
                                                           use_mag          = use_mag,
                                                           use_vec          = use_vec,
                                                           terms            = terms,
                                                           terms_A          = terms_A,
                                                           sub_diurnal      = sub_diurnal,
                                                           sub_igrf         = sub_igrf,
                                                           bpf_mag          = bpf_mag,
                                                           reorient_vec     = reorient_vec,
                                                           mod_TL           = mod_TL,
                                                           map_TL           = map_TL,
                                                           return_B         = true,
                                                           silent           = silent_debug)
    else
        (A,x,y,no_norm,features,l_segs) = get_Axy(lines,df_line,df_flight,df_map,
                                                  features_setup;
                                                  features_no_norm = features_no_norm,
                                                  y_type           = y_type,
                                                  use_mag          = use_mag,
                                                  use_vec          = use_vec,
                                                  terms            = terms,
                                                  terms_A          = terms_A,
                                                  sub_diurnal      = sub_diurnal,
                                                  sub_igrf         = sub_igrf,
                                                  bpf_mag          = bpf_mag,
                                                  reorient_vec     = reorient_vec,
                                                  mod_TL           = mod_TL,
                                                  map_TL           = map_TL,
                                                  return_B         = false,
                                                  silent           = silent_debug)
    end

    y_hat = zero(y) # initialize
    err   = 10*y    # initialize

    if drop_fi

        drop_fi_bson = remove_extension(drop_fi_bson,".bson")

        for i in axes(x,2)

            x_fi = x[:, axes(x,2) .!= i]

            # train model
            if model_type in [:m1]
                (model,data_norms,y_hat_fi,err_fi) =
                    nn_comp_1_train(x_fi,y,no_norm;
                                    norm_type_x = norm_type_x,
                                    norm_type_y = norm_type_y,
                                    η_adam      = η_adam,
                                    epoch_adam  = epoch_adam,
                                    epoch_lbfgs = epoch_lbfgs,
                                    hidden      = hidden,
                                    activation  = activation,
                                    batchsize   = batchsize,
                                    frac_train  = frac_train,
                                    α_sgl       = α_sgl,
                                    λ_sgl       = λ_sgl,
                                    k_pca       = k_pca,
                                    l_segs      = l_segs,
                                    silent      = silent)
            elseif model_type in [:m2a,:m2b,:m2c,:m2d]
                (model,TL_coef,data_norms,y_hat_fi,err_fi) =
                    nn_comp_2_train(A,x_fi,y,no_norm;
                                    model_type  = model_type,
                                    norm_type_A = norm_type_A,
                                    norm_type_x = norm_type_x,
                                    norm_type_y = norm_type_y,
                                    TL_coef     = TL_coef,
                                    η_adam      = η_adam,
                                    epoch_adam  = epoch_adam,
                                    epoch_lbfgs = epoch_lbfgs,
                                    hidden      = hidden,
                                    activation  = activation,
                                    batchsize   = batchsize,
                                    frac_train  = frac_train,
                                    α_sgl       = α_sgl,
                                    λ_sgl       = λ_sgl,
                                    k_pca       = k_pca,
                                    l_segs      = l_segs,
                                    silent      = silent)
            elseif model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc,:m3w,:m3tf]
                (model,TL_coef,data_norms,y_hat_fi,err_fi) =
                    nn_comp_3_train(A,Bt,B_dot,x_fi,y,no_norm;
                                    model_type    = model_type,
                                    norm_type_x   = norm_type_x,
                                    norm_type_y   = norm_type_y,
                                    TL_coef       = TL_coef,
                                    terms_A       = terms_A,
                                    y_type        = y_type,
                                    η_adam        = η_adam,
                                    epoch_adam    = epoch_adam,
                                    epoch_lbfgs   = epoch_lbfgs,
                                    hidden        = hidden,
                                    activation    = activation,
                                    batchsize     = batchsize,
                                    frac_train    = frac_train,
                                    α_sgl         = α_sgl,
                                    λ_sgl         = λ_sgl,
                                    k_pca         = k_pca,
                                    σ_curriculum  = σ_curriculum,
                                    l_window      = l_window,
                                    window_type   = window_type,
                                    tf_layer_type = tf_layer_type,
                                    tf_norm_type  = tf_norm_type,
                                    dropout_prob  = dropout_prob,
                                    N_tf_head     = N_tf_head,
                                    tf_gain       = tf_gain,
                                    l_segs        = l_segs,
                                    silent        = silent)
            else
                error("$model_type model type not defined")
            end

            if std(err_fi) < std(err)
                y_hat = y_hat_fi
                err   = err_fi
            end

            comp_params = NNCompParams(comp_params,
                                       data_norms = data_norms,
                                       model      = model,
                                       terms_A    = terms_A,
                                       TL_coef    = TL_coef)
            save_comp_params(comp_params,drop_fi_bson*"_$i.bson")

        end

    else

        # train model
        if model_type in [:m1]
            (model,data_norms,y_hat,err) =
                nn_comp_1_train(x,y,no_norm;
                                norm_type_x = norm_type_x,
                                norm_type_y = norm_type_y,
                                η_adam      = η_adam,
                                epoch_adam  = epoch_adam,
                                epoch_lbfgs = epoch_lbfgs,
                                hidden      = hidden,
                                activation  = activation,
                                batchsize   = batchsize,
                                frac_train  = frac_train,
                                α_sgl       = α_sgl,
                                λ_sgl       = λ_sgl,
                                k_pca       = k_pca,
                                data_norms  = data_norms,
                                model       = model,
                                l_segs      = l_segs,
                                silent      = silent)
        elseif model_type in [:m2a,:m2b,:m2c,:m2d]
            (model,TL_coef,data_norms,y_hat,err) =
                nn_comp_2_train(A,x,y,no_norm;
                                model_type  = model_type,
                                norm_type_A = norm_type_A,
                                norm_type_x = norm_type_x,
                                norm_type_y = norm_type_y,
                                TL_coef     = TL_coef,
                                η_adam      = η_adam,
                                epoch_adam  = epoch_adam,
                                epoch_lbfgs = epoch_lbfgs,
                                hidden      = hidden,
                                activation  = activation,
                                batchsize   = batchsize,
                                frac_train  = frac_train,
                                α_sgl       = α_sgl,
                                λ_sgl       = λ_sgl,
                                k_pca       = k_pca,
                                data_norms  = data_norms,
                                model       = model,
                                l_segs      = l_segs,
                                silent      = silent)
        elseif model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc,:m3w,:m3tf]
            (model,TL_coef,data_norms,y_hat,err) =
                nn_comp_3_train(A,Bt,B_dot,x,y,no_norm;
                                model_type    = model_type,
                                norm_type_x   = norm_type_x,
                                norm_type_y   = norm_type_y,
                                TL_coef       = TL_coef,
                                terms_A       = terms_A,
                                y_type        = y_type,
                                η_adam        = η_adam,
                                epoch_adam    = epoch_adam,
                                epoch_lbfgs   = epoch_lbfgs,
                                hidden        = hidden,
                                activation    = activation,
                                batchsize     = batchsize,
                                frac_train    = frac_train,
                                α_sgl         = α_sgl,
                                λ_sgl         = λ_sgl,
                                k_pca         = k_pca,
                                σ_curriculum  = σ_curriculum,
                                l_window      = l_window,
                                window_type   = window_type,
                                tf_layer_type = tf_layer_type,
                                tf_norm_type  = tf_norm_type,
                                dropout_prob  = dropout_prob,
                                N_tf_head     = N_tf_head,
                                tf_gain       = tf_gain,
                                data_norms    = data_norms,
                                model         = model,
                                l_segs        = l_segs,
                                silent        = silent)
        elseif model_type in [:TL,:mod_TL,:map_TL]
            trim = model_type in [:TL,:mod_TL] ? 20 : 0
            (model,data_norms,y_hat,err) =
                linear_fit(A,y;
                           trim=trim, λ=λ_TL,
                           norm_type_x = norm_type_A,
                           norm_type_y = norm_type_y,
                           l_segs      = l_segs,
                           silent      = silent)
            if model_type in [:TL,:mod_TL]
                (A_no_bpf,_,y_no_bpf,_,_,_) =
                    get_Axy(lines,df_line,df_flight,df_map,
                            features_setup;
                            features_no_norm = features_no_norm,
                            y_type           = :d,
                            use_mag          = use_mag,
                            use_vec          = use_vec,
                            terms            = terms,
                            terms_A          = terms_A,
                            sub_diurnal      = sub_diurnal,
                            sub_igrf         = sub_igrf,
                            bpf_mag          = bpf_mag,
                            reorient_vec     = reorient_vec,
                            mod_TL           = mod_TL,
                            map_TL           = map_TL,
                            return_B         = false,
                            silent           = silent_debug)
                (y_hat,err) = linear_test(A_no_bpf,y_no_bpf,data_norms,model;
                                          silent=silent)
            end
        elseif model_type in [:elasticnet]
            (model,data_norms,y_hat,err) = elasticnet_fit(x,y;
                                                          l_segs = l_segs,
                                                          silent = silent)
        elseif model_type in [:plsr]
            (model,data_norms,y_hat,err) = plsr_fit(x,y,k_plsr;
                                                    l_segs = l_segs,
                                                    silent = silent)
        else
            error("$model_type model type not defined")
        end

    end

    if typeof(comp_params) <: NNCompParams
        comp_params = NNCompParams(comp_params,
                                   data_norms = data_norms,
                                   model      = model,
                                   terms_A    = terms_A,
                                   TL_coef    = TL_coef)
    elseif typeof(comp_params) <: LinCompParams
        comp_params = LinCompParams(comp_params,
                                    data_norms = data_norms,
                                    model      = model)
    end

    silent || print_time(time()-t0,1)

    return (comp_params, y, y_hat, err, features)
end # function comp_train

function comp_train(lines, df_line::DataFrame, df_flight::DataFrame,
                    df_map::DataFrame, comp_params::CompParams=NNCompParams();
                    σ_curriculum            = 1.0,
                    l_window::Int           = 5,
                    window_type::Symbol     = :sliding,
                    tf_layer_type::Symbol   = :postlayer,
                    tf_norm_type::Symbol    = :batch,
                    dropout_prob            = 0.2,
                    N_tf_head::Int          = 8,
                    tf_gain                 = 1.0,
                    silent::Bool            = false)
    @warn("this version of comp_train() is deprecated & will be removed in MagNav.jl v1.2.0, see docstring for argument order")
    comp_train(comp_params,lines,df_line,df_flight,df_map;
               σ_curriculum  = σ_curriculum,
               l_window      = l_window,
               window_type   = window_type,
               tf_layer_type = tf_layer_type,
               tf_norm_type  = tf_norm_type,
               dropout_prob  = dropout_prob,
               N_tf_head     = N_tf_head,
               tf_gain       = tf_gain,
               silent        = silent)
end # function comp_train

"""
    comp_test(comp_params::CompParams, xyz::XYZ, ind,
              mapS::Union{MapS,MapSd,MapS3D} = mapS_null;
              l_window::Int = 5,
              silent::Bool  = false)

Evaluate performance of an aeromagnetic compensation model.

**Arguments:**
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct, either:
    - `NNCompParams`:  neural network-based aeromagnetic compensation parameters struct
    - `LinCompParams`: linear aeromagnetic compensation parameters struct
- `xyz`:      `XYZ` flight data struct
- `ind`:      selected data indices
- `mapS`:     (optional) `MapS`, `MapSd`, or `MapS3D` scalar magnetic anomaly map struct, only used for `y_type = :b, :c`
- `l_window`: (optional) temporal window length, only used for `model_type = :m3w, :m3tf`
- `silent`:   (optional) if true, no print outs

**Returns:**
- `y`:        length `N` target vector
- `y_hat`:    length `N` prediction vector
- `err`:      length `N` compensation error
- `features`: length `Nf` feature vector (including components of TL `A`, etc.)
"""
function comp_test(comp_params::CompParams, xyz::XYZ, ind,
                   mapS::Union{MapS,MapSd,MapS3D} = mapS_null;
                   l_window::Int = 5,
                   silent::Bool  = false)

    seed!(2) # for reproducibility
    t0 = time()

    # unpack parameters
    if typeof(comp_params) <: NNCompParams
        @unpack version, features_setup, features_no_norm, model_type, y_type,
        use_mag, use_vec, data_norms, model, terms, terms_A, sub_diurnal,
        sub_igrf, bpf_mag, reorient_vec, norm_type_A, norm_type_x, norm_type_y,
        TL_coef, η_adam, epoch_adam, epoch_lbfgs, hidden, activation,
        batchsize, frac_train, α_sgl, λ_sgl, k_pca,
        drop_fi, drop_fi_bson, drop_fi_csv, perm_fi, perm_fi_csv = comp_params
    elseif typeof(comp_params) <: LinCompParams
        @unpack version, features_setup, features_no_norm, model_type, y_type,
        use_mag, use_vec, data_norms, model, terms, terms_A, sub_diurnal,
        sub_igrf, bpf_mag, reorient_vec, norm_type_A, norm_type_x, norm_type_y,
        k_plsr, λ_TL = comp_params
        drop_fi = false
        perm_fi = false
    end

    if ((model_type in [:TL,:mod_TL]) & (y_type != :d)) | (y_type == :e)
        silent || @info("forcing y_type $(y_type) => :d (Δmag)")
        y_type = :d
    end
    if (model_type in [:map_TL]) & (y_type != :c)
        silent || @info("forcing y_type $(y_type) => :c (aircraft field #1, using map)")
        y_type = :c
    end
    if model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc,:m3w,:m3tf]
        for term in terms_A
            if term in [:fdm,:f,:fdm3,:f3,:bias,:b]
                silent || @info("ignoring $term term in terms_A")
                A_test = create_TL_A([1.0],[1.0],[1.0];terms=terms_A)
                if length(A_test) == length(TL_coef)
                    ind_term = get_TL_term_ind(term,terms_A)
                    silent || @info("ignoring $(sum(ind_term)) value(s) in TL_coef")
                    TL_coef = TL_coef[.!ind_term]
                end
                terms_A = terms_A[terms_A .!= term]
            end
        end
    end

    # map values along trajectory (if needed)
    map_val = y_type in [:b,:c] ? get_map_val(mapS,xyz.traj,ind;α=200) : -1

    # `A` matrix for selected vector magnetometer
    field_check(xyz,use_vec,MagV)
    if model_type == :mod_TL
        A = create_TL_A(getfield(xyz,use_vec),ind;
                        Bt=getfield(xyz,use_mag),terms=terms_A)
    elseif model_type == :map_TL
        A = create_TL_A(getfield(xyz,use_vec),ind;
                        Bt=map_val,terms=terms_A)
    else
        (A,Bt,B_dot) = create_TL_A(getfield(xyz,use_vec),ind;
                                   terms=terms_A,return_B=true)
    end
    fs = 1 / xyz.traj.dt
    y_type == :e && bpf_data!(A;bpf=get_bpf(;fs=fs))

    # load data
    (x,_,features) = get_x(xyz,ind,features_setup;
                           features_no_norm = features_no_norm,
                           terms            = terms,
                           sub_diurnal      = sub_diurnal,
                           sub_igrf         = sub_igrf,
                           bpf_mag          = bpf_mag)

    y = get_y(xyz,ind,map_val;
              y_type      = y_type,
              use_mag     = use_mag,
              sub_diurnal = sub_diurnal,
              sub_igrf    = sub_igrf)

    y_hat = zero(y) # initialize
    err   = 10*y    # initialize

    if drop_fi | perm_fi

        drop_fi_bson = remove_extension(drop_fi_bson,".bson")
        drop_fi_csv  = add_extension(drop_fi_csv,".csv")
        perm_fi_csv  = add_extension(perm_fi_csv,".csv")

        for i in axes(x,2)

            if perm_fi
                x_fi = deepcopy(x)
                x_fi[:,i] .= x[randperm(size(x,1)),i]
                fi_csv = perm_fi_csv
            elseif drop_fi
                x_fi = x[:, axes(x,2) .!= i]
                comp_params = get_comp_params(drop_fi_bson*"_$i.bson",silent_debug)
                data_norms  = comp_params.data_norms
                model       = comp_params.model
                fi_csv      = drop_fi_csv
            end

            # evaluate model
            if model_type in [:m1]
                (y_hat_fi,err_fi) = nn_comp_1_test(x_fi,y,data_norms,model;
                                                   silent     = silent)
            elseif model_type in [:m2a,:m2b,:m2c,:m2d]
                (y_hat_fi,err_fi) = nn_comp_2_test(A,x_fi,y,data_norms,model;
                                                   model_type = model_type,
                                                   TL_coef    = TL_coef,
                                                   silent     = silent)
            elseif model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc,:m3w,:m3tf]
                (y_hat_fi,err_fi) = nn_comp_3_test(A,Bt,B_dot,x_fi,y,data_norms,model;
                                                   model_type = model_type,
                                                   y_type     = y_type,
                                                   TL_coef    = TL_coef,
                                                   terms_A    = terms_A,
                                                   l_window   = l_window,
                                                   silent     = silent)
            else
                error("$model_type model type not defined")
            end

            if std(err_fi) < std(err)
                y_hat = y_hat_fi
                err   = err_fi
            end

            open(fi_csv,"a") do file
                writedlm(file,zip(i,std(err_fi)),',')
            end

        end

        silent || @info("returning best feature importance results")

    else

        # evaluate model
        if model_type in [:m1]
            (y_hat,err) = nn_comp_1_test(x,y,data_norms,model;
                                         silent     = silent)
        elseif model_type in [:m2a,:m2b,:m2c,:m2d]
            (y_hat,err) = nn_comp_2_test(A,x,y,data_norms,model;
                                         model_type = model_type,
                                         TL_coef    = TL_coef,
                                         silent     = silent)
        elseif model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc,:m3w,:m3tf]
            (y_hat,err) = nn_comp_3_test(A,Bt,B_dot,x,y,data_norms,model;
                                         model_type = model_type,
                                         y_type     = y_type,
                                         TL_coef    = TL_coef,
                                         terms_A    = terms_A,
                                         l_window   = l_window,
                                         silent     = silent)
        elseif model_type in [:TL,:mod_TL,:map_TL]
            (y_hat,err) = linear_test(A,y,data_norms,model;
                                         silent     = silent)
        elseif model_type in [:elasticnet,:plsr]
            (y_hat,err) = linear_test(x,y,data_norms,model;
                                         silent     = silent)
        else
            error("$model_type model type not defined")
        end

    end

    silent || print_time(time()-t0,1)

    return (y, y_hat, err, features)
end # function comp_test

function comp_test(xyz::XYZ, ind, mapS::Union{MapS,MapSd,MapS3D} = mapS_null;
                   comp_params::CompParams = NNCompParams(),
                   l_window::Int           = 5,
                   silent::Bool            = false)
    @warn("this version of comp_test() is deprecated & will be removed in MagNav.jl v1.2.0, see docstring for argument order")
    comp_test(comp_params,xyz,ind,mapS;
              l_window = l_window,
              silent   = silent)
end # function comp_test

"""
    comp_test(comp_params::CompParams, lines,
              df_line::DataFrame, df_flight::DataFrame, df_map::DataFrame;
              l_window::Int = 5,
              silent::Bool  = false)

Evaluate performance of an aeromagnetic compensation model.

**Arguments:**
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct, either:
    - `NNCompParams`:  neural network-based aeromagnetic compensation parameters struct
    - `LinCompParams`: linear aeromagnetic compensation parameters struct
- `lines`:   selected line number(s)
- `df_line`: lookup table (DataFrame) of `lines`
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
- `df_flight`: lookup table (DataFrame) of flight data HDF5 files
|**Field**|**Type**|**Description**
|:--|:--|:--
`flight`  |`Symbol`| flight name (e.g., `:Flt1001`)
`xyz_type`|`Symbol`| subtype of `XYZ` to use for flight data {`:XYZ0`,`:XYZ1`,`:XYZ20`,`:XYZ21`}
`xyz_set` |`Real`  | flight dataset number (used to prevent inproper mixing of datasets, such as different magnetometer locations)
`xyz_h5`  |`String`| path/name of flight data HDF5 file (`.h5` extension optional)
- `df_map`: lookup table (DataFrame) of map data HDF5 files
|**Field**|**Type**|**Description**
|:--|:--|:--
`map_name`|`Symbol`| name of magnetic anomaly map
`map_h5`  |`String`| path/name of map data HDF5 file (`.h5` extension optional)
`map_type`|`Symbol`| (optional) type of magnetic anomaly map {`drape`,`HAE`}
`map_alt` |`Real`  | (optional) map altitude, -1 for drape map [m]
- `l_window`: (optional) temporal window length, only used for `model_type = :m3w, :m3tf`
- `silent`:   (optional) if true, no print outs

**Returns:**
- `y`:        length `N` target vector
- `y_hat`:    length `N` prediction vector
- `err`:      length `N` mean-corrected (per line) compensation error
- `features`: length `Nf` feature vector (including components of TL `A`, etc.)
"""
function comp_test(comp_params::CompParams, lines,
                   df_line::DataFrame, df_flight::DataFrame, df_map::DataFrame;
                   l_window::Int = 5,
                   silent::Bool  = false)

    seed!(2) # for reproducibility
    t0 = time()

    # unpack parameters
    if typeof(comp_params) <: NNCompParams
        @unpack version, features_setup, features_no_norm, model_type, y_type,
        use_mag, use_vec, data_norms, model, terms, terms_A, sub_diurnal,
        sub_igrf, bpf_mag, reorient_vec, norm_type_A, norm_type_x, norm_type_y,
        TL_coef, η_adam, epoch_adam, epoch_lbfgs, hidden, activation,
        batchsize, frac_train, α_sgl, λ_sgl, k_pca,
        drop_fi, drop_fi_bson, drop_fi_csv, perm_fi, perm_fi_csv = comp_params
    elseif typeof(comp_params) <: LinCompParams
        @unpack version, features_setup, features_no_norm, model_type, y_type,
        use_mag, use_vec, data_norms, model, terms, terms_A, sub_diurnal,
        sub_igrf, bpf_mag, reorient_vec, norm_type_A, norm_type_x, norm_type_y,
        k_plsr, λ_TL = comp_params
        drop_fi = false
        perm_fi = false
    end

    if ((model_type in [:TL,:mod_TL]) & (y_type != :d)) | (y_type == :e)
        silent || @info("forcing y_type $(y_type) => :d (Δmag)")
        y_type = :d
    end
    if (model_type in [:map_TL]) & (y_type != :c)
        silent || @info("forcing y_type $(y_type) => :c (aircraft field #1, using map)")
        y_type = :c
    end
    if model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc,:m3w,:m3tf]
        for term in terms_A
            if term in [:fdm,:f,:fdm3,:f3,:bias,:b]
                silent || @info("ignoring $term term in terms_A")
                A_test = create_TL_A([1.0],[1.0],[1.0];terms=terms_A)
                if length(A_test) == length(TL_coef)
                    ind_term = get_TL_term_ind(term,terms_A)
                    silent || @info("ignoring $(sum(ind_term)) value(s) in TL_coef")
                    TL_coef = TL_coef[.!ind_term]
                end
                terms_A = terms_A[terms_A .!= term]
            end
        end
    end

    mod_TL = model_type == :mod_TL ? true : false
    map_TL = model_type == :map_TL ? true : false

    # load data
    if model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc,:m3w,:m3tf]
        (A,Bt,B_dot,x,y,_,features,l_segs) = get_Axy(lines,df_line,df_flight,df_map,
                                                     features_setup;
                                                     features_no_norm = features_no_norm,
                                                     y_type           = y_type,
                                                     use_mag          = use_mag,
                                                     use_vec          = use_vec,
                                                     terms            = terms,
                                                     terms_A          = terms_A,
                                                     sub_diurnal      = sub_diurnal,
                                                     sub_igrf         = sub_igrf,
                                                     bpf_mag          = bpf_mag,
                                                     reorient_vec     = reorient_vec,
                                                     mod_TL           = mod_TL,
                                                     map_TL           = map_TL,
                                                     return_B         = true,
                                                     silent           = silent_debug)
    else
        (A,x,y,_,features,l_segs) = get_Axy(lines,df_line,df_flight,df_map,
                                            features_setup;
                                            features_no_norm = features_no_norm,
                                            y_type           = y_type,
                                            use_mag          = use_mag,
                                            use_vec          = use_vec,
                                            terms            = terms,
                                            terms_A          = terms_A,
                                            sub_diurnal      = sub_diurnal,
                                            sub_igrf         = sub_igrf,
                                            bpf_mag          = bpf_mag,
                                            reorient_vec     = reorient_vec,
                                            mod_TL           = mod_TL,
                                            map_TL           = map_TL,
                                            return_B         = false,
                                            silent           = silent_debug)
    end

    y_hat = zero(y) # initialize
    err   = 10*y    # initialize

    if drop_fi | perm_fi

        drop_fi_bson = remove_extension(drop_fi_bson,".bson")
        drop_fi_csv  = add_extension(drop_fi_csv,".csv")
        perm_fi_csv  = add_extension(perm_fi_csv,".csv")

        for i in axes(x,2)

            if perm_fi
                x_fi = deepcopy(x)
                x_fi[:,i] .= x[randperm(size(x,1)),i]
                fi_csv = perm_fi_csv
            elseif drop_fi
                x_fi = x[:, axes(x,2) .!= i]
                comp_params = get_comp_params(drop_fi_bson*"_$i.bson",silent_debug)
                data_norms  = comp_params.data_norms
                model       = comp_params.model
                fi_csv      = drop_fi_csv
            end

            # evaluate model
            if model_type in [:m1]
                (y_hat_fi,err_fi) = nn_comp_1_test(x_fi,y,data_norms,model;
                                                   l_segs     = l_segs,
                                                   silent     = silent)
            elseif model_type in [:m2a,:m2b,:m2c,:m2d]
                (y_hat_fi,err_fi) = nn_comp_2_test(A,x_fi,y,data_norms,model;
                                                   model_type = model_type,
                                                   TL_coef    = TL_coef,
                                                   l_segs     = l_segs,
                                                   silent     = silent)
            elseif model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc,:m3w,:m3tf]
                (y_hat_fi,err_fi) = nn_comp_3_test(A,Bt,B_dot,x_fi,y,data_norms,model;
                                                   model_type = model_type,
                                                   y_type     = y_type,
                                                   TL_coef    = TL_coef,
                                                   terms_A    = terms_A,
                                                   l_segs     = l_segs,
                                                   l_window   = l_window,
                                                   silent     = silent)
            else
                error("$model_type model type not defined")
            end

            if std(err_fi) < std(err)
                y_hat = y_hat_fi
                err   = err_fi
            end

            open(fi_csv,"a") do file
                writedlm(file,zip(i,std(err_fi)),',')
            end

        end

        silent || @info("returning best feature importance results")

    else

        # evaluate model
        if model_type in [:m1]
            (y_hat,err) = nn_comp_1_test(x,y,data_norms,model;
                                         l_segs     = l_segs,
                                         silent     = silent)
        elseif model_type in [:m2a,:m2b,:m2c,:m2d]
            (y_hat,err) = nn_comp_2_test(A,x,y,data_norms,model;
                                         model_type = model_type,
                                         TL_coef    = TL_coef,
                                         l_segs     = l_segs,
                                         silent     = silent)
        elseif model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc,:m3w,:m3tf]
            (y_hat,err) = nn_comp_3_test(A,Bt,B_dot,x,y,data_norms,model;
                                         model_type = model_type,
                                         y_type     = y_type,
                                         TL_coef    = TL_coef,
                                         terms_A    = terms_A,
                                         l_segs     = l_segs,
                                         l_window   = l_window,
                                         silent     = silent)
        elseif model_type in [:TL,:mod_TL,:map_TL]
            (y_hat,err) = linear_test(A,y,data_norms,model;
                                         l_segs     = l_segs,
                                         silent     = silent)
        elseif model_type in [:elasticnet,:plsr]
            (y_hat,err) = linear_test(x,y,data_norms,model;
                                         l_segs     = l_segs,
                                         silent     = silent)
        else
            error("$model_type model type not defined")
        end

    end

    silent || print_time(time()-t0,1)

    return (y, y_hat, err, features)
end # function comp_test

function comp_test(lines, df_line::DataFrame, df_flight::DataFrame,
                   df_map::DataFrame, comp_params::CompParams=NNCompParams();
                   l_window::Int = 5,
                   silent::Bool  = false)
    @warn("this version of comp_test() is deprecated & will be removed in MagNav.jl v1.2.0, see docstring for argument order")
    comp_test(comp_params,lines,df_line,df_flight,df_map;
              l_window = l_window,
              silent   = silent)
end # function comp_test

"""
    comp_m2bc_test(comp_params::NNCompParams, lines,
                   df_line::DataFrame, df_flight::DataFrame, df_map::DataFrame;
                   silent::Bool = false)

Evaluate performance of neural network-based aeromagnetic compensation,
model 2b or 2c with additional outputs for explainability.

**Arguments:**
- `comp_params`: `NNCompParams` neural network-based aeromagnetic compensation parameters struct
- `lines`:       selected line number(s)
- `df_line`:     lookup table (DataFrame) of `lines`
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
- `df_flight`: lookup table (DataFrame) of flight data HDF5 files
|**Field**|**Type**|**Description**
|:--|:--|:--
`flight`  |`Symbol`| flight name (e.g., `:Flt1001`)
`xyz_type`|`Symbol`| subtype of `XYZ` to use for flight data {`:XYZ0`,`:XYZ1`,`:XYZ20`,`:XYZ21`}
`xyz_set` |`Real`  | flight dataset number (used to prevent inproper mixing of datasets, such as different magnetometer locations)
`xyz_h5`  |`String`| path/name of flight data HDF5 file (`.h5` extension optional)
- `df_map`: lookup table (DataFrame) of map data HDF5 files
|**Field**|**Type**|**Description**
|:--|:--|:--
`map_name`|`Symbol`| name of magnetic anomaly map
`map_h5`  |`String`| path/name of map data HDF5 file (`.h5` extension optional)
`map_type`|`Symbol`| (optional) type of magnetic anomaly map {`drape`,`HAE`}
`map_alt` |`Real`  | (optional) map altitude, -1 for drape map [m]
- `silent`: (optional) if true, no print outs

**Returns:**
- `y_nn`:     length `N` neural network compensation portion
- `y_TL`:     length `N` Tolles-Lawson  compensation portion
- `y`:        length `N` target vector
- `y_hat`:    length `N` prediction vector
- `err`:      length `N` mean-corrected (per line) compensation error
- `features`: length `Nf` feature vector (including components of TL `A`, etc.)
"""
function comp_m2bc_test(comp_params::NNCompParams, lines,
                        df_line::DataFrame, df_flight::DataFrame, df_map::DataFrame;
                        silent::Bool = false)

    seed!(2) # for reproducibility
    t0 = time()

    # unpack parameters
    @unpack version, features_setup, features_no_norm, model_type, y_type,
    use_mag, use_vec, data_norms, model, terms, terms_A, sub_diurnal,
    sub_igrf, bpf_mag, reorient_vec, norm_type_A, norm_type_x, norm_type_y,
    TL_coef, η_adam, epoch_adam, epoch_lbfgs, hidden, activation,
    batchsize, frac_train, α_sgl, λ_sgl, k_pca,
    drop_fi, drop_fi_bson, drop_fi_csv, perm_fi, perm_fi_csv = comp_params

    # load data
    (A,x,y,_,features,l_segs) = get_Axy(lines,df_line,df_flight,df_map,
                                        features_setup;
                                        features_no_norm = features_no_norm,
                                        y_type           = y_type,
                                        use_mag          = use_mag,
                                        use_vec          = use_vec,
                                        terms            = terms,
                                        terms_A          = terms_A,
                                        sub_diurnal      = sub_diurnal,
                                        sub_igrf         = sub_igrf,
                                        bpf_mag          = bpf_mag,
                                        reorient_vec     = reorient_vec,
                                        return_B         = false,
                                        silent           = silent_debug)

    # convert to Float32 for consistency with nn_comp_2_train
    A       = Float32.(A)
    x       = Float32.(x)
    y       = Float32.(y)
    TL_coef = Float32.(TL_coef)

    # unpack data normalizations
    (A_bias,A_scale,v_scale,x_bias,x_scale,y_bias,y_scale) =
        unpack_data_norms(data_norms)
    A_norm = ( (A .- A_bias) ./ A_scale           )'
    x_norm = (((x .- x_bias) ./ x_scale) * v_scale)'

    TL_coef_norm = TL_coef ./ y_scale

    # set to test mode in case model uses batchnorm or dropout
    m = model
    Flux.testmode!(m)

    y_nn  = vec(m(x_norm))       .* y_scale
    y_TL  = A_norm'*TL_coef_norm .* y_scale

    (y_hat,err) = nn_comp_2_test(A_norm,x_norm,y,y_bias,y_scale,m;
                                 model_type   = model_type,
                                 TL_coef_norm = TL_coef_norm,
                                 l_segs       = l_segs,
                                 silent       = true)

    silent || @info("std    y_nn: $(round(std(y_nn),digits=2)) nT")
    silent || @info("std    y_TL: $(round(std(y_TL),digits=2)) nT")
    silent || @info("test  error: $(round(std(err ),digits=2)) nT")
    silent || print_time(time()-t0,1)

    return (y_nn, y_TL, y, y_hat, err, features)
end # function comp_m2bc_test

function comp_m2bc_test(lines, df_line::DataFrame,
                        df_flight::DataFrame, df_map::DataFrame,
                        comp_params::NNCompParams=NNCompParams();
                        silent::Bool = false)
    @warn("this version of comp_m2bc_test() is deprecated & will be removed in MagNav.jl v1.2.0, see docstring for argument order")
    comp_m2bc_test(comp_params,lines,df_line,df_flight,df_map;
                   silent = silent)
end # function comp_m2bc_test

"""
    comp_m3_test(comp_params::NNCompParams, lines,
                 df_line::DataFrame, df_flight::DataFrame, df_map::DataFrame;
                 l_window::Int = 5,
                 silent::Bool  = false)

Evaluate performance of neural network-based aeromagnetic compensation, model 3
with additional outputs for explainability.

**Arguments:**
- `comp_params`: `NNCompParams` neural network-based aeromagnetic compensation parameters struct
- `lines`:       selected line number(s)
- `df_line`:     lookup table (DataFrame) of `lines`
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
- `df_flight`: lookup table (DataFrame) of flight data HDF5 files
|**Field**|**Type**|**Description**
|:--|:--|:--
`flight`  |`Symbol`| flight name (e.g., `:Flt1001`)
`xyz_type`|`Symbol`| subtype of `XYZ` to use for flight data {`:XYZ0`,`:XYZ1`,`:XYZ20`,`:XYZ21`}
`xyz_set` |`Real`  | flight dataset number (used to prevent inproper mixing of datasets, such as different magnetometer locations)
`xyz_h5`  |`String`| path/name of flight data HDF5 file (`.h5` extension optional)
- `df_map`: lookup table (DataFrame) of map data HDF5 files
|**Field**|**Type**|**Description**
|:--|:--|:--
`map_name`|`Symbol`| name of magnetic anomaly map
`map_h5`  |`String`| path/name of map data HDF5 file (`.h5` extension optional)
`map_type`|`Symbol`| (optional) type of magnetic anomaly map {`drape`,`HAE`}
`map_alt` |`Real`  | (optional) map altitude, -1 for drape map [m]
- `l_window`: (optional) temporal window length, only used for `model_type = :m3w, :m3tf`
- `silent`:   (optional) if true, no print outs

**Returns:**
- `TL_perm`:      `3` x `N` matrix of TL permanent vector field
- `TL_induced`:   `3` x `N` matrix of TL induced vector field
- `TL_eddy`:      `3` x `N` matrix of TL eddy current vector field
- `TL_aircraft`:  `3` x `N` matrix of TL aircraft vector field
- `B_unit`:       `3` x `N` matrix of normalized vector magnetometer measurements
- `B_vec`:        `3` x `N` matrix of vector magnetometer measurements
- `y_nn`:         `3` x `N` matrix of vector neural network correction (for scalar models, in direction of `Bt`)
- `vec_aircraft`: `3` x `N` matrix of predicted aircraft vector field
- `y`:            length `N` target vector
- `y_hat`:        length `N` prediction vector
- `err`:          length `N` mean-corrected (per line) compensation error
- `features`:     length `Nf` feature vector (including components of TL `A`, etc.)
"""
function comp_m3_test(comp_params::NNCompParams, lines,
                      df_line::DataFrame, df_flight::DataFrame, df_map::DataFrame;
                      l_window::Int = 5,
                      silent::Bool  = false)

    seed!(2) # for reproducibility
    t0 = time()

    # unpack parameters
    @unpack version, features_setup, features_no_norm, model_type, y_type,
    use_mag, use_vec, data_norms, model, terms, terms_A, sub_diurnal,
    sub_igrf, bpf_mag, reorient_vec, norm_type_A, norm_type_x, norm_type_y,
    TL_coef, η_adam, epoch_adam, epoch_lbfgs, hidden, activation,
    batchsize, frac_train, α_sgl, λ_sgl, k_pca,
    drop_fi, drop_fi_bson, drop_fi_csv, perm_fi, perm_fi_csv = comp_params

    @assert y_type in [:a,:b,:c,:d] "unsupported y_type = $y_type for nn_comp_3"
    @assert model_type in [:m3s,:m3v,:m3sc,:m3vc,:m3w,:m3tf] "unsupported model_type = $model_type for model 3 explainability"

    mod_TL = model_type == :mod_TL ? true : false
    map_TL = model_type == :map_TL ? true : false

    (A,Bt,B_dot,x,y,_,features,l_segs) = get_Axy(lines,df_line,df_flight,df_map,
                                                 features_setup;
                                                 features_no_norm = features_no_norm,
                                                 y_type           = y_type,
                                                 use_mag          = use_mag,
                                                 use_vec          = use_vec,
                                                 terms            = terms,
                                                 terms_A          = terms_A,
                                                 sub_diurnal      = sub_diurnal,
                                                 sub_igrf         = sub_igrf,
                                                 bpf_mag          = bpf_mag,
                                                 reorient_vec     = reorient_vec,
                                                 mod_TL           = mod_TL,
                                                 map_TL           = map_TL,
                                                 return_B         = true,
                                                 silent           = silent_debug)

    # convert to Float32 for consistency with nn_comp_3_train
    A       = Float32.(A)
    Bt      = Float32.(Bt)    # magnitude of total field measurements
    B_dot   = Float32.(B_dot) # finite differences of total field vector
    x       = Float32.(x)
    y       = Float32.(y)
    TL_coef = Float32.(TL_coef)

    # assume all terms are stored, but they may be zero if not trained
    Bt_scale = 50000f0
    (TL_coef_p,TL_coef_i,TL_coef_e) = TL_vec2mat(TL_coef,terms_A;Bt_scale=Bt_scale)

    B_unit    = A[:,1:3]'     # normalized vector magnetometer reading
    B_vec     = B_unit .* Bt' # vector magnetometer to be used in TL
    B_vec_dot = B_dot'        # not exactly true, but internally consistent

    # unpack data normalizations
    (_,_,v_scale,x_bias,x_scale,y_bias,y_scale) = unpack_data_norms(data_norms)
    x_norm = (((x .- x_bias) ./ x_scale) * v_scale)'

    model_type in [:m3w,:m3tf] && (x_norm = get_temporal_data(x_norm,l_window))

    # set to test mode in case model uses batchnorm or dropout
    m = model
    Flux.testmode!(m)

    # calculate TL vector field
    (TL_aircraft,TL_perm,TL_induced,TL_eddy) =
        get_TL_aircraft_vec(B_vec,B_vec_dot,TL_coef_p,TL_coef_i,TL_coef_e;
                            return_parts=true)

    # compute neural network correction
    if model_type in [:m3s,:m3sc,:m3w,:m3tf] # scalar-corrected
        y_nn = vec(m(x_norm)) .* y_scale # rescale to TL [N]
        y_nn = y_nn' .* B_unit # assume same direction [3xN]
    elseif model_type in [:m3v,:m3vc] # vector-corrected
        y_nn = m(x_norm) .* y_scale # rescale to TL [3xN]
    end

    vec_aircraft = TL_aircraft + y_nn # [3xN]

    # compute aircraft field correction or Earth field target
    (y_hat,err) = nn_comp_3_test(B_unit,B_vec,B_vec_dot,
                                 x_norm,y,y_bias,y_scale,m,
                                 TL_coef_p,TL_coef_i,TL_coef_e;
                                 model_type = model_type,
                                 y_type     = y_type,
                                 l_segs     = l_segs,
                                 use_nn     = true,
                                 denorm     = true,
                                 testmode   = true,
                                 silent     = true)

    silent || @info("test  error: $(round(std(err),digits=2)) nT")
    silent || print_time(time()-t0,1)

    return (TL_perm, TL_induced, TL_eddy, TL_aircraft,
            B_unit, B_vec, y_nn, vec_aircraft, y, y_hat, err, features)
end # function comp_m3_test

function comp_m3_test(lines, df_line::DataFrame,
                      df_flight::DataFrame, df_map::DataFrame,
                      comp_params::NNCompParams=NNCompParams();
                      l_window::Int = 5,
                      silent::Bool  = false)
    @warn("this version of comp_m3_test() is deprecated & will be removed in MagNav.jl v1.2.0, see docstring for argument order")
    comp_m3_test(comp_params,lines,df_line,df_flight,df_map;
                 l_window = l_window,
                 silent   = silent)
end # function comp_m3_test

"""
    comp_train_test(comp_params::CompParams,
                    xyz_train::XYZ, xyz_test::XYZ, ind_train, ind_test,
                    mapS_train::Union{MapS,MapSd,MapS3D} = mapS_null,
                    mapS_test::Union{MapS,MapSd,MapS3D}  = mapS_null;
                    σ_curriculum          = 1.0,
                    l_window::Int         = 5,
                    window_type::Symbol   = :sliding,
                    tf_layer_type::Symbol = :postlayer,
                    tf_norm_type::Symbol  = :batch,
                    dropout_prob          = 0.2,
                    N_tf_head::Int        = 8,
                    tf_gain               = 1.0,
                    silent::Bool          = false)

Train and evaluate performance of an aeromagnetic compensation model.

**Arguments:**
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct, either:
    - `NNCompParams`:  neural network-based aeromagnetic compensation parameters struct
    - `LinCompParams`: linear aeromagnetic compensation parameters struct
- `xyz_train`:     `XYZ` flight data struct for training
- `xyz_test`:      `XYZ` flight data struct for testing
- `ind_train`:     selected data indices for training
- `ind_test`:      selected data indices for testing
- `mapS_train`:    (optional) `MapS`, `MapSd`, or `MapS3D` scalar magnetic anomaly map struct for training, only used for `y_type = :b, :c`
- `mapS_test`:     (optional) `MapS`, `MapSd`, or `MapS3D` scalar magnetic anomaly map struct for testing,  only used for `y_type = :b, :c`
- `σ_curriculum`:  (optional) standard deviation threshold, only used for `model_type = :m3sc, :m3vc`
- `l_window`:      (optional) temporal window length, only used for `model_type = :m3w, :m3tf`
- `window_type`:   (optional) type of windowing, `:sliding` for overlapping or `:contiguous` for non-overlapping, only used for `model_type = :m3w, :m3tf`
- `tf_layer_type`: (optional) transformer normalization layer before or after skip connection {`:prelayer`,`:postlayer`}, only used for `model_type = :m3tf`
- `tf_norm_type`:  (optional) normalization for transformer encoder {`:batch`,`:layer`,`:none`}, only used for `model_type = :m3tf`
- `dropout_prob`:  (optional) dropout rate, only used for `model_type = :m3w, :m3tf`
- `N_tf_head`:     (optional) number of attention heads, only used for `model_type = :m3tf`
- `tf_gain`:       (optional) weight initialization parameter, only used for `model_type = :m3tf`
- `silent`:        (optional) if true, no print outs

**Returns:**
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct
- `y_train`:     length `N_train` training target vector
- `y_train_hat`: length `N_train` training prediction vector
- `err_train`:   length `N_train` training compensation error
- `y_test`:      length `N_test` testing target vector
- `y_test_hat`:  length `N_test` testing prediction vector
- `err_test`:    length `N_test` testing compensation error
- `features`:    length `Nf` feature vector (including components of TL `A`, etc.)
"""
function comp_train_test(comp_params::CompParams,
                         xyz_train::XYZ, xyz_test::XYZ, ind_train, ind_test,
                         mapS_train::Union{MapS,MapSd,MapS3D} = mapS_null,
                         mapS_test::Union{MapS,MapSd,MapS3D}  = mapS_null;
                         σ_curriculum          = 1.0,
                         l_window::Int         = 5,
                         window_type::Symbol   = :sliding,
                         tf_layer_type::Symbol = :postlayer,
                         tf_norm_type::Symbol  = :batch,
                         dropout_prob          = 0.2,
                         N_tf_head::Int        = 8,
                         tf_gain               = 1.0,
                         silent::Bool          = false)

    @assert typeof(xyz_train) == typeof(xyz_test) "xyz types do no match"

    (comp_params,y_train,y_train_hat,err_train,features) =
        comp_train(comp_params,xyz_train,ind_train,mapS_train;
                   σ_curriculum  = σ_curriculum,
                   l_window      = l_window,
                   window_type   = window_type,
                   tf_layer_type = tf_layer_type,
                   tf_norm_type  = tf_norm_type,
                   dropout_prob  = dropout_prob,
                   N_tf_head     = N_tf_head,
                   tf_gain       = tf_gain,
                   silent        = silent)

    (y_test,y_test_hat,err_test,_) =
        comp_test(comp_params,xyz_test,ind_test,mapS_test;
                  l_window = l_window,
                  silent   = silent)

    return (comp_params, y_train, y_train_hat, err_train,
                         y_test , y_test_hat , err_test , features)
end # function comp_train_test

function comp_train_test(xyz_train::XYZ, xyz_test::XYZ, ind_train, ind_test,
                         mapS_train::Union{MapS,MapSd,MapS3D} = mapS_null,
                         mapS_test::Union{MapS,MapSd,MapS3D}  = mapS_null;
                         comp_params::CompParams = NNCompParams(),
                         σ_curriculum            = 1.0,
                         l_window::Int           = 5,
                         window_type::Symbol     = :sliding,
                         tf_layer_type::Symbol   = :postlayer,
                         tf_norm_type::Symbol    = :batch,
                         dropout_prob            = 0.2,
                         N_tf_head::Int          = 8,
                         tf_gain                 = 1.0,
                         silent::Bool            = false)
    @warn("this version of comp_train_test() is deprecated & will be removed in MagNav.jl v1.2.0, see docstring for argument order")
    comp_train_test(comp_params,xyz_train,xyz_test,
                    ind_train,ind_test,mapS_train,mapS_test;
                    σ_curriculum  = σ_curriculum,
                    l_window      = l_window,
                    window_type   = window_type,
                    tf_layer_type = tf_layer_type,
                    tf_norm_type  = tf_norm_type,
                    dropout_prob  = dropout_prob,
                    N_tf_head     = N_tf_head,
                    tf_gain       = tf_gain,
                    silent        = silent)
end # function comp_train_test

"""
    comp_train_test(comp_params::CompParams, lines_train, lines_test,
                    df_line::DataFrame, df_flight::DataFrame, df_map::DataFrame;
                    σ_curriculum          = 1.0,
                    l_window::Int         = 5,
                    window_type::Symbol   = :sliding,
                    tf_layer_type::Symbol = :postlayer,
                    tf_norm_type::Symbol  = :batch,
                    dropout_prob          = 0.2,
                    N_tf_head::Int        = 8,
                    tf_gain               = 1.0,
                    silent::Bool          = false)

Train and evaluate performance of an aeromagnetic compensation model.

**Arguments:**
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct, either:
    - `NNCompParams`:  neural network-based aeromagnetic compensation parameters struct
    - `LinCompParams`: linear aeromagnetic compensation parameters struct
- `lines_train`: selected line number(s) for training
- `lines_test`:  selected line number(s) for testing
- `df_line`:     lookup table (DataFrame) of `lines`
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
- `df_flight`: lookup table (DataFrame) of flight data HDF5 files
|**Field**|**Type**|**Description**
|:--|:--|:--
`flight`  |`Symbol`| flight name (e.g., `:Flt1001`)
`xyz_type`|`Symbol`| subtype of `XYZ` to use for flight data {`:XYZ0`,`:XYZ1`,`:XYZ20`,`:XYZ21`}
`xyz_set` |`Real`  | flight dataset number (used to prevent inproper mixing of datasets, such as different magnetometer locations)
`xyz_h5`  |`String`| path/name of flight data HDF5 file (`.h5` extension optional)
- `df_map`: lookup table (DataFrame) of map data HDF5 files
|**Field**|**Type**|**Description**
|:--|:--|:--
`map_name`|`Symbol`| name of magnetic anomaly map
`map_h5`  |`String`| path/name of map data HDF5 file (`.h5` extension optional)
`map_type`|`Symbol`| (optional) type of magnetic anomaly map {`drape`,`HAE`}
`map_alt` |`Real`  | (optional) map altitude, -1 for drape map [m]
- `σ_curriculum`:  (optional) standard deviation threshold, only used for `model_type = :m3sc, :m3vc`
- `l_window`:      (optional) temporal window length, only used for `model_type = :m3w, :m3tf`
- `window_type`:   (optional) type of windowing, `:sliding` for overlapping or `:contiguous` for non-overlapping, only used for `model_type = :m3w, :m3tf`
- `tf_layer_type`: (optional) transformer normalization layer before or after skip connection {`:prelayer`,`:postlayer`}, only used for `model_type = :m3tf`
- `tf_norm_type`:  (optional) normalization for transformer encoder {`:batch`,`:layer`,`:none`}, only used for `model_type = :m3tf`
- `dropout_prob`:  (optional) dropout rate, only used for `model_type = :m3w, :m3tf`
- `N_tf_head`:     (optional) number of attention heads, only used for `model_type = :m3tf`
- `tf_gain`:       (optional) weight initialization parameter, only used for `model_type = :m3tf`
- `silent`:        (optional) if true, no print outs

**Returns:**
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct
- `y_train`:     length `N_train` training target vector
- `y_train_hat`: length `N_train` training prediction vector
- `err_train`:   length `N_train` mean-corrected (per line) training compensation error
- `y_test`:      length `N_test` testing target vector
- `y_test_hat`:  length `N_test` testing prediction vector
- `err_test`:    length `N_test` mean-corrected (per line) testing compensation error
- `features`:    length `Nf` feature vector (including components of TL `A`, etc.)
"""
function comp_train_test(comp_params::CompParams, lines_train, lines_test,
                         df_line::DataFrame, df_flight::DataFrame, df_map::DataFrame;
                         σ_curriculum          = 1.0,
                         l_window::Int         = 5,
                         window_type::Symbol   = :sliding,
                         tf_layer_type::Symbol = :postlayer,
                         tf_norm_type::Symbol  = :batch,
                         dropout_prob          = 0.2,
                         N_tf_head::Int        = 8,
                         tf_gain               = 1.0,
                         silent::Bool          = false)

    (comp_params,y_train,y_train_hat,err_train,features) =
        comp_train(comp_params,lines_train,df_line,df_flight,df_map;
                   σ_curriculum  = σ_curriculum,
                   l_window      = l_window,
                   window_type   = window_type,
                   tf_layer_type = tf_layer_type,
                   tf_norm_type  = tf_norm_type,
                   dropout_prob  = dropout_prob,
                   N_tf_head     = N_tf_head,
                   tf_gain       = tf_gain,
                   silent        = silent)

    (y_test,y_test_hat,err_test,_) =
        comp_test(comp_params,lines_test,df_line,df_flight,df_map;
                  l_window = l_window,
                  silent   = silent)

    return (comp_params, y_train, y_train_hat, err_train,
                         y_test , y_test_hat , err_test , features)
end # function comp_train_test

function comp_train_test(lines_train, lines_test,
                         df_line::DataFrame, df_flight::DataFrame, df_map::DataFrame,
                         comp_params::CompParams = NNCompParams();
                         σ_curriculum            = 1.0,
                         l_window::Int           = 5,
                         window_type::Symbol     = :sliding,
                         tf_layer_type::Symbol   = :postlayer,
                         tf_norm_type::Symbol    = :batch,
                         dropout_prob            = 0.2,
                         N_tf_head::Int          = 8,
                         tf_gain                 = 1.0,
                         silent::Bool            = false)
    @warn("this version of comp_train_test() is deprecated & will be removed in MagNav.jl v1.2.0, see docstring for argument order")
    comp_train_test(comp_params,lines_train,lines_test,
                    df_line,df_flight,df_map;
                    σ_curriculum  = σ_curriculum,
                    l_window      = l_window,
                    window_type   = window_type,
                    tf_layer_type = tf_layer_type,
                    tf_norm_type  = tf_norm_type,
                    dropout_prob  = dropout_prob,
                    N_tf_head     = N_tf_head,
                    tf_gain       = tf_gain,
                    silent        = silent)
end # function comp_train_test

"""
    function print_time(t, digits::Int=1)

Internal helper function to print time `t` in `sec` if <1 min, otherwise `min`.
"""
function print_time(t, digits::Int=1)
    if t < 60
        @info("time: $(round(t   ,digits=digits)) sec")
    else
        @info("time: $(round(t/60,digits=digits)) min")
    end
end # function print_time
