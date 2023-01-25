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
                    weights::Params      = Params([]),
                    l_segs::Vector       = [length(y)],
                    silent::Bool         = true)

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
                         weights::Params      = Params([]),
                         l_segs::Vector       = [length(y)],
                         silent::Bool         = true)

    # convert to Float32 for ~50% speedup
    x = convert.(Float32,x)
    y = convert.(Float32,y)
    α = convert.(Float32,α_sgl)
    λ = convert.(Float32,λ_sgl)

    if sum(data_norms[end]) == 0 # normalize data
        (x_bias,x_scale,x_norm) = norm_sets(x;norm_type=norm_type_x,no_norm=no_norm)
        (y_bias,y_scale,y_norm) = norm_sets(y;norm_type=norm_type_y)
        if k_pca > 0
            if k_pca > size(x,2)
                k_pca = size(x,2)
                @info("reducing k_pca to $k_pca, size(x,2)")
            end
            (_,S,V) = svd(cov(x_norm))
            v_scale = V[:,1:k_pca]*inv(Diagonal(sqrt.(S[1:k_pca])))
            var_ret = round(sum(sqrt.(S[1:k_pca]))/sum(sqrt.(S))*100,digits=6)
            @info("k_pca = $k_pca of $(size(x,2)), variance retained: $var_ret %")
        else
            v_scale = I(size(x,2))
        end
        x_norm = x_norm * v_scale
    else # unpack data normalizations
        (_,_,v_scale,x_bias,x_scale,y_bias,y_scale) = unpack_data_norms(data_norms)
        x_norm = ((x .- x_bias) ./ x_scale) * v_scale
        y_norm =  (y .- y_bias) ./ y_scale
    end

    # separate into training and validation
    if frac_train < 1
        N = size(x_norm,1)
        p = randperm(N)
        N_train = floor(Int,frac_train*N)
        p_train = p[1:N_train]
        p_val   = p[N_train+1:end]
        x_norm_train = x_norm[p_train,:]
        x_norm_val   = x_norm[p_val  ,:]
        y_norm_train = y_norm[p_train,:]
        y_norm_val   = y_norm[p_val  ,:]
        data_train   = Flux.DataLoader((x_norm_train',y_norm_train'),
                                       shuffle=true,batchsize=batchsize)
        data_val     = Flux.DataLoader((x_norm_val',y_norm_val'),
                                       shuffle=true,batchsize=batchsize)
    else
        data_train   = Flux.DataLoader((x_norm',y_norm'),
                                       shuffle=true,batchsize=batchsize)
        data_val     = data_train
    end

    # setup NN
    xS = size(x_norm,2) # number of features
    yS = 1 # length of output

    if weights == Params([])
        m = get_nn_m(xS,yS;hidden=hidden,activation=activation)
    else
        m = get_nn_m(weights,activation)
    end

    ## setup optimizer and loss function
    opt = Adam(η_adam)
    loss_m1(x_l,y_l)   = Flux.mse(m(x_l),y_l)
    loss_m1_λ(x_l,y_l) = Flux.mse(m(x_l),y_l) + λ*sum(sparse_group_lasso(m,α))
    loss =  λ > 0 ? loss_m1_λ : loss_m1
    function loss_all(data_l)
        l = 0f0
        for (x_l,y_l) in data_l
            l += loss(x_l,y_l)
        end
        l/length(data_l)
    end # function loss_all

    # train NN with Adam optimizer
    weights   = Flux.params(m)
    best_loss = loss_all(data_val)
    @info("Epoch 0: loss = $best_loss")
    for i = 1:epoch_adam
        Flux.train!(loss,Flux.params(m),data_train,opt)
        current_loss = loss_all(data_val)
        if current_loss < best_loss
            best_loss = current_loss
            weights   = deepcopy(Flux.params(m))
        end
        mod(i,5) == 0 && @info("Epoch $i: loss = $best_loss")
        if mod(i,10) == 0
            y_hat = denorm_sets(y_bias,y_scale,vec(m(x_norm')))
            err   = err_segs(y_hat,y,l_segs;silent=silent)
            @info("train error: $(round(std(err),digits=2)) nT")
        end
    end

    Flux.loadparams!(m,weights)

    if epoch_lbfgs > 0 # LBFGS, may overfit depending on iterations
        data = Flux.DataLoader((x_norm',y_norm'),shuffle=true,batchsize=batchsize)

        function lbfgs_train!(m_l,data_l,iter,λ_l,α_l)
            (x_l,y_l) = data_l.data
            loss_m1()   = Flux.mse(m_l(x_l),y_l)
            loss_m1_λ() = Flux.mse(m_l(x_l),y_l) + λ_l*sum(sparse_group_lasso(m_l,α_l))
            loss =  λ > 0 ? loss_m1_λ : loss_m1
            refresh()
            params = Flux.params(m_l)
            opt = LBFGS()
            (_,_,fg!,p0) = optfuns(loss,params)
            res = optimize(only_fg!(fg!),p0,opt,
                  Options(iterations=iter,show_trace=true))
            return (res)
        end # function lbfgs_train!

        # train NN with LBFGS optimizer
        res = lbfgs_train!(m,data,epoch_lbfgs,λ,α)
    end

    # set to test mode in case model uses batchnorm or dropout
    Flux.testmode!(m)

    # get results
    y_hat = denorm_sets(y_bias,y_scale,vec(m(x_norm')))
    err   = err_segs(y_hat,y,l_segs;silent=silent)
    @info("train error: $(round(std(err),digits=2)) nT")

    # pack data normalizations
    weights = deepcopy(Flux.params(m))
    data_norms = (zeros(1,1),zeros(1,1),v_scale,x_bias,x_scale,y_bias,y_scale)

    return (weights, data_norms, y_hat, err)
end # function nn_comp_1_train

"""
    nn_comp_1_test(x, y, data_norms::Tuple, weights::Params;
                   activation::Function = swish,
                   l_segs::Vector       = [length(y)],
                   silent::Bool         = false)

Evaluate neural network-based aeromagnetic compensation, model 1 performance.
"""
function nn_comp_1_test(x, y, data_norms::Tuple, weights::Params;
                        activation::Function = swish,
                        l_segs::Vector       = [length(y)],
                        silent::Bool         = false)

    # unpack data normalizations
    (_,_,v_scale,x_bias,x_scale,y_bias,y_scale) = unpack_data_norms(data_norms)

    x_norm = ((x .- x_bias) ./ x_scale) * v_scale

    # set to test mode in case model uses batchnorm or dropout
    m = get_nn_m(weights,activation)
    Flux.testmode!(m)

    # get results
    y_hat = denorm_sets(y_bias,y_scale,vec(m(x_norm')))
    err   = err_segs(y_hat,y,l_segs;silent=silent)
    @info("test  error: $(round(std(err),digits=2)) nT")

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
                    weights::Params      = Params([]),
                    l_segs::Vector       = [length(y)],
                    silent::Bool         = true)

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
                         weights::Params      = Params([]),
                         l_segs::Vector       = [length(y)],
                         silent::Bool         = true)

    # convert to Float32 for ~50% speedup
    A = convert.(Float32,A)
    x = convert.(Float32,x)
    y = convert.(Float32,y)
    α = convert.(Float32,α_sgl)
    λ = convert.(Float32,λ_sgl)
    TL_coef = convert.(Float32,TL_coef)

    if sum(data_norms[end]) == 0 # normalize data
        (A_bias,A_scale,A_norm) = norm_sets(A;norm_type=norm_type_A)
        (x_bias,x_scale,x_norm) = norm_sets(x;norm_type=norm_type_x,no_norm=no_norm)
        (y_bias,y_scale,y_norm) = norm_sets(y;norm_type=norm_type_y)
        if k_pca > 0
            if k_pca > size(x,2)
                k_pca = size(x,2)
                @info("reducing k_pca to $k_pca, size(x,2)")
            end
            (_,S,V) = svd(cov(x_norm))
            v_scale = V[:,1:k_pca]*inv(Diagonal(sqrt.(S[1:k_pca])))
            var_ret = round(sum(sqrt.(S[1:k_pca]))/sum(sqrt.(S))*100,digits=6)
            @info("k_pca = $k_pca of $(size(x,2)), variance retained: $var_ret %")
        else
            v_scale = I(size(x,2))
        end
        x_norm = x_norm * v_scale
    else # unpack data normalizations
        (A_bias,A_scale,v_scale,x_bias,x_scale,y_bias,y_scale) = 
            unpack_data_norms(data_norms)
        A_norm =  (A .- A_bias) ./ A_scale
        x_norm = ((x .- x_bias) ./ x_scale) * v_scale
        y_norm =  (y .- y_bias) ./ y_scale
    end

    # separate into training and validation
    if frac_train < 1
        N = size(x_norm,1)
        p = randperm(N)
        N_train = floor(Int,frac_train*N)
        p_train = p[1:N_train]
        p_val   = p[N_train+1:end]
        A_train = A_norm[p_train,:]
        A_val   = A_norm[p_val  ,:]
        x_norm_train = x_norm[p_train,:]
        x_norm_val   = x_norm[p_val  ,:]
        y_norm_train = y_norm[p_train,:]
        y_norm_val   = y_norm[p_val  ,:]
        data_train = Flux.DataLoader((A_train',x_norm_train',y_norm_train'),
                                     shuffle=true,batchsize=batchsize)
        data_val   = Flux.DataLoader((A_val',x_norm_val',y_norm_val'),
                                     shuffle=true,batchsize=batchsize)
    else
        data_train = Flux.DataLoader((A_norm',x_norm',y_norm'),
                                     shuffle=true,batchsize=batchsize)
        data_val   = data_train
    end

    # setup NN
    xS = size(x_norm,2) # number of features
    if model_type in [:m2a,:m2d]
        yS = size(A_norm,2) # length of output
    elseif model_type in [:m2b,:m2c]
        yS = 1 # additive correction
    end

    if weights == Params([]) # not re-training with known weights
        m = get_nn_m(xS,yS;hidden=hidden,activation=activation)
    else # initial weights already known
        if model_type in [:m2b,:m2c,:m2d]
            TL_coef = vec(weights[length(weights)])
            m = get_nn_m(Flux.params(weights[1:length(weights)-1]),activation)
        else
            m = get_nn_m(weights,activation)
        end
    end

    TL_coef = TL_coef/y_scale

    ## setup optimizer and loss function
    opt = Adam(η_adam)
    loss_m2a(  A_l,x_l,y_l) = Flux.mse(vec(sum(A_l.*m(x_l),dims=1)) ,vec(y_l))
    loss_m2b(  A_l,x_l,y_l) = Flux.mse(vec(m(x_l))+vec(A_l'*TL_coef),vec(y_l))
    loss_m2c(  A_l,x_l,y_l) = Flux.mse(vec(m(x_l))+vec(A_l'*TL_coef),vec(y_l))
    loss_m2d(  A_l,x_l,y_l) = Flux.mse(vec(sum(A_l.*(m(x_l).+TL_coef),dims=1)),vec(y_l))
    loss_m2a_λ(A_l,x_l,y_l) = loss_m2a(A_l,x_l,y_l) + λ*sum(sparse_group_lasso(m,α))
    loss_m2b_λ(A_l,x_l,y_l) = loss_m2b(A_l,x_l,y_l) + λ*sum(sparse_group_lasso(m,α))
    loss_m2c_λ(A_l,x_l,y_l) = loss_m2c(A_l,x_l,y_l) + λ*sum(sparse_group_lasso(m,α))
    loss_m2d_λ(A_l,x_l,y_l) = loss_m2d(A_l,x_l,y_l) + λ*sum(sparse_group_lasso(m,α))
    model_type == :m2a && (loss = λ > 0 ? loss_m2a_λ : loss_m2a)
    model_type == :m2b && (loss = λ > 0 ? loss_m2b_λ : loss_m2b)
    model_type == :m2c && (loss = λ > 0 ? loss_m2c_λ : loss_m2c)
    model_type == :m2d && (loss = λ > 0 ? loss_m2d_λ : loss_m2d)
    function loss_all(data_l)
        l = 0f0
        for (A_l,x_l,y_l) in data_l
            l += loss(A_l,x_l,y_l)
        end
        l/length(data_l)
    end # function loss_all

    # train NN with Adam optimizer
    if model_type in [:m2a] # store NN weights only
        weights = deepcopy(Flux.params(m))
    else # store NN weights + de-scaled TL coef
        weights = deepcopy(Flux.params(m,TL_coef*y_scale))
    end
    best_loss = loss_all(data_val)

    @info("Epoch 0: loss = $best_loss")
    for i = 1:epoch_adam
        if model_type in [:m2a,:m2b,:m2d] # train on NN weights only
            Flux.train!(loss,Flux.params(m),data_train,opt)
        elseif model_type in [:m2c] # train on NN weights + TL coef
            Flux.train!(loss,Flux.params(m,TL_coef),data_train,opt)
        end
        current_loss = loss_all(data_val)
        if current_loss < best_loss
            best_loss = current_loss
            if model_type in [:m2a] # store NN weights only
                weights = deepcopy(Flux.params(m))
            else # store NN weights + de-scaled TL coef
                weights = deepcopy(Flux.params(m,TL_coef*y_scale))
            end
        end
        mod(i,5) == 0 && @info("Epoch $i: loss = $best_loss")
        if mod(i,10) == 0
            if model_type in [:m2a]
                y_hat = denorm_sets(y_bias,y_scale,vec(sum(A_norm'.*m(x_norm'),dims=1)))
            elseif model_type in [:m2b,:m2c]
                y_hat = denorm_sets(y_bias,y_scale,vec(m(x_norm'))+vec(A_norm*TL_coef))
            elseif model_type in [:m2d]
                y_hat = denorm_sets(y_bias,y_scale,vec(sum(A_norm'.*(m(x_norm').+TL_coef),dims=1)))
            end
            err = err_segs(y_hat,y,l_segs;silent=silent)
            @info("$i train error: $(round(std(err),digits=2)) nT")
        end
    end

    Flux.loadparams!(m,weights)

    if epoch_lbfgs > 0 # LBFGS, may overfit depending on iterations
        data = Flux.DataLoader((A_norm',x_norm',y_norm'),shuffle=true,batchsize=batchsize)

        function lbfgs_train!(m_l,data_l,iter,m_t,λ_l,α_l)
            (A_l,x_l,y_l) = data_l.data
            loss_m2a()   = Flux.mse(vec(sum(A_l.*m_l(x_l),dims=1)) ,vec(y_l))
            loss_m2b()   = Flux.mse(vec(m_l(x_l))+vec(A_l'*TL_coef),vec(y_l))
            loss_m2c()   = Flux.mse(vec(m_l(x_l))+vec(A_l'*TL_coef),vec(y_l))
            loss_m2d()   = Flux.mse(vec(sum(A_l.*(m_l(x_l).+TL_coef),dims=1)),vec(y_l))
            loss_m2a_λ() = loss_m2a() + λ_l*sum(sparse_group_lasso(m_l,α_l))
            loss_m2b_λ() = loss_m2b() + λ_l*sum(sparse_group_lasso(m_l,α_l))
            loss_m2c_λ() = loss_m2c() + λ_l*sum(sparse_group_lasso(m_l,α_l))
            loss_m2d_λ() = loss_m2d() + λ_l*sum(sparse_group_lasso(m_l,α_l))
            m_t == :m2a && (loss = λ > 0 ? loss_m2a_λ : loss_m2a)
            m_t == :m2b && (loss = λ > 0 ? loss_m2b_λ : loss_m2b)
            m_t == :m2c && (loss = λ > 0 ? loss_m2c_λ : loss_m2c)
            m_t == :m2d && (loss = λ > 0 ? loss_m2d_λ : loss_m2d)
            refresh()
            params = Flux.params(m_l)
            opt = LBFGS()
            (_,_,fg!,p0) = optfuns(loss,params)
            res = optimize(only_fg!(fg!),p0,opt,
                  Options(iterations=iter,show_trace=true))
            return (res)
        end # function lbfgs_train!

        # train NN with LBFGS optimizer
        res = lbfgs_train!(m,data,epoch_lbfgs,model_type,λ,α)
    end

    # set to test mode in case model uses batchnorm or dropout
    Flux.testmode!(m)

    # get results
    if model_type in [:m2a]
        y_hat = denorm_sets(y_bias,y_scale,vec(sum(A_norm'.*m(x_norm'),dims=1)))
    elseif model_type in [:m2b,:m2c]
        y_hat = denorm_sets(y_bias,y_scale,vec(m(x_norm'))+vec(A_norm*TL_coef))
    elseif model_type in [:m2d]
        y_hat = denorm_sets(y_bias,y_scale,vec(sum(A_norm'.*(m(x_norm').+TL_coef),dims=1)))
    end
    err = err_segs(y_hat,y,l_segs;silent=silent)
    @info("train error: $(round(std(err),digits=2)) nT")

    # pack data normalizations
    if model_type in [:m2a] # store NN weights only
        weights = deepcopy(Flux.params(m))
    else # store NN weights + de-scaled TL coef
        weights = deepcopy(Flux.params(m,TL_coef*y_scale))
    end
    data_norms = (A_bias,A_scale,v_scale,x_bias,x_scale,y_bias,y_scale)

    return (weights, data_norms, y_hat, err)
end # function nn_comp_2_train

"""
    nn_comp_2_test(A, x, y, data_norms::Tuple, weights::Params;
                   model_type::Symbol   = :m2a,
                   activation::Function = swish,
                   l_segs::Vector       = [length(y)],
                   silent::Bool         = false)

Evaluate neural network-based aeromagnetic compensation, model 2 performance.
"""
function nn_comp_2_test(A, x, y, data_norms::Tuple, weights::Params;
                        model_type::Symbol   = :m2a,
                        activation::Function = swish,
                        l_segs::Vector       = [length(y)],
                        silent::Bool         = false)

    # unpack data normalizations
    (A_bias,A_scale,v_scale,x_bias,x_scale,y_bias,y_scale) = 
        unpack_data_norms(data_norms)

    if model_type in [:m2b,:m2c,:m2d] # contain NN weights + TL coef
        TL_coef = vec(weights[length(weights)])/y_scale
        weights = Flux.params(weights[1:length(weights)-1])
    end

    A_norm =  (A .- A_bias) ./ A_scale
    x_norm = ((x .- x_bias) ./ x_scale) * v_scale

    # set to test mode in case model uses batchnorm or dropout
    m = get_nn_m(weights,activation)
    Flux.testmode!(m)

    # get results
    if model_type in [:m2a]
        y_hat = denorm_sets(y_bias,y_scale,vec(sum(A_norm'.*m(x_norm'),dims=1)))
    elseif model_type in [:m2b,:m2c]
        y_hat = denorm_sets(y_bias,y_scale,vec(m(x_norm'))+vec(A_norm*TL_coef))
    elseif model_type in [:m2d]
        y_hat = denorm_sets(y_bias,y_scale,vec(sum(A_norm'.*(m(x_norm').+TL_coef),dims=1)))
    end
    err = err_segs(y_hat,y,l_segs;silent=silent)
    @info("test  error: $(round(std(err),digits=2)) nT")

    return (y_hat, err)
end # function nn_comp_2_test

"""
    get_curriculum_ind(x, x_ind, sigma=1)

Get indices for curriculum learning (to train the Tolles-Lawson section of  
the loss) and indices to train the neural network.

**Arguments:**
- `x`:     `x` matrix
- `x_ind`: selected data indices (columns)
- `sigma`: (optional) number of standard deviations

**Returns:**
- `curriculum_ind`: indices for curriculum learning
- `nn_ind`:         indices for training the neural network
"""
function get_curriculum_ind(x, x_ind, sigma=1)
    curriculum_ind = []
    nn_ind = Set()
    for i in x_ind
        union!(nn_ind, Set(findall(v -> v < mean(x[:,i]) - sigma*std(x[:,i]) ||
                                        v > mean(x[:,i]) + sigma*std(x[:,i]), x[:,i])))
    end
    nn_ind = collect(nn_ind)
    for i in axes(x,1)
        if !(i in nn_ind)
            push!(curriculum_ind, i)
        end
    end      
    return (curriculum_ind, nn_ind)
end # function get_curriculum_ind

"""
    extract_TL_matrices(TL_coef, terms; Bt_scale=50000)

**Arguments**
- `TL_coef`:  Tolles-Lawson coefficients (must include `:permanent` and `:induced`)
- `terms`:    Tolles-Lawson terms used {`:permanent`,`:induced`,`:eddy`,`:bias`}
- `Bt_scale`: (optional) scaling factor for induced and eddy current terms [nT]

**Returns**
- `TL_coef_p`: `3`x`1` vector of permanent field coefficients
- `TL_coef_i`: `3`x`3` symmetric matrix of induced field coefficients, denormalized
- `TL_coef_e`: `3`x`3` matrix of eddy current coefficients, denormalized
"""
function extract_TL_matrices(TL_coef, terms; Bt_scale=50000f0)
    @assert any([:permanent,:p,:permanent3,:p3] .∈ (terms,)) "Permanent terms are required"
    @assert any([:induced,:i,:induced6,:i6] .∈ (terms,)) "Induced terms are required"
    @assert length(TL_coef) >= 9 "TL_coef must contain at least 9 elements"

    TL_coef_p = vec(TL_coef[1:3])

    TL_coef_i = Symmetric([TL_coef[4] TL_coef[5]/2 TL_coef[6]/2
                           0.0f32     TL_coef[7]   TL_coef[8]/2
                           0.0f32     0.0f32       TL_coef[9]  ] / Bt_scale, :U)

    TL_coef_e = I(3)
    if any([:eddy,:e,:eddy9,:e9] .∈ (terms,))
        TL_coef_e = [TL_coef[10] TL_coef[11] TL_coef[12]
                     TL_coef[13] TL_coef[14] TL_coef[15]
                     TL_coef[16] TL_coef[17] TL_coef[18]] / Bt_scale
    end

    return (TL_coef_p, TL_coef_i, TL_coef_e)
end # function extract_TL_matrices

"""
    get_linear_vector_aircraft_field(B_vec, B_vec_deriv, TL_coef_p, TL_coef_i, TL_coef_e)
    
**Arguments**
- `B_vec`:       `3`x`N` matrix of vector magnetometer measurements
- `B_vec_deriv`: `3`x`N` matrix of vector magnetometer measurement derivatives
- `TL_coef_p`:   `3`x`1` vector of permanent field coefficients
- `TL_coef_i`:   `3`x`3` symmetric matrix of induced field coefficients, denormalized
- `TL_coef_e`:   `3`x`3` matrix of eddy current coefficients, denormalized

**Returns**
- `TL_aircraft_field`: `3`x`N` matrix of vector aircraft field 
"""
function get_linear_vector_aircraft_field(B_vec, B_vec_deriv, TL_coef_p, TL_coef_i, TL_coef_e)
    perm_field = TL_coef_p 
    ind_field  = TL_coef_i * B_vec
    eddy_field = TL_coef_e * B_vec_deriv
    TL_aircraft_field = perm_field .+ ind_field + eddy_field
    return (TL_aircraft_field) # (3,batch_size)
end # function get_linear_vector_aircraft_field

"""
    nn_comp_3_train(A, B, B_dot, x, y, no_norm;
                    model_type::Symbol   = :m3s,
                    norm_type_A::Symbol  = :none,
                    norm_type_x::Symbol  = :standardize,
                    norm_type_y::Symbol  = :none,
                    TL_coef::Vector      = zeros(Float32,18),
                    terms                = [:p, :i, :e],
                    y_type               = [:d],
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
                    weights::Params      = Params([]),
                    l_segs::Vector       = [length(y)],
                    silent::Bool         = true)

Train neural network-based aeromagnetic compensation, model 3. Model 3 architectures retain
the Tolles-Lawson (T-L) terms in vector form, making it possible to remove the Taylor expansion 
approximation used for predicting the Earth field in the loss function, as well as create vector-based
neural network corrections. 

Note that this model is subject to change, and not all options are supported (e.g., no LBFGS, 
and generally you should not subtract the IGRF or diurnal fields for the y-value)

Currently, we recommend using a low-pass (not a band-pass) filter to preset the Tolles-Lawson
coefficients, as a band-pass filter removes the Earth field and leaves a large bias in the 
prediction of the aircraft model, eg.:
`tlc = create_TL_coef(getfield(xyz,use_vec), getfield(xyz,use_mag)-xyz.mag_1_c, TL_ind;
                      terms=terms, pass1=0.0, pass2=0.9)`


- Model info:
    - `Model 3tl`: Only fine-tuned Tolles-Lawson terms via SGD, without the Taylor expansion for Earth field targets
    - `Model 3s`:  Scalar-corrected T-L with expanded vector terms for explainability
    - `Model 3v`:  Vector-corrected T-L with expanded vector terms for explainability
    - `Model 3sc`: Scalar-corrected T-L with curriculum learning 
    - `Model 3vc`: Vector-corrected T-L with curriculum learning 
"""
function nn_comp_3_train(A, B, B_dot, x, y, no_norm;
                         model_type::Symbol   = :m3s,
                         norm_type_x::Symbol  = :standardize,
                         norm_type_y::Symbol  = :standardize,
                         TL_coef::Vector      = zeros(Float32,18),
                         terms                = [:p, :i, :e],
                         y_type               = :d,
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
                         weights::Params      = Params([]),
                         l_segs::Vector       = [length(y)],
                         silent::Bool         = true)

    @assert epoch_lbfgs == 0 "LBFGS not implemented for model 3"

    # convert to Float32 for ~50% speedup
    A = convert.(Float32,A) 
    B = convert.(Float32,B)         # magnitude of magnetic field (batch_size, 1)
    B_dot = convert.(Float32,B_dot) # time-derivative of unit vector (batch_size, 3)
    x = convert.(Float32,x)
    y = convert.(Float32,y)
    α = convert.(Float32,α_sgl)
    λ = convert.(Float32,λ_sgl)
    Bt_scale = 50000f0

    TL_coef = convert.(Float32,TL_coef)
    TL_coef_p, TL_coef_i, TL_coef_e = extract_TL_matrices(TL_coef, terms; Bt_scale=Bt_scale)
    add_curriculum = true
    
    B_unit       = A[:,1:3]     # Normalized vector magnetometer reading
    B_vec        = B_unit .* B  # The vector magnetometer to be used in TL
    B_vec_deriv  = B_dot        # This is not exactly true, but is internally consistent

    if y_type in [:c,:d] 
        y_type_aircraft = true
    elseif y_type in [:a,:b]
        y_type_aircraft = false
    else
        @error("Unsupported y_type in nn_comp_3_train, $y_type")
    end
    
    if sum(data_norms[end]) == 0 # normalize data
        (x_bias,x_scale,x_norm) = norm_sets(x;norm_type=norm_type_x,no_norm=no_norm)
        (y_bias,y_scale,y_norm) = norm_sets(y;norm_type=norm_type_y)
        if k_pca > 0
            if k_pca > size(x,2)
                k_pca = size(x,2)
                @info("reducing k_pca to $k_pca, size(x,2)")
            end
            (_,S,V) = svd(cov(x_norm))
            v_scale = V[:,1:k_pca]*inv(Diagonal(sqrt.(S[1:k_pca])))
            var_ret = round(sum(sqrt.(S[1:k_pca]))/sum(sqrt.(S))*100,digits=6)
            @info("k_pca = $k_pca of $(size(x,2)), variance retained: $var_ret %")
        else
            v_scale = I(size(x,2))
        end
        x_norm = x_norm * v_scale
    else # unpack data normalizations
        (_,_,v_scale,x_bias,x_scale,y_bias,y_scale) = unpack_data_norms(data_norms)
        x_norm = ((x .- x_bias) ./ x_scale) * v_scale
        y_norm =  (y .- y_bias) ./ y_scale
    end

    # separate into training and validation
    if frac_train < 1
        N = size(x_norm,1)
        p = randperm(N)
        N_train = floor(Int,frac_train*N)
        p_train = p[1:N_train]
        p_val   = p[N_train+1:end]
        B_vec_deriv_train = B_vec_deriv[p_train,:] 
        B_vec_deriv_val   = B_vec_deriv[p_val  ,:]
        B_unit_train    = B_unit[p_train,:]
        B_unit_val      = B_unit[p_val  ,:]
        B_vec_train     = B_vec[ p_train,:]
        B_vec_val       = B_vec[ p_val  ,:]
        x_norm_train    = x_norm[p_train,:]
        x_norm_val      = x_norm[p_val  ,:]
        y_norm_train    = y_norm[p_train,:]
        y_norm_val      = y_norm[p_val  ,:]

        if model_type in [:m3vc,:m3sc] && add_curriculum
            @info("Making curriculum")
            (curriculum_ind,_) = get_curriculum_ind(x_norm_train,[2,3,4,5])
            data_train = Flux.DataLoader((B_vec_deriv_train[curriculum_ind,:]',
                                          B_unit_train[curriculum_ind,:]',
                                           B_vec_train[curriculum_ind,:]',
                                          x_norm_train[curriculum_ind,:]',
                                          y_norm_train[curriculum_ind,:]'),
                                          shuffle=true,batchsize=batchsize)
            data_train_2 = Flux.DataLoader((B_vec_deriv_train',B_unit_train',
                                            B_vec_train',x_norm_train',y_norm_train'),
                                            shuffle=true,batchsize=batchsize)
        else
            data_train = Flux.DataLoader((B_vec_deriv_train',B_unit_train',B_vec_train',x_norm_train',y_norm_train'),
                                          shuffle=true,batchsize=batchsize)
        end
        data_val   = Flux.DataLoader((B_vec_deriv_val',B_unit_val',B_vec_val',x_norm_val',y_norm_val'),
                                      shuffle=true,batchsize=batchsize)
    else
        data_train = Flux.DataLoader((B_vec_deriv',B_unit',B_vec',x_norm',y_norm'),
                                      shuffle=true,batchsize=batchsize)
        data_val   = data_train
    end

    # setup NN
    xS = size(x_norm,2) # number of features
    if model_type in [:m3s,:m3tl,:m3sc]
        yS = 1 # additive correction
    elseif model_type in [:m3v,:m3vc]
        yS = 3
    end

    if weights == Params([]) # not re-training with known weights
       m = get_nn_m(xS,yS;hidden=hidden,activation=activation)
    else # initial weights already known
        TL_coef_p = weights[length(weights)-2]
        TL_coef_i = weights[length(weights)-1]/Bt_scale
        TL_coef_e = weights[length(weights)]/Bt_scale
        m = get_nn_m(Flux.params(weights[1:length(weights)-3]),activation)
    end

    ## setup optimizer and loss function
    if model_type in [:m3g]
        #TODO change these constants to be globally initialized
        opt = Scheduler(Step(λ = η_adam, γ = 0.8, step_sizes = [epoch_adam ÷ 2,epoch_adam ÷ 5, epoch_adam ÷ 5]), Adam())
    else
        opt = Adam(η_adam)
    end
   
    # Computes the scalar correction to the total field targeted (aircraft or Earth)
    function get_scalar_TL(B_vec_deriv, B_unit, B_vec) 
        TL_aircraft_field = get_linear_vector_aircraft_field(B_vec, B_vec_deriv, TL_coef_p, TL_coef_i, TL_coef_e)
        if y_type_aircraft # Need the amount to subtract from scalar mag
            yhat = sum(TL_aircraft_field .* B_unit, dims=1) # Dot product 
            #println("Aircraft correction=", yhat)
        else
            B_e = B_vec - TL_aircraft_field #(3,batch)
            yhat = sqrt.(sum(B_e.^2,dims=1)) # Magnitude of scalar Earth field
            #println("Aircraft correction=", sum(TL_aircraft_field .* B_unit, dims=1))
            #println("|Be|=", yhat)
        end
        return (yhat)
    end

    # Computes the TL vector and adds an NN(x) correction to that
    function get_vector_corrected_TL(B_vec_deriv_in, B_unit_in, B_vec_in, x_in, has_nn)
        y_TL  = get_linear_vector_aircraft_field(B_vec_in, B_vec_deriv_in, TL_coef_p, TL_coef_i, TL_coef_e)
        y_NN  = has_nn.*m(x_in)
        y_NN_scaled = y_scale .* y_NN # Scale factor keeps NN output near TL
        aircraft_field = y_TL + y_NN_scaled
        if y_type_aircraft # Need the amount to subtract from scalar mag
            yhat = sum(aircraft_field .* B_unit_in, dims=1) # Dot product 
            #println("Aircraft correction=", yhat)
        else
            B_e = B_vec_in - aircraft_field #(3,batch)
            yhat = sqrt.(sum(B_e.^2,dims=1)) # Magnitude of scalar Earth field
            #println("Aircraft correction=", sum(TL_aircraft_field .* B_unit_in, dims=1))
            #println("|Be|=", yhat)
        end
        return (yhat)
    end

    function loss_m3s(B_vec_deriv_l,B_unit_l,B_vec_l,x_l,y_l)  
        y_TL = get_scalar_TL(B_vec_deriv_l, B_unit_l, B_vec_l)
        y_TL_norm = vec((y_TL .- y_bias)./y_scale)  # Get into the same scale as NN output
        y_NN = vec(m(x_l))   # Computes a scalar correction to the TL correction
        return (Flux.mse(y_TL_norm+y_NN,vec(y_l)))
    end

    function loss_m3tl(B_vec_deriv_l,B_unit_l,B_vec_l,_,y_l) 
        y_TL = get_scalar_TL(B_vec_deriv_l, B_unit_l, B_vec_l)
        y_TL_norm = (y_TL .- y_bias)./y_scale 
        return (Flux.mse(vec(y_TL_norm),vec(y_l)))
    end
 
    function loss_m3v(B_vec_deriv_l,B_unit_l,B_vec_l,x_l,y_l;has_nn=1f0)
        yhat = get_vector_corrected_TL(B_vec_deriv_l, B_unit_l, B_vec_l, x_l, has_nn)
        yhat_norm = (yhat .- y_bias)./y_scale
        return (Flux.mse(vec(yhat_norm),vec(y_l)))
    end

    loss_m3s_λ(B_vec_deriv_l,B_unit_l,B_vec_l,x_l,y_l) = loss_m3s(B_vec_deriv_l,B_unit_l,B_vec_l,x_l,y_l) + λ*sum(sparse_group_lasso(m,α))
    loss_m3tl_λ(B_vec_deriv_l,B_unit_l,B_vec_l,x_l,y_l) = loss_m3tl(B_vec_deriv_l,B_unit_l,B_vec_l,x_l,y_l) + λ*sum(sparse_group_lasso(m,α))
    loss_m3v_λ(B_vec_deriv_l,B_unit_l,B_vec_l,x_l,y_l) = loss_m3v(B_vec_deriv_l,B_unit_l,B_vec_l,x_l,y_l) + λ*sum(sparse_group_lasso(m,α))
    
    #model_type == :m3s && (loss = λ > 0 ? loss_m3s_λ : loss_m3s)
    #model_type == :m3tl && (loss = λ > 0 ? loss_m3tl_λ : loss_m3tl)
    #model_type == :m3v && (loss = λ > 0 ? loss_m3v_λ : loss_m3v)

    if model_type in [:m3tl, :m3sc]
        loss = loss_m3tl
    elseif model_type == :m3s
        loss = loss_m3s
    elseif model_type == :m3v
        loss = loss_m3v
    elseif model_type == :m3vc
        loss(Bvp,Bh,Bv,xl,yl) = loss_m3v(Bvp,Bh,Bv,xl,yl;has_nn=0f0)
    end

    function loss_all(data_l)
        l = 0f0
       
        for (B_vec_deriv_l,B_unit_l,B_vec_l,x_l,y_l) in data_l
            l += loss(B_vec_deriv_l,B_unit_l,B_vec_l,x_l,y_l)
        end
        
        l/length(data_l)
    end # function loss_all

    # train NN with ADAM optimizer
    weights = deepcopy(Flux.params(m,TL_coef_p, TL_coef_i,TL_coef_e))
    best_loss = loss_all(data_val)

    @info("Epoch 0: loss = $best_loss")
    has_nn = 1f0 # Factor to keep/excluded NN contribution into y_hat calculation
    for i = 1:epoch_adam
        if model_type in [:m3tl,:m3s,:m3v] # train on NN weights + TL coef
            Flux.train!(loss,Flux.params(m,TL_coef_p,TL_coef_i.data,TL_coef_e),data_train,opt)
        elseif model_type in [:m3vc,:m3sc] #TODO change these constants to be globally initialized
            if i < epoch_adam * (1. / 10.)  
                pms = Flux.params(TL_coef_p)
            elseif i < epoch_adam * (2. / 10.)  
                pms = Flux.params(TL_coef_p,TL_coef_i.data)
            elseif i < epoch_adam * (3. / 10.)  
                pms = Flux.params(TL_coef_p,TL_coef_i.data,TL_coef_e)
            elseif i < epoch_adam * (6. / 10.)  
                pms = Flux.params(m)
                data_train = data_train_2
                if model_type == :m3vc
                    loss = loss_m3v
                elseif model_type == :m3sc
                    loss = loss_m3s
                end
            else
                pms = Flux.params(m,TL_coef_p,TL_coef_i.data,TL_coef_e)
            end
            Flux.train!(loss,pms,data_train,opt)
        end
        current_loss = loss_all(data_val)
        if current_loss < best_loss
            best_loss = current_loss
            weights = deepcopy(Flux.params(m,TL_coef_p,TL_coef_i*Bt_scale,TL_coef_e*Bt_scale))
        end
        mod(i,5) == 0 && @info("Epoch $i: loss = $best_loss")
        if mod(i,10) == 0
           if model_type in [:m3s,:m3sc]
                y_TL = get_scalar_TL(transpose(B_vec_deriv), transpose(B_unit), transpose(B_vec))
                y_TL_norm = vec((y_TL .- y_bias)./y_scale)  # Get into the same scale as NN output
                y_NN = vec(m(transpose(x_norm)))
                y_hat          = denorm_sets(y_bias,y_scale, (y_TL_norm + y_NN))
            elseif model_type in [:m3tl]
                y_hat = vec(get_scalar_TL(transpose(B_vec_deriv), transpose(B_unit), transpose(B_vec)))
            elseif model_type in [:m3v,:m3vc]
                y_hat          = vec(get_vector_corrected_TL(transpose(B_vec_deriv), transpose(B_unit), 
                                                     transpose(B_vec), transpose(x_norm),1f0))
            end
            err = err_segs(y_hat,y,l_segs;silent=silent)
            @info("$i train error: $(round(std(err),digits=2)) nT")
        end
    end

    Flux.loadparams!(m,weights)

    # set to test mode in case model uses batchnorm or dropout
    Flux.testmode!(m)

    # get results
    if model_type in [:m3s,:m3sc]
        y_TL = get_scalar_TL(transpose(B_vec_deriv), transpose(B_unit), transpose(B_vec))
        y_TL_norm = vec((y_TL .- y_bias)./y_scale)  # Get into the same scale as NN output
        y_NN = vec(m(transpose(x_norm)))
        y_hat          = denorm_sets(y_bias,y_scale, (y_TL_norm + y_NN))
    elseif model_type in [:m3tl]
        y_hat = vec(get_scalar_TL(transpose(B_vec_deriv), transpose(B_unit), transpose(B_vec)))
    elseif model_type in [:m3v,:m3vc]
        y_hat          = vec(get_vector_corrected_TL(transpose(B_vec_deriv), transpose(B_unit), 
                                                     transpose(B_vec), transpose(x_norm),1f0))
    end
    err = err_segs(y_hat,y,l_segs;silent=silent)
    @info("train error: $(round(std(err),digits=2)) nT")

    # pack data normalizations
    weights = deepcopy(Flux.params(m,TL_coef_p,TL_coef_i*Bt_scale,TL_coef_e*Bt_scale))
    data_norms = (zeros(1,1),zeros(1,1),v_scale,x_bias,x_scale,y_bias,y_scale)

    return (weights, data_norms, y_hat, err)
end # function nn_comp_3_train

"""
    nn_comp_3_test(A, x, y, data_norms::Tuple, weights::Params;
                   model_type::Symbol   = :m3s,
                   y_type               = :d,
                   activation::Function = swish,
                   l_segs::Vector       = [length(y)],
                   silent::Bool         = false)

Evaluate neural network-based aeromagnetic compensation, model 3 performance.
"""
function nn_comp_3_test(A, B, B_dot, x, y, data_norms::Tuple, weights::Params;
                        model_type::Symbol   = :m3s,
                        y_type               = :d,
                        activation::Function = swish,
                        l_segs::Vector       = [length(y)],
                        silent::Bool         = false)

    B_unit      = A[:,1:3]
    B_vec       = B_unit .* B 
    B_vec_deriv = B_dot 

    # unpack data normalizations
    (_,_,v_scale,x_bias,x_scale,y_bias,y_scale) = unpack_data_norms(data_norms)

    # We assume all the terms are stored, but they may be zero if they were not trained
    Bt_scale = 50000f0
    TL_coef_p = weights[length(weights)-2]
    TL_coef_i = weights[length(weights)-1]/Bt_scale
    TL_coef_e = weights[length(weights)]/Bt_scale
    weights   = Flux.params(weights[1:length(weights)-3])

    x_norm = ((x .- x_bias) ./ x_scale) * v_scale

    if y_type in [:c,:d] 
        y_type_aircraft = true
    elseif y_type in [:a,:b]
        y_type_aircraft = false
    else
        @error("Unsupported y_type in nn_comp_3_train, $y_type")
    end

    # set to test mode in case model uses batchnorm or dropout
    m = get_nn_m(weights,activation)
    Flux.testmode!(m)
    
    # Computes the scalar correction to the total field targeted (aircraft or Earth)
    function get_scalar_TL(B_vec_deriv, B_unit, B_vec)
        TL_aircraft_field = get_linear_vector_aircraft_field(B_vec, B_vec_deriv, TL_coef_p, TL_coef_i, TL_coef_e)
        if y_type_aircraft # Need the amount to subtract from scalar mag
            yhat = sum(TL_aircraft_field .* B_unit, dims=1) # Dot product 
            #println("Aircraft correction=", yhat)
        else
            B_e = B_vec - TL_aircraft_field #(3,batch)
            yhat = sqrt.(sum(B_e.^2,dims=1)) # Magnitude of scalar Earth field
            #println("Aircraft correction=", sum(TL_aircraft_field .* B_unit, dims=1))
            #println("|Be|=", yhat)
        end
        return (yhat)
    end

    # Computes the TL vector and adds an NN(x) correction to that
    function get_vector_corrected_TL(B_vec_deriv_in, B_unit_in, B_vec_in, x_in)
        y_TL  = get_linear_vector_aircraft_field(B_vec_in, B_vec_deriv_in, TL_coef_p, TL_coef_i, TL_coef_e)
        y_NN  = m(x_in)
        y_NN_scaled = y_scale .* y_NN # Scale factor keeps NN output near TL
        aircraft_field = y_TL + y_NN_scaled
        if y_type_aircraft # Need the amount to subtract from scalar mag
            yhat = sum(aircraft_field .* B_unit_in, dims=1) # Dot product 
        else
            B_e = B_vec_in - aircraft_field #(3,batch)
            yhat = sqrt.(sum(B_e.^2,dims=1)) # Magnitude of scalar Earth field
        end
        return (yhat)
    end

    # get results
    if model_type in [:m3s,:m3sc]
        y_TL = get_scalar_TL(transpose(B_vec_deriv), transpose(B_unit), transpose(B_vec))
        y_TL_norm = vec((y_TL .- y_bias)./y_scale)  # Get into the same scale as NN output
        y_NN = vec(m(transpose(x_norm)))
        y_hat          = denorm_sets(y_bias,y_scale, (y_TL_norm + y_NN))
        y_NN_scaled = y_NN.*y_scale .+ y_bias
    elseif model_type in [:m3tl]
        y_hat = vec(get_scalar_TL(transpose(B_vec_deriv), transpose(B_unit), transpose(B_vec)))
    elseif model_type in [:m3v,:m3vc]
        y_hat          = vec(get_vector_corrected_TL(transpose(B_vec_deriv), transpose(B_unit), 
                                                     transpose(B_vec), transpose(x_norm)))
    end
    err = err_segs(y_hat,y,l_segs;silent=silent)
    @info("test  error: $(round(std(err),digits=2)) nT")

    return (y_hat, err)
end # function nn_comp_3_test

"""
    plsr_fit(x, y, k::Int=size(x,2);
             l_segs::Vector = [length(y)],
             return_set::Bool = false,
             silent::Bool     = false)

Fit a multi-input, multi-output (MIMO) partial least squares regression (PLSR) 
model to data with a specified output dimension. PLSR is a type of regularized 
linear regression where the number of components controls the strength of the 
regularization.

**Arguments:**
- `x`:          input data
- `y`:          observed data
- `k`:          (optional) number of components
- `l_segs`:     (optional) vector of lengths of `lines`, sum(l_segs) == length(y)
- `return_set`: (optional) if true, return `coef_set` (size `nx` x `ny` x `k`)
- `silent`:     (optional) if true, no print outs

**Returns:**
- `weights`:    Tuple of PLSR-based `(coefficients, bias=0)`
- `data_norms`: Tuple of data normalizations, e.g. `(x_bias,x_scale,y_bias,y_scale)`
- `y_hat`:      predicted data
- `err`:        mean-corrected (per line) error
"""
function plsr_fit(x, y, k::Int=size(x,2);   # N x nx , N x ny , k
                  l_segs::Vector = [length(y)],
                  return_set::Bool = false,
                  silent::Bool     = false)

    # standardize data
    (x_bias,x_scale,x_norm) = norm_sets(x;norm_type=:standardize)
    (y_bias,y_scale,y_norm) = norm_sets(y;norm_type=:standardize)

    k = clamp(k,1,size(x_norm,2))

	nx       = size(x_norm,2)           # nx
    ny       = size(y_norm,2)           # ny
    x_temp   = sum(x_norm.^2)           # scalar
    y_temp   = sum(y_norm.^2)           # scalar
    p_out    = zeros(eltype(x),nx,k)    # nx x k
    q_out    = zeros(eltype(x),ny,k)    # ny x k
    u_out    = zeros(eltype(x),nx,k)    # nx x k
    coef_set = zeros(eltype(x),nx,ny,k) # nx x ny x k

    # covariance & cross-covariance matrices
	Cxx = cov(x_norm)                   # nx x nx
	Cyx = collect(cov(y_norm,x_norm))   # ny x nx

    for i = 1:k

        # unit vectors that maximize correlation between input & output scores
        (U,_,V) = svd(Cyx')      # nx x ny , _ , ny x ny
        u = U[:,1:1]             # nx                      # x'*y ./ norm(x'*y)
        v = V[:,1:1]             # ny

        # input & output scores, input & output loading vectors
        z = x_norm*u             # N
        r = y_norm*v             # N
        p = (Cxx*u) / (u'*Cxx*u) # nx                      # x'*z ./ norm(z)^2
        q = (Cyx*u) / (u'*Cxx*u) # ny                      # y'*z ./ norm(z)^2

        # deflated covariance & cross-covariance matrices
        Cxx = (I(nx) - p*u')*Cxx # nx x nx
        Cyx = Cyx*(I(nx) - u*p') # ny x nx

        # deflated input & output data
        x_norm = x_norm - z*p'   # N  x nx                 # x*(u*p')
        y_norm = y_norm - z*q'   # N  x ny                 # x*(u*q')

        p_out[:,i] = p           # nx x k
        q_out[:,i] = q           # ny x k
        u_out[:,i] = u           # nx x k

        if return_set
            coef_set[:,:,i] = u_out[:,1:i]*inv(p_out[:,1:i]'*u_out[:,1:i])*q_out[:,1:i]' # nx x ny x k
        end

    end

    return_set && return (coef_set)

    coef = vec(u_out*inv(p_out'*u_out)*q_out') # nx x ny
    bias = 0.0

    # input & output residue variance %
    silent || @info("input  residue variance = $(round(sum(x_norm.^2)/x_temp*100,digits=2)) %")
    silent || @info("output residue variance = $(round(sum(y_norm.^2)/y_temp*100,digits=2)) %")

    # get results
    y_hat_norm = norm_sets(x;norm_type=:standardize)[3]*coef .+ bias
    y_hat      = denorm_sets(y_bias,y_scale,y_hat_norm)
    err        = err_segs(y_hat,y,l_segs;silent=silent)
    @info("fit  error: $(round(std(err),digits=2)) nT")

    # pack data normalizations
    data_norms = (x_bias,x_scale,y_bias,y_scale)

    return ((coef, bias), data_norms, y_hat, err)
end # function plsr_fit

# note: the existing PLSR Julia package (PartialLeastSquaresRegressor.jl) gives  
# the exact same result, but takes 3x longer, requires more dependencies, has  
# issues working in src, and only provides output for a single k per evaluation
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
- `x`:      input data
- `y`:      observed data
- `α`:      (optional) ridge regression (`α=0`) vs Lasso (`α=1`) balancing parameter {0:1}
- `λ`:      (optional) elastic net parameter (otherwise determined with cross-validation), `-1` to ignore
- `l_segs`: (optional) vector of lengths of `lines`, sum(l_segs) == length(y)
- `silent`: (optional) if true, no print outs

**Returns:**
- `weights`:    Tuple of elastic net-based `(coefficients, bias)`
- `data_norms`: Tuple of data normalizations, e.g. `(x_bias,x_scale,y_bias,y_scale)`
- `y_hat`:      predicted data
- `err`:        mean-corrected (per line) error
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
    err        = err_segs(y_hat,y,l_segs;silent=silent)
    @info("fit  error: $(round(std(err),digits=2)) nT")

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
- `x`:           input data
- `y`:           observed data
- `trim`         (optional) number of elements to trim (e.g. due to bpf)
- `λ`:           (optional) ridge parameter
- `norm_type_x`: (optional) normalization for `x` matrix
- `norm_type_y`: (optional) normalization for `y` target vector
- `l_segs`:      (optional) vector of lengths of `lines`, sum(l_segs) == length(y)
- `silent`:      (optional) if true, no print outs

**Returns:**
- `weights`:    Tuple of linear regression `(coefficients, bias=0)`
- `data_norms`: Tuple of data normalizations, e.g. `(x_bias,x_scale,y_bias,y_scale)`
- `y_hat`:      predicted data
- `err`:        mean-corrected (per line) error
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

    # linear regression to get Tolles-Lawson coefficients
    coef = vec(linreg(y_norm[ind,:],x_norm[ind,:];λ=λ))
    bias = 0.0

    y_hat_norm = x_norm*coef .+ bias
    y_hat      = denorm_sets(y_bias,y_scale,y_hat_norm)
    err        = err_segs(y_hat,y,l_segs;silent=silent)
    @info("fit  error: $(round(std(err),digits=2)) nT")
    @info("note that error may be misleading if using bpf")

    # pack data normalizations
    data_norms = (x_bias,x_scale,y_bias,y_scale)

    return ((coef, bias), data_norms, y_hat, err)
end # function linear_fit

"""
    linear_test(x, y, data_norms::Tuple, weights;
                l_segs::Vector = [length(y)],
                silent::Bool   = false)

Evaluate linear model performance.

**Arguments:**
- `x`:          input data
- `y`:          observed data
- `data_norms`: Tuple of data normalizations, e.g. `(x_bias,x_scale,y_bias,y_scale)`
- `weights`:    Tuple of `(coefficients, bias)` or only `coefficients`
- `l_segs`:     (optional) vector of lengths of `lines`, sum(l_segs) == length(y)
- `silent`:     (optional) if true, no print outs

**Returns:**
- `y_hat`: predicted data
- `err`:   mean-corrected (per line) error
"""
function linear_test(x, y, data_norms::Tuple, weights;
                     l_segs::Vector = [length(y)],
                     silent::Bool   = false)

    # unpack weights
    (coef,bias) = weights

    # unpack data normalizations
    (x_bias,x_scale,y_bias,y_scale) = data_norms

    x_norm = (x .- x_bias) ./ x_scale

    # get results
    y_hat_norm = x_norm*coef .+ bias
    y_hat      = denorm_sets(y_bias,y_scale,y_hat_norm)
    err        = err_segs(y_hat,y,l_segs;silent=silent)
    @info("test error: $(round(std(err),digits=2)) nT")

    return (y_hat, err)
end # function linear_test

"""
    comp_train(xyz::XYZ, ind, mapS::MapS=MapS(zeros(1,1),[0.0],[0.0],0.0);
               comp_params::CompParams=NNCompParams(), silent::Bool=true)

Train an aeromagnetic compensation model.

**Arguments:**
- `xyz`:         `XYZ` flight data struct
- `ind`:         selected data indices
- `mapS`:        (optional) `MapS` scalar magnetic anomaly map struct, only used for `y_type = :b, :c`
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct, either:
    - `NNCompParams`:  neural network-based aeromagnetic compensation parameters struct
    - `LinCompParams`: linear aeromagnetic compensation parameters struct
- `silent`       (optional) if true, no print outs

**Returns:**
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct
- `y`:           observed data
- `y_hat`:       predicted data
- `err`:         compensation error
"""
function comp_train(xyz::XYZ, ind, mapS::MapS=MapS(zeros(1,1),[0.0],[0.0],0.0);
                    comp_params::CompParams=NNCompParams(), silent::Bool=true)

    seed!(2) # for reproducibility
    t0 = time()

    # unpack parameters
    if typeof(comp_params) <: NNCompParams
        @unpack features_setup, features_no_norm, model_type, y_type, use_mag, 
        use_vec, data_norms, weights, terms, terms_A, sub_diurnal, sub_igrf, 
        bpf_mag, norm_type_A, norm_type_x, norm_type_y, 
        TL_coef, η_adam, epoch_adam, epoch_lbfgs, hidden, activation, 
        batchsize, frac_train, α_sgl, λ_sgl, k_pca, 
        drop_fi, drop_fi_bson, drop_fi_csv, perm_fi, perm_fi_csv = comp_params
    elseif typeof(comp_params) <: LinCompParams
        @unpack features_setup, features_no_norm, model_type, y_type, use_mag, 
        use_vec, data_norms, weights, terms, terms_A, sub_diurnal, sub_igrf, 
        bpf_mag, norm_type_A, norm_type_x, norm_type_y, 
        k_plsr, λ_TL = comp_params
        drop_fi = false
        perm_fi = false
    end

    if (model_type in [:TL,:mod_TL]) & (y_type != :e)
        @info("forcing y_type = :e (BPF'd total field)")
        y_type = :e
    end
    if (model_type in [:map_TL]) & (y_type != :c)
        y_type  = :c
        @info("forcing y_type = :c (aircraft field #1, using map)")
    end

    # map values along trajectory (if needed)
    if y_type in [:b,:c]
        traj_alt = median(xyz.traj.alt[ind])
        mapS.alt > 0 && (mapS = upward_fft(mapS,traj_alt;α=200))
        itp_mapS = map_itp(mapS)
        map_val  = itp_mapS.(xyz.traj.lon[ind],xyz.traj.lat[ind])
    else
        map_val  = -1
    end

    # `A` matrix for selected vector magnetometer
    field_check(xyz,use_vec,MagV)
    if model_type == :mod_TL
        A = create_TL_A(getfield(xyz,use_vec),ind;
                        Bt=getfield(xyz,use_mag),terms=terms_A)
    elseif model_type == :map_TL
        A = create_TL_A(getfield(xyz,use_vec),ind;
                        Bt=map_val,terms=terms_A)
    else
        (A, B_t, B_dot) = create_TL_A(getfield(xyz,use_vec),ind;terms=terms_A, return_B=true)
    end
    fs = 1 / xyz.traj.dt
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

    y_hat = zero(y)
    err   = 10*y

    if drop_fi

        for i in axes(x,2)

            x_fi = x[:, axes(x,2) .!= i]

            # train model
            if model_type in [:m1]
                (weights,data_norms,y_hat_fi,err_fi) = 
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
                                    silent      = silent)
            elseif model_type in [:m2a,:m2b,:m2c,:m2d]
                (weights,data_norms,y_hat_fi,err_fi) = 
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
                                    silent      = silent)
            elseif model_type in [:m3s,:m3tl,:m3v,:m3vc,:m3sc]
                (weights,data_norms,y_hat_fi,err_fi) = 
                    nn_comp_3_train(A,B_t,B_dot,x_fi,y,no_norm;
                                    model_type  = model_type,
                                    norm_type_x = norm_type_x,
                                    norm_type_y = norm_type_y,
                                    TL_coef     = TL_coef,
                                    terms       = terms,
                                    y_type      = y_type,
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
            else
                error("$model_type model type not defined")
            end

            if std(err_fi) < std(err)
                y_hat = y_hat_fi
                err   = err_fi
            end

            comp_params = NNCompParams(comp_params,
                                       weights    = weights,
                                       data_norms = data_norms)
            @save drop_fi_bson*"_$i.bson" comp_params

        end

    else

        # train model
        if model_type in [:m1]
            (weights,data_norms,y_hat,err) = 
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
                                weights     = weights,
                                silent      = silent)
        elseif model_type in [:m2a,:m2b,:m2c,:m2d]
            (weights,data_norms,y_hat,err) = 
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
                                weights     = weights,
                                silent      = silent)
        elseif model_type in [:m3s,:m3tl,:m3v,:m3vc,:m3sc]
            (weights,data_norms,y_hat,err) = 
                nn_comp_3_train(A,B_t,B_dot,x,y,no_norm;
                                model_type  = model_type,
                                norm_type_x = norm_type_x,
                                norm_type_y = norm_type_y,
                                TL_coef     = TL_coef,
                                terms       = terms,
                                y_type      = y_type,
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
                                silent      = silent)
        elseif model_type in [:TL,:mod_TL,:map_TL]
            trim = model_type in [:TL,:mod_TL] ? 20 : 0
            (weights,data_norms,y_hat,err) = 
                linear_fit(A,y;
                           trim=trim, λ=λ_TL,
                           norm_type_x = norm_type_A,
                           norm_type_y = norm_type_y,
                           silent      = silent)
        elseif model_type in [:elasticnet]
            (weights,data_norms,y_hat,err) = 
                elasticnet_fit(x,y; silent=silent)
        elseif model_type in [:plsr]
            (weights,data_norms,y_hat,err) = 
                plsr_fit(x,y,k_plsr; silent=silent)
        else
            error("$model_type model type not defined")
        end

    end

    if typeof(comp_params) <: NNCompParams
        comp_params = NNCompParams(comp_params,
                                   data_norms = data_norms,
                                   weights    = weights)
    elseif typeof(comp_params) <: LinCompParams
        comp_params = LinCompParams(comp_params,
                                    data_norms = data_norms,
                                    weights    = weights)
    end

    print_time(time()-t0,1)

    return (comp_params, y, y_hat, err, features)
end # function comp_train

"""
    comp_train(xyz_array, ind, mapS::MapS=MapS(zeros(1,1),[0.0],[0.0],0.0);
               comp_params::CompParams=NNCompParams(), silent::Bool=true)

Train an aeromagnetic compensation model.

**Arguments:**
- `xyz_array`:   vector of `XYZ` flight data structs, use case is displaced trajectory
- `ind`:         selected data indices
- `mapS`:        (optional) `MapS` scalar magnetic anomaly map struct, only used for `y_type = :b, :c`
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct, either:
    - `NNCompParams`:  neural network-based aeromagnetic compensation parameters struct
    - `LinCompParams`: linear aeromagnetic compensation parameters struct
- `silent`       (optional) if true, no print outs

**Returns:**
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct
- `y`:           observed data
- `y_hat`:       predicted data
- `err`:         compensation error
"""
function comp_train(xyz_array::Vector{XYZ20{Int64,Float64}}, ind, mapS::MapS=MapS(zeros(1,1),[0.0],[0.0],0.0);
                    comp_params::CompParams=NNCompParams(), silent::Bool=true)

    seed!(2) # for reproducibility
    t0 = time()

    # unpack parameters
    if typeof(comp_params) <: NNCompParams
        @unpack features_setup, features_no_norm, model_type, y_type, use_mag, 
        use_vec, data_norms, weights, terms, terms_A, sub_diurnal, sub_igrf, 
        bpf_mag, norm_type_A, norm_type_x, norm_type_y, 
        TL_coef, η_adam, epoch_adam, epoch_lbfgs, hidden, activation, 
        batchsize, frac_train, α_sgl, λ_sgl, k_pca, 
        drop_fi, drop_fi_bson, drop_fi_csv, perm_fi, perm_fi_csv = comp_params
    elseif typeof(comp_params) <: LinCompParams
        @unpack features_setup, features_no_norm, model_type, y_type, use_mag, 
        use_vec, data_norms, weights, terms, terms_A, sub_diurnal, sub_igrf, 
        bpf_mag, norm_type_A, norm_type_x, norm_type_y, 
        k_plsr, λ_TL = comp_params
        drop_fi = false
        perm_fi = false
    end

    if (model_type in [:TL,:mod_TL]) & (y_type != :e)
        @info("forcing y_type = :e (BPF'd total field)")
        y_type = :e
    end
    if (model_type in [:map_TL]) & (y_type != :c)
        y_type  = :c
        @info("forcing y_type = :c (aircraft field #1, using map)")
    end

    # map values along trajectory (if needed)
    if y_type in [:b,:c]
        traj_alt = median(xyz.traj.alt[ind])
        mapS.alt > 0 && (mapS = upward_fft(mapS,traj_alt;α=200))
        itp_mapS = map_itp(mapS)
        map_val  = itp_mapS.(xyz.traj.lon[ind],xyz.traj.lat[ind])
    else
        map_val  = -1
    end

    # `A` matrix for selected vector magnetometer
    field_check(xyz_array[1],use_vec,MagV)
    if model_type == :mod_TL
        A = create_TL_A(getfield(xyz_array[1],use_vec),ind;
                        Bt=getfield(xyz_array[1],use_mag),terms=terms_A)
    elseif model_type == :map_TL
        A = create_TL_A(getfield(xyz_array[1],use_vec),ind;
                        Bt=map_val,terms=terms_A)
    else
        A = create_TL_A(getfield(xyz_array[1],use_vec),ind;terms=terms_A)
    end
    fs = 1 / xyz_array[1].traj.dt
    y_type == :e && bpf_data!(A;bpf=get_bpf(;fs=fs))

    # load data
    (x,no_norm,features) = get_x(xyz_array[1],ind,features_setup;
                                 features_no_norm = features_no_norm,
                                 terms            = terms,
                                 sub_diurnal      = sub_diurnal,
                                 sub_igrf         = sub_igrf,
                                 bpf_mag          = bpf_mag)

    y = get_y(xyz_array[1],ind,map_val;
              y_type      = y_type,
              use_mag     = use_mag,
              sub_diurnal = sub_diurnal,
              sub_igrf    = sub_igrf)
    y_hat = zero(y)
    err   = 10*y

    for xyz in xyz_array[2:end]
        (x_i,no_norm_i,_) = get_x(xyz,ind,features_setup;
                                  features_no_norm = features_no_norm,
                                  terms            = terms,
                                  sub_diurnal      = sub_diurnal,
                                  sub_igrf         = sub_igrf,
                                  bpf_mag          = bpf_mag)
    
        y_i = get_y(xyz,ind,map_val;
                    y_type      = y_type,
                    use_mag     = use_mag,
                    sub_diurnal = sub_diurnal,
                    sub_igrf    = sub_igrf)
        x = vcat(x, x_i)
        y = vcat(y, y_i)
        y_hat   = vcat(y_hat, zero(y))
        no_norm = vcat(no_norm, no_norm_i)

        if model_type == :mod_TL
            A_i = create_TL_A(getfield(xyz,use_vec),ind;
                              Bt=getfield(xyz,use_mag),terms=terms_A)
        elseif model_type == :map_TL
            A_i = create_TL_A(getfield(xyz,use_vec),ind;
                              Bt=map_val,terms=terms_A)
        else
            A_i = create_TL_A(getfield(xyz,use_vec),ind;terms=terms_A)
        end
        A = vcat(A, A_i)

    end

    if drop_fi

        for i in axes(x,2)

            x_fi = x[:, axes(x,2) .!= i]

            # train model
            if model_type in [:m1]
                (weights,data_norms,y_hat_fi,err_fi) = 
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
                                    silent      = silent)
            elseif model_type in [:m2a,:m2b,:m2c,:m2d]
                (weights,data_norms,y_hat_fi,err_fi) = 
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
                                    silent      = silent)
            else
                error("$model_type model type not defined")
            end

            if std(err_fi) < std(err)
                y_hat = y_hat_fi
                err   = err_fi
            end

            comp_params = NNCompParams(comp_params,
                                       weights    = weights,
                                       data_norms = data_norms)
            @save drop_fi_bson*"_$i.bson" comp_params

        end

    else

        # train model
        if model_type in [:m1]
            (weights,data_norms,y_hat,err) = 
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
                                weights     = weights,
                                silent      = silent)
        elseif model_type in [:m2a,:m2b,:m2c,:m2d]
            (weights,data_norms,y_hat,err) = 
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
                                weights     = weights,
                                silent      = silent)
        elseif model_type in [:TL,:mod_TL,:map_TL]
            trim = model_type in [:TL,:mod_TL] ? 20 : 0
            (weights,data_norms,y_hat,err) = 
                linear_fit(A,y;
                           trim=trim, λ=λ_TL,
                           norm_type_x = norm_type_A,
                           norm_type_y = norm_type_y,
                           silent      = silent)
        elseif model_type in [:elasticnet]
            (weights,data_norms,y_hat,err) = 
                elasticnet_fit(x,y; silent=silent)
        elseif model_type in [:plsr]
            (weights,data_norms,y_hat,err) = 
                plsr_fit(x,y,k_plsr; silent=silent)
        else
            error("$model_type model type not defined")
        end

    end

    if typeof(comp_params) <: NNCompParams
        comp_params = NNCompParams(comp_params,
                                   data_norms = data_norms,
                                   weights    = weights)
    elseif typeof(comp_params) <: LinCompParams
        comp_params = LinCompParams(comp_params,
                                    data_norms = data_norms,
                                    weights    = weights)
    end

    print_time(time()-t0,1)

    return (comp_params, y, y_hat, err, features)
end # function comp_train

"""
    comp_train(lines, df_line::DataFrame, df_flight::DataFrame,
               df_map::DataFrame, comp_params::CompParams=NNCompParams();
               silent::Bool=true)

Train an aeromagnetic compensation model.

**Arguments:**
- `lines`:        selected line number(s)
- `df_line`:      lookup table (DataFrame) of `lines`
- `df_flight`:    lookup table (DataFrame) of flight files
- `df_map`:       lookup table (DataFrame) of map files
- `comp_params`:  `CompParams` aeromagnetic compensation parameters struct, either:
    - `NNCompParams`:  neural network-based aeromagnetic compensation parameters struct
    - `LinCompParams`: linear aeromagnetic compensation parameters struct
- `silent`        (optional) if true, no print outs
- `reorient_vec`: (optional) if true, align vector magnetometer with body frame

**Returns:**
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct
- `y`:           observed data
- `y_hat`:       predicted data
- `err`:         mean-corrected (per line) compensation error
"""
function comp_train(lines, df_line::DataFrame, df_flight::DataFrame,
                    df_map::DataFrame, comp_params::CompParams=NNCompParams();
                    silent::Bool=true, reorient_vec=false)

    seed!(2) # for reproducibility
    t0 = time()

    # unpack parameters
    if typeof(comp_params) <: NNCompParams
        @unpack features_setup, features_no_norm, model_type, y_type, use_mag, 
        use_vec, data_norms, weights, terms, terms_A, sub_diurnal, sub_igrf, 
        bpf_mag, norm_type_A, norm_type_x, norm_type_y, 
        TL_coef, η_adam, epoch_adam, epoch_lbfgs, hidden, activation, 
        batchsize, frac_train, α_sgl, λ_sgl, k_pca, 
        drop_fi, drop_fi_bson, drop_fi_csv, perm_fi, perm_fi_csv = comp_params
    elseif typeof(comp_params) <: LinCompParams
        @unpack features_setup, features_no_norm, model_type, y_type, use_mag, 
        use_vec, data_norms, weights, terms, terms_A, sub_diurnal, sub_igrf, 
        bpf_mag, norm_type_A, norm_type_x, norm_type_y, 
        k_plsr, λ_TL = comp_params
        drop_fi = false
        perm_fi = false
    end

    if (model_type in [:TL,:mod_TL]) & (y_type != :e)
        @info("forcing y_type = :e (BPF'd total field)")
        y_type = :e
    end
    if (model_type in [:map_TL]) & (y_type != :c)
        y_type  = :c
        @info("forcing y_type = :c (aircraft field #1, using map)")
    end

    mod_TL = model_type == :mod_TL ? true : false
    map_TL = model_type == :map_TL ? true : false

    # load data
    if model_type in [:m3s,:m3tl,:m3v,:m3vc,:m3sc]
        (A,B_t,B_dot,x,y,no_norm,features,l_segs) = get_Axy(lines,df_line,df_flight,df_map,
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
                                                            mod_TL           = mod_TL,
                                                            map_TL           = map_TL,
                                                            silent           = true,
                                                            return_B         = true,
                                                            reorient_vec     = reorient_vec)
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
                                                  mod_TL           = mod_TL,
                                                  map_TL           = map_TL,
                                                  silent           = true,
                                                  reorient_vec     = reorient_vec)
    end

    y_hat = zero(y)
    err   = 10*y

    if drop_fi

        for i in axes(x,2)

            x_fi = x[:, axes(x,2) .!= i]

            # train model
            if model_type in [:m1]
                (weights,data_norms,y_hat_fi,err_fi) = 
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
                (weights,data_norms,y_hat_fi,err_fi) = 
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
            elseif model_type in [:m3s,:m3tl,:m3v,:m3vc,:m3sc,]
                (weights,data_norms,y_hat_fi,err_fi) = 
                    nn_comp_3_train(A,B_t,B_dot,x_fi,y,no_norm;
                                    model_type  = model_type,
                                    norm_type_x = norm_type_x,
                                    norm_type_y = norm_type_y,
                                    TL_coef     = TL_coef,
                                    terms       = terms,
                                    y_type      = y_type,
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
            else
                error("$model_type model type not defined")
            end

            if std(err_fi) < std(err)
                y_hat = y_hat_fi
                err   = err_fi
            end

            comp_params = NNCompParams(comp_params,
                                       weights    = weights,
                                       data_norms = data_norms)
            @save drop_fi_bson*"_$i.bson" comp_params

        end

    else

        # train model
        if model_type in [:m1]
            (weights,data_norms,y_hat,err) = 
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
                                weights     = weights,
                                l_segs      = l_segs,
                                silent      = silent)
        elseif model_type in [:m2a,:m2b,:m2c,:m2d]
            (weights,data_norms,y_hat,err) = 
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
                                weights     = weights,
                                l_segs      = l_segs,
                                silent      = silent)
        elseif model_type in [:m3s,:m3tl,:m3v,:m3vc,:m3sc]
            (weights,data_norms,y_hat,err) = 
                nn_comp_3_train(A,B_t,B_dot,x,y,no_norm;
                                model_type  = model_type,
                                norm_type_x = norm_type_x,
                                norm_type_y = norm_type_y,
                                terms       = terms,
                                y_type      = y_type,
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
                                weights     = weights,
                                l_segs      = l_segs,
                                silent      = silent)
        elseif model_type in [:TL,:mod_TL,:map_TL]
            trim = model_type in [:TL,:mod_TL] ? 20 : 0
            (weights,data_norms,y_hat,err) = 
                linear_fit(A,y;
                           trim=trim, λ=λ_TL,
                           norm_type_x = norm_type_A,
                           norm_type_y = norm_type_y,
                           l_segs      = l_segs,
                           silent      = silent)
        elseif model_type in [:elasticnet]
            (weights,data_norms,y_hat,err) = 
                elasticnet_fit(x,y;
                               l_segs      = l_segs,
                               silent      = silent)
        elseif model_type in [:plsr]
            (weights,data_norms,y_hat,err) = 
                plsr_fit(x,y,k_plsr;
                         l_segs      = l_segs,
                         silent      = silent)
        else
            error("$model_type model type not defined")
        end

    end

    if typeof(comp_params) <: NNCompParams
        comp_params = NNCompParams(comp_params,
                                   data_norms = data_norms,
                                   weights    = weights)
    elseif typeof(comp_params) <: LinCompParams
        comp_params = LinCompParams(comp_params,
                                    data_norms = data_norms,
                                    weights    = weights)
    end

    print_time(time()-t0,1)

    return (comp_params, y, y_hat, err, features)
end # function comp_train

"""
    comp_test(xyz::XYZ, ind, mapS::MapS=MapS(zeros(1,1),[0.0],[0.0],0.0);
              comp_params::CompParams=NNCompParams(), silent::Bool=false)

Evaluate aeromagnetic compensation model performance.

**Arguments:**
- `xyz`:  `XYZ` flight data struct
- `ind`:  selected data indices
- `mapS`: (optional) `MapS` scalar magnetic anomaly map struct, only used for `y_type = :b, :c`
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct, either:
    - `NNCompParams`:  neural network-based aeromagnetic compensation parameters struct
    - `LinCompParams`: linear aeromagnetic compensation parameters struct
- `silent`       (optional) if true, no print outs

**Returns:**
- `y`:     observed data
- `y_hat`: predicted data
- `err`:   compensation error
"""
function comp_test(xyz::XYZ, ind, mapS::MapS=MapS(zeros(1,1),[0.0],[0.0],0.0);
                   comp_params::CompParams=NNCompParams(), silent::Bool=false)

    seed!(2) # for reproducibility
    t0 = time()

    # unpack parameters
    if typeof(comp_params) <: NNCompParams
        @unpack features_setup, features_no_norm, model_type, y_type, use_mag, 
        use_vec, data_norms, weights, terms, terms_A, sub_diurnal, sub_igrf, 
        bpf_mag, norm_type_A, norm_type_x, norm_type_y, 
        TL_coef, η_adam, epoch_adam, epoch_lbfgs, hidden, activation, 
        batchsize, frac_train, α_sgl, λ_sgl, k_pca, 
        drop_fi, drop_fi_bson, drop_fi_csv, perm_fi, perm_fi_csv = comp_params
    elseif typeof(comp_params) <: LinCompParams
        @unpack features_setup, features_no_norm, model_type, y_type, use_mag, 
        use_vec, data_norms, weights, terms, terms_A, sub_diurnal, sub_igrf, 
        bpf_mag, norm_type_A, norm_type_x, norm_type_y, 
        k_plsr, λ_TL = comp_params
        drop_fi = false
        perm_fi = false
    end

    if y_type == :e
        @info("forcing y_type = :d (Δmag)")
        y_type = :d
    end
    if (model_type in [:TL,:mod_TL]) & (y_type != :d)
        @info("forcing y_type = :d (Δmag)")
        y_type = :d
    end
    if (model_type in [:map_TL]) & (y_type != :c)
        y_type  = :c
        @info("forcing y_type = :c (aircraft field #1, using map)")
    end

    # map values along trajectory (if needed)
    if y_type in [:b,:c]
        traj_alt = median(xyz.traj.alt[ind])
        mapS.alt > 0 && (mapS = upward_fft(mapS,traj_alt;α=200))
        itp_mapS = map_itp(mapS)
        map_val  = itp_mapS.(xyz.traj.lon[ind],xyz.traj.lat[ind])
    else
        map_val  = -1
    end

    # `A` matrix for selected vector magnetometer
    field_check(xyz,use_vec,MagV)
    if model_type == :mod_TL
        A = create_TL_A(getfield(xyz,use_vec),ind;
                        Bt=getfield(xyz,use_mag),terms=terms_A)
    elseif model_type == :map_TL
        A = create_TL_A(getfield(xyz,use_vec),ind;
                        Bt=map_val,terms=terms_A)
    else
        (A, B_t, B_dot) = create_TL_A(getfield(xyz,use_vec),ind;terms=terms_A, return_B=true)
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

    y_hat = zero(y)
    err   = 10*y

    if drop_fi | perm_fi

        for i in axes(x,2)

            if perm_fi
                x_fi = deepcopy(x)
                x_fi[:,i] .= x[randperm(size(x,1)),i]
                fi_csv = perm_fi_csv
            elseif drop_fi
                x_fi = x[:, axes(x,2) .!= i]
                df_i = drop_fi_bson*"_$i.bson"
                comp_params = load(df_i,@__MODULE__)[:comp_params]
                data_norms  = comp_params.data_norms
                weights     = comp_params.weights
                fi_csv      = drop_fi_csv
            end

            # evaluate model
            if model_type in [:m1]
                (y_hat_fi,err_fi) = nn_comp_1_test(x_fi,y,data_norms,weights;
                                                   activation = activation,
                                                   silent     = silent)
            elseif model_type in [:m2a,:m2b,:m2c,:m2d]
                (y_hat_fi,err_fi) = nn_comp_2_test(A,x_fi,y,data_norms,weights;
                                                   model_type = model_type,
                                                   activation = activation,
                                                   silent     = silent)
            elseif model_type in [:m3s,:m3tl,:m3v,:m3vc,:m3sc]
                (y_hat_fi,err_fi) = nn_comp_3_test(A,B_t,B_dot,x_fi,y,data_norms,weights;
                                                   model_type = model_type,
                                                   y_type     = y_type,
                                                   activation = activation,
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

        @info("returning best feature importance results")

    else

        # evaluate model
        if model_type in [:m1]
            (y_hat,err) = nn_comp_1_test(x,y,data_norms,weights;
                                         activation = activation,
                                         silent     = silent)
        elseif model_type in [:m2a,:m2b,:m2c,:m2d]
            (y_hat,err) = nn_comp_2_test(A,x,y,data_norms,weights;
                                         model_type = model_type,
                                         activation = activation,
                                         silent     = silent)
        elseif model_type in [:m3s,:m3tl,:m3v,:m3vc,:m3sc]
            (y_hat,err) = nn_comp_3_test(A,B_t,B_dot,x,y,data_norms,weights;
                                         model_type = model_type,
                                         y_type     = y_type,
                                         activation = activation,
                                         silent     = silent)
        elseif model_type in [:TL,:mod_TL,:map_TL]
            (y_hat,err) = linear_test(A,y,data_norms,weights;
                                         silent     = silent)
        elseif model_type in [:elasticnet,:plsr]
            (y_hat,err) = linear_test(x,y,data_norms,weights;
                                         silent     = silent)
        else
            error("$model_type model type not defined")
        end

    end

    print_time(time()-t0,1)

    return (y, y_hat, err, features)
end # function comp_test

"""
    comp_test(lines, df_line::DataFrame, df_flight::DataFrame,
              df_map::DataFrame, comp_params::CompParams=NNCompParams();
              silent::Bool=false)

Evaluate aeromagnetic compensation model performance.

**Arguments:**
- `lines`:        selected line number(s)
- `df_line`:      lookup table (DataFrame) of `lines`
- `df_flight`:    lookup table (DataFrame) of flight files
- `df_map`:       lookup table (DataFrame) of map files
- `comp_params`:  `CompParams` aeromagnetic compensation parameters struct
- `comp_params`:  `CompParams` aeromagnetic compensation parameters struct, either:
    - `NNCompParams`:  neural network-based aeromagnetic compensation parameters struct
    - `LinCompParams`: linear aeromagnetic compensation parameters struct
- `silent`        (optional) if true, no print outs
- `reorient_vec`: (optional) if true, align vector magnetometer with body frame

**Returns:**
- `y`:     observed data
- `y_hat`: predicted data
- `err`:   mean-corrected (per line) compensation error
"""
function comp_test(lines, df_line::DataFrame, df_flight::DataFrame,
                   df_map::DataFrame, comp_params::CompParams=NNCompParams();
                   silent::Bool=false, reorient_vec::Bool=false)

    seed!(2) # for reproducibility
    t0 = time()

    # unpack parameters
    if typeof(comp_params) <: NNCompParams
        @unpack features_setup, features_no_norm, model_type, y_type, use_mag, 
        use_vec, data_norms, weights, terms, terms_A, sub_diurnal, sub_igrf, 
        bpf_mag, norm_type_A, norm_type_x, norm_type_y, 
        TL_coef, η_adam, epoch_adam, epoch_lbfgs, hidden, activation, 
        batchsize, frac_train, α_sgl, λ_sgl, k_pca, 
        drop_fi, drop_fi_bson, drop_fi_csv, perm_fi, perm_fi_csv = comp_params
    elseif typeof(comp_params) <: LinCompParams
        @unpack features_setup, features_no_norm, model_type, y_type, use_mag, 
        use_vec, data_norms, weights, terms, terms_A, sub_diurnal, sub_igrf, 
        bpf_mag, norm_type_A, norm_type_x, norm_type_y, 
        k_plsr, λ_TL = comp_params
        drop_fi = false
        perm_fi = false
    end

    if y_type == :e
        @info("forcing y_type = :d (Δmag)")
        y_type = :d
    end
    if (model_type in [:TL,:mod_TL]) & (y_type != :d)
        @info("forcing y_type = :d (Δmag)")
        y_type = :d
    end
    if (model_type in [:map_TL]) & (y_type != :c)
        y_type  = :c
        @info("forcing y_type = :c (aircraft field #1, using map)")
    end

    mod_TL = model_type == :mod_TL ? true : false
    map_TL = model_type == :map_TL ? true : false

    # load data
    if model_type in [:m3s,:m3tl,:m3v,:m3vc,:m3sc]
        (A,nn_B,nn_B_dot,x,y,_,features,l_segs) = get_Axy(lines,df_line,df_flight,df_map,
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
                                                          mod_TL           = mod_TL,
                                                          map_TL           = map_TL,
                                                          silent           = true,
                                                          reorient_vec     = reorient_vec,
                                                          return_B         = true)
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
                                            mod_TL           = mod_TL,
                                            map_TL           = map_TL,
                                            reorient_vec     = reorient_vec,
                                            silent           = true)
    end

    y_hat = zero(y)
    err   = 10*y

    if drop_fi | perm_fi

        for i in axes(x,2)

            if perm_fi
                x_fi = deepcopy(x)
                x_fi[:,i] .= x[randperm(size(x,1)),i]
                fi_csv = perm_fi_csv
            elseif drop_fi
                x_fi = x[:, axes(x,2) .!= i]
                df_i = drop_fi_bson*"_$i.bson"
                comp_params = load(df_i,@__MODULE__)[:comp_params]
                data_norms  = comp_params.data_norms
                weights     = comp_params.weights
                fi_csv      = drop_fi_csv
            end

            # evaluate model
            if model_type in [:m1]
                (y_hat_fi,err_fi) = nn_comp_1_test(x_fi,y,data_norms,weights;
                                                   activation = activation,
                                                   l_segs     = l_segs,
                                                   silent     = silent)
            elseif model_type in [:m2a,:m2b,:m2c,:m2d]
                (y_hat_fi,err_fi) = nn_comp_2_test(A,x_fi,y,data_norms,weights;
                                                   model_type = model_type,
                                                   activation = activation,
                                                   l_segs     = l_segs,
                                                   silent     = silent)
            elseif model_type in [:m3s,:m3tl,:m3v,:m3vc,:m3sc]
                (y_hat_fi,err_fi) = nn_comp_3_test(A,nn_B,nn_B_dot,x_fi,y,data_norms,weights;
                                                   model_type = model_type,
                                                   y_type     = y_type,
                                                   activation = activation,
                                                   l_segs     = l_segs,
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

        @info("returning best feature importance results")

    else

        # evaluate model
        if model_type in [:m1]
            (y_hat,err) = nn_comp_1_test(x,y,data_norms,weights;
                                         activation = activation,
                                         l_segs     = l_segs,
                                         silent     = silent)
        elseif model_type in [:m2a,:m2b,:m2c,:m2d]
            (y_hat,err) = nn_comp_2_test(A,x,y,data_norms,weights;
                                         model_type = model_type,
                                         activation = activation,
                                         l_segs     = l_segs,
                                         silent     = silent)
        elseif model_type in [:m3s,:m3tl,:m3v,:m3vc,:m3sc]
            (y_hat,err) = nn_comp_3_test(A,nn_B,nn_B_dot,x,y,data_norms,weights;
                                         model_type = model_type,
                                         y_type     = y_type,
                                         activation = activation,
                                         l_segs     = l_segs,
                                         silent     = silent)
        elseif model_type in [:TL,:mod_TL,:map_TL]
            (y_hat,err) = linear_test(A,y,data_norms,weights;
                                         l_segs     = l_segs,
                                         silent     = silent)
        elseif model_type in [:elasticnet,:plsr]
            (y_hat,err) = linear_test(x,y,data_norms,weights;
                                         l_segs     = l_segs,
                                         silent     = silent)
        else
            error("$model_type model type not defined")
        end

    end

    print_time(time()-t0,1)

    return (y, y_hat, err, features)
end # function comp_test

"""
    comp_m2bc_test(lines, df_line::DataFrame,
                   df_flight::DataFrame, df_map::DataFrame,
                   comp_params::NNCompParams=NNCompParams();
                   silent::Bool=false)

Evaluate neural network-based aeromagnetic compensation, model 2b or 2c 
performance with additional outputs.

**Arguments:**
- `lines`:        selected line number(s)
- `df_line`:      lookup table (DataFrame) of `lines`
- `df_flight`:    lookup table (DataFrame) of flight files
- `df_map`:       lookup table (DataFrame) of map files
- `comp_params`:  `CompParams` aeromagnetic compensation parameters struct, either:
    - `NNCompParams`:  neural network-based aeromagnetic compensation parameters struct
    - `LinCompParams`: linear aeromagnetic compensation parameters struct
- `silent`        (optional) if true, no print outs
- `reorient_vec`: (optional) if true, align vector magnetometer with body frame

**Returns:**
- `y_nn`:  neural network compensation portion
- `y_TL`:  Tolles-Lawson  compensation portion
- `y`:     observed data
- `y_hat`: predicted data
- `err`:   mean-corrected (per line) compensation error
"""
function comp_m2bc_test(lines, df_line::DataFrame,
                        df_flight::DataFrame, df_map::DataFrame,
                        comp_params::NNCompParams=NNCompParams();
                        silent::Bool=false, reorient_vec::Bool=true)

    seed!(2) # for reproducibility
    t0 = time()

    # unpack parameters
    @unpack features_setup, features_no_norm, model_type, y_type, use_mag, 
    use_vec, data_norms, weights, terms, terms_A, sub_diurnal, sub_igrf, 
    bpf_mag, norm_type_A, norm_type_x, norm_type_y, 
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
                                        silent           = true,
                                        reorient_vec     = reorient_vec)

    # unpack data normalizations
    (A_bias,A_scale,v_scale,x_bias,x_scale,y_bias,y_scale) = 
        unpack_data_norms(data_norms)

    TL_coef = vec(weights[length(weights)])/y_scale
    weights = Flux.params(weights[1:length(weights)-1])

    A_norm =  (A .- A_bias) ./ A_scale
    x_norm = ((x .- x_bias) ./ x_scale) * v_scale

    # set to test mode in case model uses batchnorm or dropout
    m = get_nn_m(weights,activation)
    Flux.testmode!(m)

    y_nn  = vec(m(x_norm'))     .* y_scale
    y_TL  = vec(A_norm*TL_coef) .* y_scale
    y_hat = denorm_sets(y_bias,one.(y_scale),y_nn+y_TL)
    err   = err_segs(y_hat,y,l_segs;silent=silent)
    @info("std    y_nn: $(round(std(y_nn),digits=2)) nT")
    @info("std    y_TL: $(round(std(y_TL),digits=2)) nT")
    @info("test  error: $(round(std(err),digits=2)) nT")

    print_time(time()-t0,1)

    return (y_nn, y_TL, y, y_hat, err, features)
end # function comp_m2bc_test


"""
    comp_m3_test(lines, df_line::DataFrame,
                 df_flight::DataFrame, df_map::DataFrame,
                 comp_params::NNCompParams=NNCompParams();
                 silent::Bool=false)

Evaluate neural network-based aeromagnetic compensation, model 3 
performance with additional outputs for explainability.

**Arguments:**
- `lines`:        selected line number(s)
- `df_line`:      lookup table (DataFrame) of `lines`
- `df_flight`:    lookup table (DataFrame) of flight files
- `df_map`:       lookup table (DataFrame) of map files
- `comp_params`:  `CompParams` aeromagnetic compensation parameters struct, either:
    - `NNCompParams`:  neural network-based aeromagnetic compensation parameters struct
- `silent`        (optional) if true, no print outs
- `reorient_vec`: (optional) if true, align vector magnetometer with body frame

**Returns:**
- `TL_perm`:     TL permanent vector field
- `TL_induced`:  TL induced vector field
- `TL_eddy`:     TL eddy vector field
- `TL_aircraft`: TL aircraft field
- `B_vec`:       Vector magnetometer information
- `B_unit`:      Unit vector for magnetometer data
- `y_NN`:        vector neural network correction (for scalar models, in direction of Bt) 
- `vec_aircraft` Estimated vector aircraft field
- `y`:           observed scalar magnetometer data
- `y_hat`:       predicted scalar magnetometer data
- `err`:         mean-corrected (per line) compensation error
"""
function comp_m3_test(lines, df_line::DataFrame,
                      df_flight::DataFrame, df_map::DataFrame,
                      comp_params::NNCompParams=NNCompParams();
                      silent::Bool=false, reorient_vec::Bool=true)

    seed!(2) # for reproducibility
    t0 = time()

    # unpack parameters
    @unpack features_setup, features_no_norm, model_type, y_type, use_mag, 
    use_vec, data_norms, weights, terms, terms_A, sub_diurnal, sub_igrf, 
    bpf_mag, norm_type_A, norm_type_x, norm_type_y, 
    TL_coef, η_adam, epoch_adam, epoch_lbfgs, hidden, activation, 
    batchsize, frac_train, α_sgl, λ_sgl, k_pca, 
    drop_fi, drop_fi_bson, drop_fi_csv, perm_fi, perm_fi_csv = comp_params

    mod_TL = model_type == :mod_TL ? true : false
    map_TL = model_type == :map_TL ? true : false

    (A,B,B_dot,x,y,_,features,l_segs) = get_Axy(lines,df_line,df_flight,df_map,
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
                                                mod_TL           = mod_TL,
                                                map_TL           = map_TL,
                                                silent           = silent,
                                                reorient_vec     = reorient_vec,
                                                return_B         = true)

    B_unit      = A[:,1:3]
    B_vec       = B_unit.*B
    B_unit      = transpose(B_unit)
    B_vec       = transpose(B_vec)
    B_vec_deriv = transpose(B_dot)

    # unpack data normalizations
    (_,_,v_scale,x_bias,x_scale,y_bias,y_scale) = unpack_data_norms(data_norms)
    x_norm = ((x .- x_bias) ./ x_scale) * v_scale
    x_norm = transpose(x_norm)

    # We assume all the terms are stored, but they may be zero if they were not trained
    Bt_scale = 50000f0
    TL_coef_p = weights[length(weights)-2]
    TL_coef_i = weights[length(weights)-1]/Bt_scale
    TL_coef_e = weights[length(weights)]/Bt_scale
    weights   = Flux.params(weights[1:length(weights)-3])

    y_type_aircraft = true
    if y_type in [:c,:d] 
        y_type_aircraft = true
    elseif y_type in [:a,:b]
        y_type_aircraft = false
    else
        @error("Unsupported y_type for Model 3")
    end

    # set to test mode in case model uses batchnorm or dropout
    m = get_nn_m(weights,activation)
    Flux.testmode!(m)
    
    # Calculate T-L vector field
    TL_perm    = TL_coef_p .* ones(size(B_vec))
    TL_induced = TL_coef_i * B_vec
    TL_eddy    = TL_coef_e * B_vec_deriv
    TL_aircraft_field = TL_perm + TL_induced + TL_eddy

    # Compute Neural Network correction 
    y_NN = zeros(size(B_vec))
    if model_type in [:m3s, :m3sc]
       # scalar-corrected, dimensionless; rescale to TL and assume same direction
       y_NN = vec(y_scale.*m(x_norm))
       y_NN = transpose(y_NN .* transpose(B_unit))
    elseif model_type in [:m3v, :m3vc]
       y_NN = y_scale.*m(x_norm)  # vector-corrected 3xN, just rescale
    else
        @error("Unsupported model designation for Model 3 explainability")
    end

    # Compute the target field (aircraft or Earth)
    vec_aircraft = TL_aircraft_field + y_NN
    if y_type_aircraft
        y_hat = vec(sum(vec_aircraft .* B_unit, dims=1))
    else 
        Be = B_vec - vec_aircraft
        y_hat = vec(sqrt.(sum(Be.^2,dims=1)))
    end

    err = err_segs(y_hat,y,l_segs;silent=silent)
    @info("test  error: $(round(std(err),digits=2)) nT") 

    print_time(time()-t0,1)

    return (TL_perm, TL_induced, TL_eddy, TL_aircraft_field, 
            B_vec, B_unit, y_NN, vec_aircraft, y, y_hat, err)
end # function comp_m3_test

"""
    comp_train_test(xyz_train::XYZ, xyz_test::XYZ, ind_train, ind_test,
                    mapS_train::MapS = MapS(zeros(1,1),[0.0],[0.0],0.0),
                    mapS_test::MapS  = MapS(zeros(1,1),[0.0],[0.0],0.0);
                    comp_params::CompParams=NNCompParams(),
                    silent::Bool=true)

Train and evaluate aeromagnetic compensation model performance.

**Arguments:**
- `xyz_train`:   `XYZ` flight data struct for training
- `xyz_test`:    `XYZ` flight data struct for testing
- `ind_train`:   selected data indices for training
- `ind_test`:    selected data indices for testing
- `mapS_train`:  (optional) `MapS` scalar magnetic anomaly map struct for training, only used for `y_type = :b, :c`
- `mapS_test`:   (optional) `MapS` scalar magnetic anomaly map struct for testing,  only used for `y_type = :b, :c`
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct, either:
    - `NNCompParams`:  neural network-based aeromagnetic compensation parameters struct
    - `LinCompParams`: linear aeromagnetic compensation parameters struct
- `silent`:       (optional) if true, no print outs

**Returns:**
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct
- `y_train`:     training observed data
- `y_train_hat`: training predicted data
- `err_train`:   training compensation error
- `y_test`:      testing observed data
- `y_test_hat`:  testing predicted data
- `err_test`:    testing compensation error
"""
function comp_train_test(xyz_train::XYZ, xyz_test::XYZ, ind_train, ind_test,
                         mapS_train::MapS = MapS(zeros(1,1),[0.0],[0.0],0.0),
                         mapS_test::MapS  = MapS(zeros(1,1),[0.0],[0.0],0.0);
                         comp_params::CompParams=NNCompParams(),
                         silent::Bool=true)

    @assert typeof(xyz_train) == typeof(xyz_test) "xyz types do no match"

    (comp_params,y_train,y_train_hat,err_train,features) = 
        comp_train(xyz_train,ind_train,mapS_train;
                   comp_params=comp_params,silent=silent)

    (y_test,y_test_hat,err_test,_) = 
        comp_test(xyz_test,ind_test,mapS_test;
                  comp_params=comp_params,silent=silent)

    return (comp_params, y_train, y_train_hat, err_train,
                         y_test , y_test_hat , err_test , features)
end # function comp_train_test

"""
    comp_train_test(lines_train, lines_test,
                    df_line::DataFrame, df_flight::DataFrame,
                    df_map::DataFrame, comp_params::CompParams=NNCompParams();
                    silent::Bool=true, reorient_vec=false)

Train and evaluate aeromagnetic compensation model performance.

**Arguments:**
- `lines_train`:  selected line number(s) for training
- `lines_test`:   selected line number(s) for testing
- `df_line`:      lookup table (DataFrame) of `lines`
- `df_flight`:    lookup table (DataFrame) of flight files
- `df_map`:       lookup table (DataFrame) of map files
- `comp_params`:  `CompParams` aeromagnetic compensation parameters struct, either:
    - `NNCompParams`:  neural network-based aeromagnetic compensation parameters struct
    - `LinCompParams`: linear aeromagnetic compensation parameters struct
- `silent`        (optional) if true, no print outs
- `reorient_vec`: (optional) if true, align vector magnetometer with body frame

**Returns:**
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct
- `y_train`:     training observed data
- `y_train_hat`: training predicted data
- `err_train`:   mean-corrected (per line) training compensation error
- `y_test`:      testing observed data
- `y_test_hat`:  testing predicted data
- `err_test`:    mean-corrected (per line) testing compensation error
"""
function comp_train_test(lines_train, lines_test,
                         df_line::DataFrame, df_flight::DataFrame,
                         df_map::DataFrame, comp_params::CompParams=NNCompParams();
                         silent::Bool=true, reorient_vec::Bool=false)

    (comp_params,y_train,y_train_hat,err_train,features) = 
        comp_train(lines_train,df_line,df_flight,df_map,comp_params;
                   silent=silent, reorient_vec=reorient_vec)

    (y_test,y_test_hat,err_test,_) = 
        comp_test(lines_test,df_line,df_flight,df_map,comp_params;
                  silent=silent,reorient_vec=reorient_vec)

    return (comp_params, y_train, y_train_hat, err_train,
                         y_test , y_test_hat , err_test , features)
end # function comp_train_test

"""
    function print_time(t)

Internal helper function to print time `t` in `sec` if <1 min, otherwise `min`.
"""
function print_time(t, digits=1)
    if t < 60
        @info("time: $(round(t   ,digits=digits)) sec")
    else
        @info("time: $(round(t/60,digits=digits)) min")
    end
end # function print_time
