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
                    x_test::Matrix       = Array{Any}(undef,0,0),
                    y_test::Vector       = [],
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
                         model::Chain         = Chain(),
                         l_segs::Vector       = [length(y)],
                         x_test::Matrix       = Array{Any}(undef,0,0),
                         y_test::Vector       = [],
                         silent::Bool         = true)

    # convert to Float32 for ~50% speedup
    x      = convert.(Float32,x)
    y      = convert.(Float32,y)
    α      = convert.(Float32,α_sgl)
    λ      = convert.(Float32,λ_sgl)
    x_test = convert.(Float32,x_test)
    y_test = convert.(Float32,y_test)

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

    # normalize test data if it was provided
    isempty(x_test) || (x_norm_test = ((x_test .- x_bias) ./ x_scale) * v_scale)

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

    if model == Chain() # not re-training with known model
        m = get_nn_m(xS,yS;hidden=hidden,activation=activation)
    else # initial model already known
        m = model
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

    function test_error_nT(y_test, y_bias, y_scale, m, x_norm_test)
        y_hat_test = denorm_sets(y_bias,y_scale,vec(m(x_norm_test')))
        err_test   = std(err_segs(y_hat_test,y_test,[length(y_hat_test)];silent=silent))
        return (err_test)
    end # function test_error_nT

    # train NN with Adam optimizer
    m_store = deepcopy(m)
    best_loss = loss_all(data_val)
    isempty(x_test) || (best_test_error = test_error_nT(y_test, y_bias, y_scale, m, x_norm_test))

    @info("Epoch 0: loss = $best_loss")
    for i = 1:epoch_adam
        Flux.train!(loss,Flux.params(m),data_train,opt)
        current_loss  = loss_all(data_val)
        # update NN model weights for lowest validation loss or lowest test error
        if isempty(x_test)
            if current_loss < best_loss
                best_loss = current_loss
                m_store   = deepcopy(m)
            end
            mod(i,5) == 0 && @info("Epoch $i: loss = $best_loss")
        else
            test_error = test_error_nT(y_test, y_bias, y_scale, m, x_norm_test)
            if test_error < best_test_error
                best_test_error = test_error
                m_store = deepcopy(m)
            end
            mod(i,5) == 0 && @info("Epoch $i: loss = $current_loss, test error = $(round(best_test_error,digits=2)) nT")
        end
        if mod(i,10) == 0
            y_hat = denorm_sets(y_bias,y_scale,vec(m(x_norm')))
            err   = err_segs(y_hat,y,l_segs;silent=silent)
            @info("$i train error: $(round(std(err),digits=2)) nT")
            isempty(x_test) || @info("$i test error: $(round((test_error),digits=2)) nT")
        end
    end

    Flux.loadmodel!(m,m_store)

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
    model = m

    # get results
    y_hat = denorm_sets(y_bias,y_scale,vec(m(x_norm')))
    err   = err_segs(y_hat,y,l_segs;silent=silent)
    @info("train error: $(round(std(err),digits=2)) nT")

    if !isempty(x_test)
        y_hat_test = denorm_sets(y_bias,y_scale,vec(m(x_norm_test')))
        err_test   = err_segs(y_hat_test,y_test,[length(y_test)];silent=silent)
        @info("test error: $(round(std(err_test),digits=2)) nT")
    end

    # pack data normalizations
    data_norms = (zeros(1,1),zeros(1,1),v_scale,x_bias,x_scale,y_bias,y_scale)

    return (model, data_norms, y_hat, err)
end # function nn_comp_1_train

"""
    nn_comp_1_test(x, y, data_norms::Tuple, model::Chain;
                   l_segs::Vector       = [length(y)],
                   silent::Bool         = false)

Evaluate neural network-based aeromagnetic compensation, model 1 performance.
"""
function nn_comp_1_test(x, y, data_norms::Tuple, model::Chain;
                        l_segs::Vector       = [length(y)],
                        silent::Bool         = false)

    # convert to Float32 for consistency with nn_comp_1_train
    x = convert.(Float32,x)
    y = convert.(Float32,y)

    # unpack data normalizations
    (_,_,v_scale,x_bias,x_scale,y_bias,y_scale) = unpack_data_norms(data_norms)
    x_norm = ((x .- x_bias) ./ x_scale) * v_scale

    # set to test mode in case model uses batchnorm or dropout
    m = model
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
                    model::Chain         = Chain(),
                    l_segs::Vector       = [length(y)],
                    A_test::Matrix       = Array{Any}(undef,0,0),
                    x_test::Matrix       = Array{Any}(undef,0,0),
                    y_test::Vector       = [],
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
                         model::Chain         = Chain(),
                         l_segs::Vector       = [length(y)],
                         A_test::Matrix       = Array{Any}(undef,0,0),
                         x_test::Matrix       = Array{Any}(undef,0,0),
                         y_test::Vector       = [],
                         silent::Bool         = true)

    # convert to Float32 for ~50% speedup
    A       = convert.(Float32,A)
    x       = convert.(Float32,x)
    y       = convert.(Float32,y)
    α       = convert.(Float32,α_sgl)
    λ       = convert.(Float32,λ_sgl)
    A_test  = convert.(Float32,A_test)
    x_test  = convert.(Float32,x_test)
    y_test  = convert.(Float32,y_test)
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

    # normalize test data if it was provided
    isempty(A_test) || (A_norm_test =  (A_test .- A_bias) ./ A_scale)
    isempty(x_test) || (x_norm_test = ((x_test .- x_bias) ./ x_scale) * v_scale)

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

    if model == Chain() # not re-training with known model
        m = get_nn_m(xS,yS;hidden=hidden,activation=activation)
    else # initial model already known
        m = model
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

    function get_test_yhat(y_bias, y_scale, m, A_norm_test, x_norm_test)
        if model_type in [:m2a]
            y_hat_test = denorm_sets(y_bias,y_scale,vec(sum(A_norm_test'.*m(x_norm_test'),dims=1)))
        elseif model_type in [:m2b,:m2c]
            y_hat_test = denorm_sets(y_bias,y_scale,vec(m(x_norm_test'))+vec(A_norm_test*TL_coef))
        elseif model_type in [:m2d]
            y_hat_test = denorm_sets(y_bias,y_scale,vec(sum(A_norm_test'.*(m(x_norm_test').+TL_coef),dims=1)))
        end
        return (y_hat_test)
    end # function get_test_yhat

    function test_error_nT(y_test, y_bias, y_scale, m, A_norm_test, x_norm_test)
        y_hat_test = get_test_yhat(y_bias, y_scale, m, A_norm_test, x_norm_test)
        err_test   = std(err_segs(y_hat_test,y_test,[length(y_hat_test)];silent=silent))
        return (err_test)
    end # function test_error_nT

    # train NN with Adam optimizer
    m_store = deepcopy(m)
    TL_coef_store = deepcopy(TL_coef*y_scale)
    best_loss = loss_all(data_val)
    isempty(x_test) || (best_test_error = test_error_nT(y_test, y_bias, y_scale, m, A_norm_test, x_norm_test))

    @info("Epoch 0: loss = $best_loss")
    for i = 1:epoch_adam
        if model_type in [:m2a,:m2b,:m2d] # train on NN model weights only
            Flux.train!(loss,Flux.params(m),data_train,opt)
        elseif model_type in [:m2c] # train on NN model weights + TL coef
            Flux.train!(loss,Flux.params(m,TL_coef),data_train,opt)
        end
        current_loss = loss_all(data_val)
        if isempty(x_test)
            if current_loss < best_loss
                best_loss = current_loss
                m_store = deepcopy(m)
                TL_coef_store = deepcopy(TL_coef*y_scale)
            end
            mod(i,5) == 0 && @info("Epoch $i: loss = $best_loss")
        else
            test_error = test_error_nT(y_test, y_bias, y_scale, m, A_norm_test, x_norm_test)
            if test_error < best_test_error
                best_test_error = test_error
                m_store = deepcopy(m)
                TL_coef_store = deepcopy(TL_coef*y_scale)
            end
            mod(i,5) == 0 && @info("Epoch $i: loss = $current_loss, test error = $(round(best_test_error,digits=2)) nT")
        end
        if mod(i,10) == 0
            y_hat = get_test_yhat(y_bias, y_scale, m, A_norm, x_norm)
            err = err_segs(y_hat,y,l_segs;silent=silent)
            @info("$i train error: $(round(std(err),digits=2)) nT")
            isempty(x_test) || @info("$i test error: $(round((test_error),digits=2)) nT")
        end
    end

    Flux.loadmodel!(m,m_store)
    TL_coef = vec(TL_coef_store/y_scale)

    if epoch_lbfgs > 0 # LBFGS, may overfit depending on iterations
        data = Flux.DataLoader((A_norm',x_norm',y_norm'),shuffle=true,batchsize=batchsize)

        function lbfgs_train!(m_l,data_l,iter,m_t,λ_l,α_l)
            (A_l,x_l,y_l) = data_l.data
            loss_m2a()    = Flux.mse(vec(sum(A_l.*m_l(x_l),dims=1)) ,vec(y_l))
            loss_m2b()    = Flux.mse(vec(m_l(x_l))+vec(A_l'*TL_coef),vec(y_l))
            loss_m2c()    = Flux.mse(vec(m_l(x_l))+vec(A_l'*TL_coef),vec(y_l))
            loss_m2d()    = Flux.mse(vec(sum(A_l.*(m_l(x_l).+TL_coef),dims=1)),vec(y_l))
            loss_m2a_λ()  = loss_m2a() + λ_l*sum(sparse_group_lasso(m_l,α_l))
            loss_m2b_λ()  = loss_m2b() + λ_l*sum(sparse_group_lasso(m_l,α_l))
            loss_m2c_λ()  = loss_m2c() + λ_l*sum(sparse_group_lasso(m_l,α_l))
            loss_m2d_λ()  = loss_m2d() + λ_l*sum(sparse_group_lasso(m_l,α_l))
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
    model = m

    # get results
    y_hat = get_test_yhat(y_bias, y_scale, m, A_norm, x_norm)
    err = err_segs(y_hat,y,l_segs;silent=silent)
    @info("train error: $(round(std(err),digits=2)) nT")

    # pack data normalizations
    data_norms = (A_bias,A_scale,v_scale,x_bias,x_scale,y_bias,y_scale)

    return (model, TL_coef, data_norms, y_hat, err)
end # function nn_comp_2_train

"""
    nn_comp_2_test(A, x, y, data_norms::Tuple, model::Chain;
                   model_type::Symbol = :m2a,
                   TL_coef::Vector    = zeros(Float32,18),
                   l_segs::Vector     = [length(y)],
                   silent::Bool       = false)

Evaluate neural network-based aeromagnetic compensation, model 2 performance.
"""
function nn_comp_2_test(A, x, y, data_norms::Tuple, model::Chain;
                        model_type::Symbol = :m2a,
                        TL_coef::Vector    = zeros(Float32,18),
                        l_segs::Vector     = [length(y)],
                        silent::Bool       = false)

    # convert to Float32 for consistency with nn_comp_2_train
    A       = convert.(Float32,A)
    x       = convert.(Float32,x)
    y       = convert.(Float32,y)
    TL_coef = convert.(Float32,TL_coef)

    # unpack data normalizations
    (A_bias,A_scale,v_scale,x_bias,x_scale,y_bias,y_scale) =
        unpack_data_norms(data_norms)
    A_norm =  (A .- A_bias) ./ A_scale
    x_norm = ((x .- x_bias) ./ x_scale) * v_scale

    # set to test mode in case model uses batchnorm or dropout
    m = model
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
    get_curriculum_indices(feats, sigma=1)

Get indices for curriculum learning (to train the Tolles-Lawson section of
the loss) and indices to train the neural network.

**Arguments:**
- `TL_diff`: difference of TL model to ground truth
- `sigma`:   (optional) number of standard deviations

**Returns:**
- `curriculum_ind`: indices for curriculum learning
- `nn_ind`:         indices for training the neural network
"""
function get_curriculum_indices(TL_diff, sigma=1)
     zeroed_diff = vec(detrend(TL_diff;mean_only=true))
     cutoff = sigma*std(zeroed_diff)
     # NN indices are those outside cutoff (outliers)
     nn_inds = findall(v -> (v<-cutoff) || (cutoff < v),zeroed_diff)
     curriculum_inds = collect(1:length(zeroed_diff))
     # TL indices are those closer to mean
     deleteat!(curriculum_inds, nn_inds)

    return (curriculum_inds, nn_inds)
end # function get_curriculum_indices

"""
    extract_TL_matrices(TL_coef, terms; Bt_scale=50000f0)

**Arguments**
- `TL_coef`:  Tolles-Lawson coefficients (must include `:permanent` and `:induced`)
- `terms`:    Tolles-Lawson terms used {`:permanent`,`:induced`,`:eddy`}
- `Bt_scale`: (optional) scaling factor for induced and eddy current terms [nT]

**Returns**
- `TL_coef_p`: `3`x`1` vector of permanent field coefficients
- `TL_coef_i`: `3`x`3` symmetric matrix of induced field coefficients, denormalized
- `TL_coef_e`: `3`x`3` matrix of eddy current coefficients, denormalized
"""
function extract_TL_matrices(TL_coef, terms; Bt_scale=50000f0)
    @assert any([:permanent,:p,:permanent3,:p3] .∈ (terms,)) "permanent terms are required"
    @assert any([:induced,:i,:induced6,:i6,:induced5,:i5,:induced3,:i3] .∈ (terms,)) "induced terms are required"
    @assert !any([:fdm,:f,:d,:fdm3,:f3,:d3,:bias,:b] .∈ (terms,)) "derivative and bias terms may not be used"

    N = length(TL_coef)
    A_test = create_TL_A([1.0],[1.0],[1.0];terms=terms)
    @assert N == length(A_test) "TL_coef does not agree with specified terms"

    TL_coef_p = vec(TL_coef[1:3])

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
end # function extract_TL_matrices

"""
    extract_TL_vector(TL_coef_p, TL_coef_i, TL_coef_e, terms; Bt_scale=50000f0)

**Arguments**
- `TL_coef_p`: `3`x`1` vector of permanent field coefficients
- `TL_coef_i`: `3`x`3` symmetric matrix of induced field coefficients, denormalized
- `TL_coef_e`: `3`x`3` matrix of eddy current coefficients, denormalized
- `terms`:     Tolles-Lawson terms used {`:permanent`,`:induced`,`:eddy`}
- `Bt_scale`:  (optional) scaling factor for induced and eddy current terms [nT]

**Returns**
- `TL_coef`:  Tolles-Lawson coefficients
"""
function extract_TL_vector(TL_coef_p, TL_coef_i, TL_coef_e, terms; Bt_scale=50000f0)

    TL_coef_p = vec(TL_coef_p)

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
        TL_coef_e = vec(TL_coef_e')                    * Bt_scale
    elseif any([:eddy8,:e8] .∈ (terms,))
        TL_coef_e = vec(TL_coef_e')[1:end-1]           * Bt_scale
    elseif any([:eddy3,:e3] .∈ (terms,))
        TL_coef_e = vec(TL_coef_e')[[1,1],[2,2],[3,3]] * Bt_scale
    else
        TL_coef_e = []
    end

    TL_coef = [TL_coef_p; TL_coef_i; TL_coef_e]

    return (TL_coef)
end # function extract_TL_vector

"""
    get_TL_aircraft_vector(B_vec, B_vec_dot, TL_coef_p, TL_coef_i, TL_coef_e;
                           return_parts::Bool=false)

**Arguments**
- `B_vec`:        `3`x`N` matrix of vector magnetometer measurements
- `B_vec_dot`:    `3`x`N` matrix of vector magnetometer measurement derivatives
- `TL_coef_p`:    `3`x`1` vector of permanent field coefficients
- `TL_coef_i`:    `3`x`3` symmetric matrix of induced field coefficients, denormalized
- `TL_coef_e`:    `3`x`3` matrix of eddy current coefficients, denormalized
- `return_parts`: (optional) if true, also return `TL_perm`, `TL_induced`, and `TL_eddy`

**Returns**
- `TL_aircraft`: `3`x`N` matrix of TL aircraft vector field
- `TL_perm`:     `3`x`N` if `return_parts = true`, matrix of TL permanent vector field
- `TL_induced`:  `3`x`N` if `return_parts = true`, matrix of TL induced vector field
- `TL_eddy`:     `3`x`N` if `return_parts = true`, matrix of TL eddy current vector field
"""
function get_TL_aircraft_vector(B_vec, B_vec_dot, TL_coef_p, TL_coef_i, TL_coef_e;
                                return_parts::Bool=false)

    TL_perm    = TL_coef_p .* ones(size(B_vec))
    TL_induced = TL_coef_i * B_vec

    if length(TL_coef_e) > 0
        TL_eddy     = TL_coef_e * B_vec_dot
        TL_aircraft = TL_perm + TL_induced + TL_eddy
    else
        TL_eddy     = []
        TL_aircraft = TL_perm + TL_induced
    end

    if return_parts
        return (TL_aircraft, TL_perm, TL_induced, TL_eddy) # (3,batch_size)
    else
        return (TL_aircraft) # (3,batch_size)
    end
end # function get_TL_aircraft_vector

"""
    nn_comp_3_train(A, Bt, B_dot, x, y, no_norm;
                    model_type::Symbol   = :m3s,
                    norm_type_x::Symbol  = :standardize,
                    norm_type_y::Symbol  = :standardize,
                    TL_coef::Vector      = zeros(Float32,18),
                    terms_A              = [:permanent,:induced,:eddy],
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
                    model::Chain         = Chain(),
                    l_segs::Vector       = [length(y)],
                    A_test::Matrix       = Array{Any}(undef,0,0),
                    Bt_test::Vector      = [],
                    B_dot_test::Matrix   = Array{Any}(undef,0,0),
                    x_test::Matrix       = Array{Any}(undef,0,0),
                    y_test::Vector       = [],
                    silent::Bool         = true)

Train neural network-based aeromagnetic compensation, model 3. Model 3
architectures retain the Tolles-Lawson (TL) terms in vector form, making it
possible to remove the Taylor expansion approximation used for predicting
the Earth field in the loss function, as well as create vector-based neural
network corrections.

Note that this model is subject to change, and not all options are supported
(e.g., no LBFGS, and generally the IGRF and diurnal fields should not be
subtracted out from the `y` target value)

Currently, it is recommended to use a low-pass (not a band-pass) filter to
initialize the Tolles-Lawson coefficients, as a band-pass filter removes the
Earth field and leaves a large bias in the aircraft field prediction, e.g.:
`TL_coef = create_TL_coef(getfield(xyz,use_vec), getfield(xyz,use_mag)-xyz.mag_1_c, TL_ind;
                          terms=terms, pass1=0.0, pass2=0.9)`

- `model_type`:
    - `:m3tl` = no NN, TL coefficients fine-tuned via SGD, without Taylor expansion for `y_type` :b and :c (for testing)
    - `:m3s`  = NN determines scalar correction to TL, using expanded TL vector terms for explainability
    - `:m3v`  = NN determines vector correction to TL, using expanded TL vector terms for explainability
    - `:m3sc` = `:m3s` with curriculum learning based on TL error
    - `:m3vc` = `:m3v` with curriculum learning based on TL error
"""
function nn_comp_3_train(A, Bt, B_dot, x, y, no_norm;
                         model_type::Symbol   = :m3s,
                         norm_type_x::Symbol  = :standardize,
                         norm_type_y::Symbol  = :standardize,
                         TL_coef::Vector      = zeros(Float32,18),
                         terms_A              = [:permanent,:induced,:eddy],
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
                         model::Chain         = Chain(),
                         l_segs::Vector       = [length(y)],
                         A_test::Matrix       = Array{Any}(undef,0,0),
                         Bt_test::Vector      = [],
                         B_dot_test::Matrix   = Array{Any}(undef,0,0),
                         x_test::Matrix       = Array{Any}(undef,0,0),
                         y_test::Vector       = [],
                         silent::Bool         = true)

    @assert epoch_lbfgs == 0 "LBFGS not implemented for model 3"

    # convert to Float32 for ~50% speedup
    A          = convert.(Float32,A)
    Bt         = convert.(Float32,Bt)    # magnitude of total field measurements
    B_dot      = convert.(Float32,B_dot) # finite differences of total field vector
    x          = convert.(Float32,x)
    y          = convert.(Float32,y)
    α          = convert.(Float32,α_sgl)
    λ          = convert.(Float32,λ_sgl)
    A_test     = convert.(Float32,A_test)
    Bt_test    = convert.(Float32,Bt_test)
    B_dot_test = convert.(Float32,B_dot_test)
    x_test     = convert.(Float32,x_test)
    y_test     = convert.(Float32,y_test)
    TL_coef    = convert.(Float32,TL_coef)

    # assume all terms are stored, but they may be zero if not trained
    Bt_scale = 50000f0
    (TL_coef_p,TL_coef_i,TL_coef_e) = extract_TL_matrices(TL_coef,terms_A;Bt_scale=Bt_scale)

    B_unit    = A[:,1:3]     # normalized vector magnetometer reading
    B_vec     = B_unit .* Bt # vector magnetometer to be used in TL
    B_vec_dot = B_dot        # not exactly true, but internally consistent
    isempty(A_test)     || (B_unit_test    = A_test[:,1:3])
    isempty(Bt_test)    || (B_vec_test     = B_unit_test .* Bt_test)
    isempty(B_dot_test) || (B_vec_dot_test = B_dot_test)

    if y_type in [:c,:d]
        y_type_aircraft = true
    elseif y_type in [:a,:b]
        y_type_aircraft = false
    else
        error("unsupported y_type in nn_comp_3_train, $y_type")
    end

    # compute scalar correction to aircraft or Earth field target
    function get_scalar_TL(B_vec, B_vec_dot, B_unit, TL_coef_p, TL_coef_i, TL_coef_e, y_type_aircraft)
        TL_aircraft = get_TL_aircraft_vector(B_vec, B_vec_dot, TL_coef_p, TL_coef_i, TL_coef_e)
        if y_type_aircraft # need the amount to subtract from scalar mag
            yhat = sum(TL_aircraft .* B_unit, dims=1) # dot product
            #println("Aircraft correction=", yhat)
        else
            B_e = B_vec - TL_aircraft # (3,batch)
            yhat = sqrt.(sum(B_e.^2,dims=1)) # magnitude of scalar Earth field
            #println("Aircraft correction=", sum(TL_aircraft .* B_unit, dims=1))
            #println("|Be|=", yhat)
        end
        return (yhat)
    end # function get_scalar_TL

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

    # normalize test data if it was provided
    isempty(x_test) || (x_norm_test = ((x_test .- x_bias) ./ x_scale) * v_scale)

    # separate into training and validation
    if frac_train < 1
        N = size(x_norm,1)
        p = randperm(N)
        N_train = floor(Int,frac_train*N)
        p_train = p[1:N_train]
        p_val   = p[N_train+1:end]
        B_vec_dot_train = B_vec_dot[p_train,:]
        B_vec_dot_val   = B_vec_dot[p_val  ,:]
        B_unit_train    = B_unit[p_train,:]
        B_unit_val      = B_unit[p_val  ,:]
        B_vec_train     = B_vec[ p_train,:]
        B_vec_val       = B_vec[ p_val  ,:]
        x_norm_train    = x_norm[p_train,:]
        x_norm_val      = x_norm[p_val  ,:]
        y_norm_train    = y_norm[p_train,:]
        y_norm_val      = y_norm[p_val  ,:]

        if model_type in [:m3sc,:m3vc]
            @info("Making curriculum")
            # calculate TL estimate in training data
            y_TL = get_scalar_TL(B_vec_train', B_vec_dot_train', B_unit_train', 
                                 TL_coef_p, TL_coef_i, TL_coef_e, y_type_aircraft)
            y_train   = y[p_train,:]
            y_TL_diff = y_train' - y_TL

            (curriculum_ind,_) = get_curriculum_indices(y_TL_diff)
            data_train   = Flux.DataLoader((B_vec_train[ curriculum_ind, :]',
                                           B_vec_dot_train[curriculum_ind, :]',
                                           B_unit_train[curriculum_ind, :]',
                                           x_norm_train[curriculum_ind, :]',
                                           y_norm_train[curriculum_ind, :]'),
                                           shuffle=true,batchsize=batchsize)
            data_train_2 = Flux.DataLoader((B_vec_train',B_vec_dot_train',
                                           B_unit_train',x_norm_train',y_norm_train'),
                                           shuffle=true,batchsize=batchsize)
        else
            data_train   = Flux.DataLoader((B_vec_train',B_vec_dot_train',
                                           B_unit_train',x_norm_train',y_norm_train'),
                                           shuffle=true,batchsize=batchsize)
        end
        data_val   = Flux.DataLoader((B_vec_val',B_vec_dot_val',
                                     B_unit_val',x_norm_val',y_norm_val'),
                                     shuffle=true,batchsize=batchsize)
    else
        data_train = Flux.DataLoader((B_vec',B_vec_dot',
                                     B_unit',x_norm',y_norm'),
                                     shuffle=true,batchsize=batchsize)
        data_val   = data_train
    end

    # setup NN
    xS = size(x_norm,2) # number of features
    if model_type in [:m3tl,:m3s,:m3sc]
        yS = 1 # additive correction
    elseif model_type in [:m3v,:m3vc]
        yS = 3
    end

    if model == Chain() # not re-training with known model
        m = get_nn_m(xS,yS;hidden=hidden,activation=activation)
    else # initial model already known
        m = model
    end

    ## setup optimizer and loss function
    if model_type in [:m3g]
        #TODO change these constants to be globally initialized
        opt = Scheduler(Step(λ = η_adam, γ = 0.8, step_sizes = [epoch_adam ÷ 2,epoch_adam ÷ 5, epoch_adam ÷ 5]), Adam())
    else
        opt = Adam(η_adam)
    end

    # compute TL vector, then add NN(x) correction to that
    function get_vector_corrected_TL(B_vec_in, B_vec_dot_in, B_unit_in, x_in, has_nn::Float32=1f0)
        y_TL = get_TL_aircraft_vector(B_vec_in, B_vec_dot_in, TL_coef_p, TL_coef_i, TL_coef_e)
        y_NN = has_nn .* m(x_in)
        y_NN_scaled    = y_scale .* y_NN # scaling factor keeps NN output near TL
        aircraft_field = y_TL + y_NN_scaled
        if y_type_aircraft # need the amount to subtract from scalar mag
            yhat = sum(aircraft_field .* B_unit_in, dims=1) # dot product
            #println("Aircraft correction=", yhat)
        else
            B_e  = B_vec_in - aircraft_field # (3,batch)
            yhat = sqrt.(sum(B_e.^2,dims=1)) # magnitude of scalar Earth field
            #println("Aircraft correction=", sum(aircraft_field .* B_unit_in, dims=1))
            #println("|Be|=", yhat)
        end
        return (yhat)
    end # function get_vector_corrected_TL

    function loss_m3tl(B_vec_l,B_vec_dot_l,B_unit_l,_,y_l)
        y_TL = get_scalar_TL(B_vec_l, B_vec_dot_l, B_unit_l, TL_coef_p, TL_coef_i, TL_coef_e, y_type_aircraft)
        y_TL_norm = (y_TL .- y_bias)./y_scale
        return (Flux.mse(vec(y_TL_norm),vec(y_l)))
    end # function loss_m3tl

    function loss_m3s(B_vec_l,B_vec_dot_l,B_unit_l,x_l,y_l)
        y_TL = get_scalar_TL(B_vec_l, B_vec_dot_l, B_unit_l, TL_coef_p, TL_coef_i, TL_coef_e, y_type_aircraft)
        y_TL_norm = vec((y_TL .- y_bias)./y_scale) # get into same scale as NN output
        y_NN = vec(m(x_l)) # computes scalar correction to TL correction
        return (Flux.mse(y_TL_norm+y_NN,vec(y_l)))
    end # function loss_m3s

    function loss_m3v(B_vec_l,B_vec_dot_l,B_unit_l,x_l,y_l;has_nn=1f0)
        yhat = get_vector_corrected_TL(B_vec_l, B_vec_dot_l, B_unit_l, x_l, has_nn)
        yhat_norm = (yhat .- y_bias)./y_scale
        return (Flux.mse(vec(yhat_norm),vec(y_l)))
    end # function loss_m3v

    loss_m3tl_λ(B_vec_l,B_vec_dot_l,B_unit_l,x_l,y_l) = loss_m3tl(B_vec_l,B_vec_dot_l,B_unit_l,x_l,y_l) + λ*sum(sparse_group_lasso(m,α))
    loss_m3s_λ( B_vec_l,B_vec_dot_l,B_unit_l,x_l,y_l) = loss_m3s( B_vec_l,B_vec_dot_l,B_unit_l,x_l,y_l) + λ*sum(sparse_group_lasso(m,α))
    loss_m3v_λ( B_vec_l,B_vec_dot_l,B_unit_l,x_l,y_l) = loss_m3v( B_vec_l,B_vec_dot_l,B_unit_l,x_l,y_l) + λ*sum(sparse_group_lasso(m,α))

    #model_type == :m3tl && (loss = λ > 0 ? loss_m3tl_λ : loss_m3tl)
    #model_type == :m3s  && (loss = λ > 0 ? loss_m3s_λ  : loss_m3s)
    #model_type == :m3v  && (loss = λ > 0 ? loss_m3v_λ  : loss_m3v)

    if model_type in [:m3tl,:m3sc]
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

        for (B_vec_l,B_vec_dot_l,B_unit_l,x_l,y_l) in data_l
            l += loss(B_vec_l,B_vec_dot_l,B_unit_l,x_l,y_l)
        end

        l/length(data_l)
    end # function loss_all

    function get_test_yhat(y_bias, y_scale, m, x_norm_test, B_vec_test, B_vec_dot_test, B_unit_test)
        if model_type in [:m3tl]
            y_hat_test = vec(get_scalar_TL(transpose(B_vec_test), transpose(B_vec_dot_test), transpose(B_unit_test), TL_coef_p, TL_coef_i, TL_coef_e, y_type_aircraft))
        elseif model_type in [:m3s,:m3sc]
            y_TL       = get_scalar_TL(transpose(B_vec_test), transpose(B_vec_dot_test), transpose(B_unit_test), TL_coef_p, TL_coef_i, TL_coef_e, y_type_aircraft)
            y_TL_norm  = vec((y_TL .- y_bias)./y_scale) # get into same scale as NN output
            y_NN       = vec(m(transpose(x_norm_test)))
            y_hat_test = denorm_sets(y_bias,y_scale, (y_TL_norm + y_NN))
        elseif model_type in [:m3v,:m3vc]
            has_nn     = 1f0
            y_hat_test = vec(get_vector_corrected_TL(transpose(B_vec_test), transpose(B_vec_dot_test),
                                                     transpose(B_unit_test), transpose(x_norm_test), has_nn))
        end
        return (y_hat_test)
    end # function get_test_yhat

    function test_error_nT(y_test, y_bias, y_scale, m, x_norm_test, B_vec_test, B_vec_dot_test, B_unit_test)
        y_hat_test = get_test_yhat(y_bias, y_scale, m, x_norm_test, B_vec_test, B_vec_dot_test, B_unit_test)
        err_test   = std(err_segs(y_hat_test,y_test,[length(y_hat_test)];silent=silent))
        return (err_test)
    end # function test_error_nT

    # train NN with ADAM optimizer
    m_store = deepcopy(m)
    TL_coef_store = extract_TL_vector(TL_coef_p,TL_coef_i,TL_coef_e,terms_A;Bt_scale=Bt_scale)
    best_loss = loss_all(data_val)
    isempty(x_test) || (best_test_error = test_error_nT(y_test, y_bias, y_scale, m,
                                                        x_norm_test, B_vec_test,
                                                        B_vec_dot_test, B_unit_test))

    @info("Epoch 0: loss = $best_loss")
    has_nn = 1f0 # multiplier to keep/exclude NN contribution into y_hat calculation
    for i = 1:epoch_adam
        if model_type in [:m3tl,:m3s,:m3v] # train on NN model weights + TL coef
            Flux.train!(loss,Flux.params(m,TL_coef_p,TL_coef_i.data,TL_coef_e),data_train,opt)
        elseif model_type in [:m3sc,:m3vc] #TODO change these constants to be globally initialized
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
        if isempty(x_test)
            if current_loss < best_loss
                best_loss = current_loss
                m_store = deepcopy(m)
                TL_coef_store = extract_TL_vector(TL_coef_p,TL_coef_i,TL_coef_e,terms_A;Bt_scale=Bt_scale)
            end
            mod(i,5) == 0 && @info("Epoch $i: loss = $best_loss")
        else
            test_error = test_error_nT(y_test, y_bias, y_scale, m, x_norm_test,
                                       B_vec_test, B_vec_dot_test, B_unit_test)
            if test_error < best_test_error
                best_test_error = test_error
                m_store = deepcopy(m)
                TL_coef_store = extract_TL_vector(TL_coef_p,TL_coef_i,TL_coef_e,terms_A;Bt_scale=Bt_scale)
            end
            mod(i,5) == 0 && @info("Epoch $i: loss = $current_loss, test error = $(round(best_test_error,digits=2)) nT")
        end
        if mod(i,10) == 0
            y_hat = get_test_yhat(y_bias, y_scale, m, x_norm, B_vec, B_vec_dot, B_unit)
            err = err_segs(y_hat,y,l_segs;silent=silent)
            @info("$i train error: $(round(std(err),digits=2)) nT")
            isempty(x_test) || @info("$i test error: $(round((test_error),digits=2)) nT")
        end
    end

    Flux.loadmodel!(m,m_store)
    TL_coef = TL_coef_store

    # set to test mode in case model uses batchnorm or dropout
    Flux.testmode!(m)
    model = m

    # get results
    y_hat = get_test_yhat(y_bias, y_scale, m, x_norm, B_vec, B_vec_dot, B_unit)
    err = err_segs(y_hat,y,l_segs;silent=silent)
    @info("train error: $(round(std(err),digits=2)) nT")

    # pack data normalizations
    data_norms = (zeros(1,1),zeros(1,1),v_scale,x_bias,x_scale,y_bias,y_scale)

    return (model, TL_coef, data_norms, y_hat, err)
end # function nn_comp_3_train

"""
    nn_comp_3_test(A, Bt, B_dot, x, y, data_norms::Tuple, model::Chain;
                   model_type::Symbol = :m3s,
                   y_type             = :d,
                   TL_coef::Vector    = zeros(Float32,18),
                   terms_A            = [:permanent,:induced,:eddy],
                   l_segs::Vector     = [length(y)],
                   silent::Bool       = false)

Evaluate neural network-based aeromagnetic compensation, model 3 performance.
"""
function nn_comp_3_test(A, Bt, B_dot, x, y, data_norms::Tuple, model::Chain;
                        model_type::Symbol = :m3s,
                        y_type             = :d,
                        TL_coef::Vector    = zeros(Float32,18),
                        terms_A            = [:permanent,:induced,:eddy],
                        l_segs::Vector     = [length(y)],
                        silent::Bool       = false)

    # convert to Float32 for consistency with nn_comp_3_train
    A       = convert.(Float32,A)
    Bt      = convert.(Float32,Bt)    # magnitude of total field measurements
    B_dot   = convert.(Float32,B_dot) # finite differences of total field vector
    x       = convert.(Float32,x)
    y       = convert.(Float32,y)
    TL_coef = convert.(Float32,TL_coef)

    # assume all terms are stored, but they may be zero if not trained
    Bt_scale = 50000f0
    (TL_coef_p,TL_coef_i,TL_coef_e) = extract_TL_matrices(TL_coef,terms_A;Bt_scale=Bt_scale)

    B_unit    = A[:,1:3]     # normalized vector magnetometer reading
    B_vec     = B_unit .* Bt # vector magnetometer to be used in TL
    B_vec_dot = B_dot        # not exactly true, but internally consistent

    # unpack data normalizations
    (_,_,v_scale,x_bias,x_scale,y_bias,y_scale) = unpack_data_norms(data_norms)
    x_norm = ((x .- x_bias) ./ x_scale) * v_scale

    if y_type in [:c,:d]
        y_type_aircraft = true
    elseif y_type in [:a,:b]
        y_type_aircraft = false
    else
        error("unsupported y_type in nn_comp_3_test, $y_type")
    end

    # set to test mode in case model uses batchnorm or dropout
    m = model
    Flux.testmode!(m)

    # compute scalar correction to aircraft or Earth field target
    function get_scalar_TL(B_vec, B_vec_dot, B_unit, TL_coef_p, TL_coef_i, TL_coef_e, y_type_aircraft)
        TL_aircraft = get_TL_aircraft_vector(B_vec, B_vec_dot, TL_coef_p, TL_coef_i, TL_coef_e)
        if y_type_aircraft # need the amount to subtract from scalar mag
            yhat = sum(TL_aircraft .* B_unit, dims=1) # dot product
            #println("Aircraft correction=", yhat)
        else
            B_e = B_vec - TL_aircraft # (3,batch)
            yhat = sqrt.(sum(B_e.^2,dims=1)) # magnitude of scalar Earth field
            #println("Aircraft correction=", sum(TL_aircraft .* B_unit, dims=1))
            #println("|Be|=", yhat)
        end
        return (yhat)
    end # function get_scalar_TL

    # compute TL vector, then add NN(x) correction to that
    function get_vector_corrected_TL(B_vec_in, B_vec_dot_in, B_unit_in, x_in, has_nn::Float32=1f0)
        y_TL = get_TL_aircraft_vector(B_vec_in, B_vec_dot_in, TL_coef_p, TL_coef_i, TL_coef_e)
        y_NN = has_nn .* m(x_in)
        y_NN_scaled    = y_scale .* y_NN # scaling factor keeps NN output near TL
        aircraft_field = y_TL + y_NN_scaled
        if y_type_aircraft # need the amount to subtract from scalar mag
            yhat = sum(aircraft_field .* B_unit_in, dims=1) # dot product
            #println("Aircraft correction=", yhat)
        else
            B_e  = B_vec_in - aircraft_field # (3,batch)
            yhat = sqrt.(sum(B_e.^2,dims=1)) # magnitude of scalar Earth field
            #println("Aircraft correction=", sum(aircraft_field .* B_unit_in, dims=1))
            #println("|Be|=", yhat)
        end
        return (yhat)
    end # function get_vector_corrected_TL

    # get results
    if model_type in [:m3tl]
        y_hat     = vec(get_scalar_TL(transpose(B_vec), transpose(B_vec_dot), transpose(B_unit), TL_coef_p, TL_coef_i, TL_coef_e, y_type_aircraft))
    elseif model_type in [:m3s,:m3sc]
        y_TL      = get_scalar_TL(transpose(B_vec), transpose(B_vec_dot), transpose(B_unit), TL_coef_p, TL_coef_i, TL_coef_e, y_type_aircraft)
        y_TL_norm = vec((y_TL .- y_bias)./y_scale)  # get into same scale as NN output
        y_NN      = vec(m(transpose(x_norm)))
        y_hat     = denorm_sets(y_bias,y_scale, (y_TL_norm + y_NN))
    elseif model_type in [:m3v,:m3vc]
        has_nn    = 1f0
        y_hat     = vec(get_vector_corrected_TL(transpose(B_vec), transpose(B_vec_dot),
                                                transpose(B_unit), transpose(x_norm), has_nn))
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
- `return_set`: (optional) if true, return `coef_set` instead of other outputs
- `silent`:     (optional) if true, no print outs

**Returns:**
- `model`:      Tuple of PLSR-based model `(coefficients, bias=0)`
- `data_norms`: Tuple of data normalizations, e.g. `(x_bias,x_scale,y_bias,y_scale)`
- `y_hat`:      predicted data
- `err`:        mean-corrected (per line) error
- `coef_set`:   if `return_set = true`, set of coefficients (size `nx` x `ny` x `k`)
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
    bias = zero(eltype(coef))

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
- `model`:      Tuple of elastic net-based model `(coefficients, bias)`
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
- `trim`:        (optional) number of elements to trim (e.g. due to bpf)
- `λ`:           (optional) ridge parameter
- `norm_type_x`: (optional) normalization for `x` matrix
- `norm_type_y`: (optional) normalization for `y` target vector
- `l_segs`:      (optional) vector of lengths of `lines`, sum(l_segs) == length(y)
- `silent`:      (optional) if true, no print outs

**Returns:**
- `model`:      Tuple of linear regression model `(coefficients, bias=0)`
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
    bias = zero(eltype(coef))

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
    linear_test(x, y, data_norms::Tuple, model;
                l_segs::Vector = [length(y)],
                silent::Bool   = false)

Evaluate linear model performance.

**Arguments:**
- `x`:          input data
- `y`:          observed data
- `data_norms`: Tuple of data normalizations, e.g. `(x_bias,x_scale,y_bias,y_scale)`
- `model`:      Tuple of model `(coefficients, bias)` or only `coefficients`
- `l_segs`:     (optional) vector of lengths of `lines`, sum(l_segs) == length(y)
- `silent`:     (optional) if true, no print outs

**Returns:**
- `y_hat`: predicted data
- `err`:   mean-corrected (per line) error
"""
function linear_test(x, y, data_norms::Tuple, model;
                     l_segs::Vector = [length(y)],
                     silent::Bool   = false)

    # unpack linear model weights
    (coef,bias) = model

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
               comp_params::CompParams = NNCompParams(),
               xyz_test::XYZ           = xyz,
               ind_test::BitVector     = BitVector(),
               silent::Bool            = true)

Train an aeromagnetic compensation model.

**Arguments:**
- `xyz`:         `XYZ` flight data struct
- `ind`:         selected data indices
- `mapS`:        (optional) `MapS` scalar magnetic anomaly map struct, only used for `y_type = :b, :c`
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct, either:
    - `NNCompParams`:  neural network-based aeromagnetic compensation parameters struct
    - `LinCompParams`: linear aeromagnetic compensation parameters struct
- `xyz_test`:    (optional) `XYZ` held-out test data struct
- `ind_test`:    (optional) indices for test data struct
- `silent`:      (optional) if true, no print outs

**Returns:**
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct
- `y`:           observed data
- `y_hat`:       predicted data
- `err`:         compensation error
- `features`:    full list of features (including components of TL `A`, etc.)
"""
function comp_train(xyz::XYZ, ind, mapS::MapS=MapS(zeros(1,1),[0.0],[0.0],0.0);
                    comp_params::CompParams = NNCompParams(),
                    xyz_test::XYZ           = xyz,
                    ind_test::BitVector     = BitVector(),
                    silent::Bool            = true)

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
        (A,Bt,B_dot) = create_TL_A(getfield(xyz,use_vec),ind;
                                   terms=terms_A,return_B=true)
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

    # set held-out test line
    A_test     = Array{Any}(undef,0,0)
    Bt_test    = []
    B_dot_test = Array{Any}(undef,0,0)
    x_test     = Array{Any}(undef,0,0)
    y_test     = []
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
            elseif model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc]
                (model,TL_coef,data_norms,y_hat_fi,err_fi) =
                    nn_comp_3_train(A,Bt,B_dot,x_fi,y,no_norm;
                                    model_type  = model_type,
                                    norm_type_x = norm_type_x,
                                    norm_type_y = norm_type_y,
                                    TL_coef     = TL_coef,
                                    terms_A     = terms_A,
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
                                    A_test      = A_test,
                                    Bt_test     = Bt_test,
                                    B_dot_test  = B_dot_test,
                                    x_test      = x_test_fi,
                                    y_test      = y_test,
                                    silent      = silent)
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
                                       TL_coef    = vec(TL_coef))
            @save drop_fi_bson*"_$i.bson" comp_params

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
                                model       = model,
                                A_test      = A_test,
                                x_test      = x_test,
                                y_test      = y_test,
                                silent      = silent)
        elseif model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc]
            (model,TL_coef,data_norms,y_hat,err) =
                nn_comp_3_train(A,Bt,B_dot,x,y,no_norm;
                                model_type  = model_type,
                                norm_type_x = norm_type_x,
                                norm_type_y = norm_type_y,
                                TL_coef     = TL_coef,
                                terms_A     = terms_A,
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
                                A_test      = A_test,
                                Bt_test     = Bt_test,
                                B_dot_test  = B_dot_test,
                                x_test      = x_test,
                                y_test      = y_test,
                                silent      = silent)
        elseif model_type in [:TL,:mod_TL,:map_TL]
            trim = model_type in [:TL,:mod_TL] ? 20 : 0
            (model,data_norms,y_hat,err) =
                linear_fit(A,y;
                           trim=trim, λ=λ_TL,
                           norm_type_x = norm_type_A,
                           norm_type_y = norm_type_y,
                           silent      = silent)
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
                                   TL_coef    = vec(TL_coef))
    elseif typeof(comp_params) <: LinCompParams
        comp_params = LinCompParams(comp_params,
                                    data_norms = data_norms,
                                    model      = model)
    end

    print_time(time()-t0,1)

    return (comp_params, y, y_hat, err, features)
end # function comp_train

"""
    comp_train(xyz_array::Vector{MagNav.XYZ20{Int64, Float64}},
               ind_array::Vector{BitVector},
               mapS::MapS              = MapS(zeros(1,1),[0.0],[0.0],0.0);
               comp_params::CompParams = NNCompParams(),
               xyz_test::XYZ           = xyz_array[1],
               ind_test::BitVector     = BitVector(),
               silent::Bool            = true)

Train an aeromagnetic compensation model.

**Arguments:**
- `xyz_array`:   vector of `XYZ` flight data structs
- `ind_array`:   vector of selected data indices
- `mapS`:        (optional) `MapS` scalar magnetic anomaly map struct, only used for `y_type = :b, :c`
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct, either:
    - `NNCompParams`:  neural network-based aeromagnetic compensation parameters struct
    - `LinCompParams`: linear aeromagnetic compensation parameters struct
- `xyz_test`:    (optional) `XYZ` held-out test data struct
- `ind_test`:    (optional) indices for test data struct
- `silent`:      (optional) if true, no print outs

**Returns:**
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct
- `y`:           observed data
- `y_hat`:       predicted data
- `err`:         compensation error
- `features`:    full list of features (including components of TL `A`, etc.)
"""
function comp_train(xyz_array::Vector{MagNav.XYZ20{Int64, Float64}},
                    ind_array::Vector{BitVector},
                    mapS::MapS              = MapS(zeros(1,1),[0.0],[0.0],0.0);
                    comp_params::CompParams = NNCompParams(),
                    xyz_test::XYZ           = xyz_array[1],
                    ind_test::BitVector     = BitVector(),
                    silent::Bool            = true)

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
        @info("forcing y_type = :e (BPF'd total field)")
        y_type = :e
    end
    if (model_type in [:map_TL]) & (y_type != :c)
        y_type  = :c
        @info("forcing y_type = :c (aircraft field #1, using map)")
    end

    # initialize loop over XYZ structs and indices
    xyz = xyz_array[1]
    ind = ind_array[1]

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
        (A,Bt,B_dot) = create_TL_A(getfield(xyz_array[1],use_vec),ind;
                                   terms=terms_A,return_B=true)
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

    y_hat = zero(y) # initialize
    err   = 10*y    # initialize

    for (xyz, ind) in zip(xyz_array[2:end],ind_array[2:end])
         # map values along trajectory (if needed)
        if y_type in [:b,:c]
            traj_alt = median(xyz.traj.alt[ind])
            mapS.alt > 0 && (mapS = upward_fft(mapS,traj_alt;α=200))
            itp_mapS = map_itp(mapS)
            map_val  = itp_mapS.(xyz.traj.lon[ind],xyz.traj.lat[ind])
        else
            map_val  = -1
        end

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

        y_hat = zero(y) # initialize
        err   = 10*y    # initialize

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
        A = vcat(A,A_)

    end

    # set held-out test line
    A_test     = Array{Any}(undef,0,0)
    Bt_test    = []
    B_dot_test = Array{Any}(undef,0,0)
    x_test     = Array{Any}(undef,0,0)
    y_test     = []
    if !isempty(ind_test)
        (x_test,_,_) = get_x(xyz_test,ind_test,features_setup;
                             features_no_norm = features_no_norm,
                             terms            = terms,
                             sub_diurnal      = sub_diurnal,
                             sub_igrf         = sub_igrf,
                             bpf_mag          = bpf_mag);
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
            elseif model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc]
                (model,TL_coef,data_norms,y_hat_fi,err_fi) =
                    nn_comp_3_train(A,Bt,B_dot,x_fi,y,no_norm;
                                    model_type  = model_type,
                                    norm_type_x = norm_type_x,
                                    norm_type_y = norm_type_y,
                                    TL_coef     = TL_coef,
                                    terms_A     = terms_A,
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
                                    A_test      = A_test,
                                    Bt_test     = Bt_test,
                                    B_dot_test  = B_dot_test,
                                    x_test      = x_test_fi,
                                    y_test      = y_test,
                                    silent      = silent)
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
                                       TL_coef    = vec(TL_coef))
            @save drop_fi_bson*"_$i.bson" comp_params

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
                                model       = model,
                                A_test      = A_test,
                                x_test      = x_test,
                                y_test      = y_test,
                                silent      = silent)
        elseif model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc]
            (model,TL_coef,data_norms,y_hat,err) =
                nn_comp_3_train(A,Bt,B_dot,x,y,no_norm;
                                model_type  = model_type,
                                norm_type_x = norm_type_x,
                                norm_type_y = norm_type_y,
                                TL_coef     = TL_coef,
                                terms_A     = terms_A,
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
                                A_test      = A_test,
                                Bt_test     = Bt_test,
                                B_dot_test  = B_dot_test,
                                x_test      = x_test,
                                y_test      = y_test,
                                silent      = silent)
        elseif model_type in [:TL,:mod_TL,:map_TL]
            trim = model_type in [:TL,:mod_TL] ? 20 : 0
            (model,data_norms,y_hat,err) =
                linear_fit(A,y;
                           trim=trim, λ=λ_TL,
                           norm_type_x = norm_type_A,
                           norm_type_y = norm_type_y,
                           silent      = silent)
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
                                   TL_coef    = vec(TL_coef))
    elseif typeof(comp_params) <: LinCompParams
        comp_params = LinCompParams(comp_params,
                                    data_norms = data_norms,
                                    model      = model)
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
- `lines`:       selected line number(s)
- `df_line`:     lookup table (DataFrame) of `lines`
- `df_flight`:   lookup table (DataFrame) of flight files
- `df_map`:      lookup table (DataFrame) of map files
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct, either:
    - `NNCompParams`:  neural network-based aeromagnetic compensation parameters struct
    - `LinCompParams`: linear aeromagnetic compensation parameters struct
- `silent`:      (optional) if true, no print outs

**Returns:**
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct
- `y`:           observed data
- `y_hat`:       predicted data
- `err`:         mean-corrected (per line) compensation error
- `features`:    full list of features (including components of TL `A`, etc.)
"""
function comp_train(lines, df_line::DataFrame, df_flight::DataFrame,
                    df_map::DataFrame, comp_params::CompParams=NNCompParams();
                    silent::Bool=true)

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
    if model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc]
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
                                                           silent           = true)
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
                                                  silent           = true)
    end

    y_hat = zero(y) # initialize
    err   = 10*y    # initialize

    if drop_fi

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
            elseif model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc]
                (model,TL_coef,data_norms,y_hat_fi,err_fi) =
                    nn_comp_3_train(A,Bt,B_dot,x_fi,y,no_norm;
                                    model_type  = model_type,
                                    norm_type_x = norm_type_x,
                                    norm_type_y = norm_type_y,
                                    TL_coef     = TL_coef,
                                    terms_A     = terms_A,
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
                                       data_norms = data_norms,
                                       model      = model,
                                       TL_coef    = vec(TL_coef))
            @save drop_fi_bson*"_$i.bson" comp_params

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
        elseif model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc]
            (model,TL_coef,data_norms,y_hat,err) =
                nn_comp_3_train(A,Bt,B_dot,x,y,no_norm;
                                model_type  = model_type,
                                norm_type_x = norm_type_x,
                                norm_type_y = norm_type_y,
                                TL_coef     = TL_coef,
                                terms_A     = terms_A,
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
                                data_norms  = data_norms,
                                model       = model,
                                l_segs      = l_segs,
                                silent      = silent)
        elseif model_type in [:TL,:mod_TL,:map_TL]
            trim = model_type in [:TL,:mod_TL] ? 20 : 0
            (model,data_norms,y_hat,err) =
                linear_fit(A,y;
                           trim=trim, λ=λ_TL,
                           norm_type_x = norm_type_A,
                           norm_type_y = norm_type_y,
                           l_segs      = l_segs,
                           silent      = silent)
        elseif model_type in [:elasticnet]
            (model,data_norms,y_hat,err) =
                elasticnet_fit(x,y;
                               l_segs      = l_segs,
                               silent      = silent)
        elseif model_type in [:plsr]
            (model,data_norms,y_hat,err) =
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
                                   model      = model,
                                   TL_coef    = vec(TL_coef))
    elseif typeof(comp_params) <: LinCompParams
        comp_params = LinCompParams(comp_params,
                                    data_norms = data_norms,
                                    model      = model)
    end

    print_time(time()-t0,1)

    return (comp_params, y, y_hat, err, features)
end # function comp_train

"""
    comp_test(xyz::XYZ, ind, mapS::MapS=MapS(zeros(1,1),[0.0],[0.0],0.0);
              comp_params::CompParams=NNCompParams(), silent::Bool=false)

Evaluate aeromagnetic compensation model performance.

**Arguments:**
- `xyz`:         `XYZ` flight data struct
- `ind`:         selected data indices
- `mapS`:        (optional) `MapS` scalar magnetic anomaly map struct, only used for `y_type = :b, :c`
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct, either:
    - `NNCompParams`:  neural network-based aeromagnetic compensation parameters struct
    - `LinCompParams`: linear aeromagnetic compensation parameters struct
- `silent`:      (optional) if true, no print outs

**Returns:**
- `y`:        observed data
- `y_hat`:    predicted data
- `err`:      compensation error
- `features`: full list of features (including components of TL `A`, etc.)
"""
function comp_test(xyz::XYZ, ind, mapS::MapS=MapS(zeros(1,1),[0.0],[0.0],0.0);
                   comp_params::CompParams=NNCompParams(), silent::Bool=false)

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
            elseif model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc]
                (y_hat_fi,err_fi) = nn_comp_3_test(A,Bt,B_dot,x_fi,y,data_norms,model;
                                                   model_type = model_type,
                                                   y_type     = y_type,
                                                   TL_coef    = TL_coef,
                                                   terms_A    = terms_A,
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
            (y_hat,err) = nn_comp_1_test(x,y,data_norms,model;
                                         silent     = silent)
        elseif model_type in [:m2a,:m2b,:m2c,:m2d]
            (y_hat,err) = nn_comp_2_test(A,x,y,data_norms,model;
                                         model_type = model_type,
                                         TL_coef    = TL_coef,
                                         silent     = silent)
        elseif model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc]
            (y_hat,err) = nn_comp_3_test(A,Bt,B_dot,x,y,data_norms,model;
                                         model_type = model_type,
                                         y_type     = y_type,
                                         TL_coef    = TL_coef,
                                         terms_A    = terms_A,
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

    print_time(time()-t0,1)

    return (y, y_hat, err, features)
end # function comp_test

"""
    comp_test(lines, df_line::DataFrame, df_flight::DataFrame,
              df_map::DataFrame, comp_params::CompParams=NNCompParams();
              silent::Bool=false)

Evaluate aeromagnetic compensation model performance.

**Arguments:**
- `lines`:       selected line number(s)
- `df_line`:     lookup table (DataFrame) of `lines`
- `df_flight`:   lookup table (DataFrame) of flight files
- `df_map`:      lookup table (DataFrame) of map files
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct, either:
    - `NNCompParams`:  neural network-based aeromagnetic compensation parameters struct
    - `LinCompParams`: linear aeromagnetic compensation parameters struct
- `silent`:      (optional) if true, no print outs

**Returns:**
- `y`:        observed data
- `y_hat`:    predicted data
- `err`:      mean-corrected (per line) compensation error
- `features`: full list of features (including components of TL `A`, etc.)
"""
function comp_test(lines, df_line::DataFrame, df_flight::DataFrame,
                   df_map::DataFrame, comp_params::CompParams=NNCompParams();
                   silent::Bool=false)

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
    if model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc]
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
                                                     silent           = true)
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
                                            silent           = true)
    end

    y_hat = zero(y) # initialize
    err   = 10*y    # initialize

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
            elseif model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc]
                (y_hat_fi,err_fi) = nn_comp_3_test(A,Bt,B_dot,x_fi,y,data_norms,model;
                                                   model_type = model_type,
                                                   y_type     = y_type,
                                                   TL_coef    = TL_coef,
                                                   terms_A    = terms_A,
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
            (y_hat,err) = nn_comp_1_test(x,y,data_norms,model;
                                         l_segs     = l_segs,
                                         silent     = silent)
        elseif model_type in [:m2a,:m2b,:m2c,:m2d]
            (y_hat,err) = nn_comp_2_test(A,x,y,data_norms,model;
                                         model_type = model_type,
                                         TL_coef    = TL_coef,
                                         l_segs     = l_segs,
                                         silent     = silent)
        elseif model_type in [:m3tl,:m3s,:m3v,:m3sc,:m3vc]
            (y_hat,err) = nn_comp_3_test(A,Bt,B_dot,x,y,data_norms,model;
                                         model_type = model_type,
                                         y_type     = y_type,
                                         TL_coef    = TL_coef,
                                         terms_A    = terms_A,
                                         l_segs     = l_segs,
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

    print_time(time()-t0,1)

    return (y, y_hat, err, features)
end # function comp_test

"""
    comp_m2bc_test(lines, df_line::DataFrame,
                   df_flight::DataFrame, df_map::DataFrame,
                   comp_params::NNCompParams=NNCompParams();
                   silent::Bool=false)

Evaluate neural network-based aeromagnetic compensation, model 2b or 2c
performance with additional outputs for explainability.

**Arguments:**
- `lines`:       selected line number(s)
- `df_line`:     lookup table (DataFrame) of `lines`
- `df_flight`:   lookup table (DataFrame) of flight files
- `df_map`:      lookup table (DataFrame) of map files
- `comp_params`: neural network-based aeromagnetic compensation parameters struct
- `silent`:      (optional) if true, no print outs

**Returns:**
- `y_nn`:     neural network compensation portion
- `y_TL`:     Tolles-Lawson  compensation portion
- `y`:        observed data
- `y_hat`:    predicted data
- `err`:      mean-corrected (per line) compensation error
- `features`: full list of features (including components of TL `A`, etc.)
"""
function comp_m2bc_test(lines, df_line::DataFrame,
                        df_flight::DataFrame, df_map::DataFrame,
                        comp_params::NNCompParams=NNCompParams();
                        silent::Bool=false)

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
                                        silent           = true,)

    # convert to Float32 for consistency with nn_comp_2_train
    A       = convert.(Float32,A)
    x       = convert.(Float32,x)
    y       = convert.(Float32,y)
    TL_coef = convert.(Float32,TL_coef)

    # unpack data normalizations
    (A_bias,A_scale,v_scale,x_bias,x_scale,y_bias,y_scale) =
        unpack_data_norms(data_norms)
    A_norm =  (A .- A_bias) ./ A_scale
    x_norm = ((x .- x_bias) ./ x_scale) * v_scale

    TL_coef = TL_coef/y_scale

    # set to test mode in case model uses batchnorm or dropout
    m = model
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
- `lines`:       selected line number(s)
- `df_line`:     lookup table (DataFrame) of `lines`
- `df_flight`:   lookup table (DataFrame) of flight files
- `df_map`:      lookup table (DataFrame) of map files
- `comp_params`: neural network-based aeromagnetic compensation parameters struct
- `silent`:      (optional) if true, no print outs

**Returns:**
- `TL_perm`:      TL permanent vector field
- `TL_induced`:   TL induced vector field
- `TL_eddy`:      TL eddy current vector field
- `TL_aircraft`:  TL aircraft vector field
- `B_vec`:        vector magnetometer measurements
- `B_unit`:       normalized `B_vec`
- `y_NN`:         vector neural network correction (for scalar models, in direction of `Bt`)
- `vec_aircraft`: estimated vector aircraft field
- `y`:            observed scalar magnetometer data
- `y_hat`:        predicted scalar magnetometer data
- `err`:          mean-corrected (per line) compensation error
- `features`:     full list of features (including components of TL `A`, etc.)
"""
function comp_m3_test(lines, df_line::DataFrame,
                      df_flight::DataFrame, df_map::DataFrame,
                      comp_params::NNCompParams=NNCompParams();
                      silent::Bool=false)

    seed!(2) # for reproducibility
    t0 = time()

    # unpack parameters
    @unpack version, features_setup, features_no_norm, model_type, y_type,
    use_mag, use_vec, data_norms, model, terms, terms_A, sub_diurnal,
    sub_igrf, bpf_mag, reorient_vec, norm_type_A, norm_type_x, norm_type_y,
    TL_coef, η_adam, epoch_adam, epoch_lbfgs, hidden, activation,
    batchsize, frac_train, α_sgl, λ_sgl, k_pca,
    drop_fi, drop_fi_bson, drop_fi_csv, perm_fi, perm_fi_csv = comp_params

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
                                                 silent           = silent)

    # convert to Float32 for consistency with nn_comp_3_train
    A       = convert.(Float32,A)
    Bt      = convert.(Float32,Bt)    # magnitude of total field measurements
    B_dot   = convert.(Float32,B_dot) # finite differences of total field vector
    x       = convert.(Float32,x)
    y       = convert.(Float32,y)
    TL_coef = convert.(Float32,TL_coef)

    # assume all terms are stored, but they may be zero if not trained
    Bt_scale = 50000f0
    (TL_coef_p,TL_coef_i,TL_coef_e) = extract_TL_matrices(TL_coef,terms_A;Bt_scale=Bt_scale)

    B_unit    = A[:,1:3]     # normalized vector magnetometer reading
    B_vec     = B_unit .* Bt # vector magnetometer to be used in TL
    B_vec_dot = B_dot        # not exactly true, but internally consistent

    B_unit    = transpose(B_unit)
    B_vec     = transpose(B_vec)
    B_vec_dot = transpose(B_vec_dot)

    # unpack data normalizations
    (_,_,v_scale,x_bias,x_scale,y_bias,y_scale) = unpack_data_norms(data_norms)
    x_norm = ((x .- x_bias) ./ x_scale) * v_scale

    if y_type in [:c,:d]
        y_type_aircraft = true
    elseif y_type in [:a,:b]
        y_type_aircraft = false
    else
        error("unsupported y_type in comp_m3_test, $y_type")
    end

    # set to test mode in case model uses batchnorm or dropout
    m = model
    Flux.testmode!(m)

    # calculate TL vector field
    (TL_aircraft,TL_perm,TL_induced,TL_eddy) =
        get_TL_aircraft_vector(B_vec,B_vec_dot,TL_coef_p,TL_coef_i,TL_coef_e;
                               return_parts=true)

    # compute neural network correction
    y_NN = zeros(size(B_vec))
    if model_type in [:m3s,:m3sc]
       # scalar-corrected, dimensionless; rescale to TL and assume same direction
       y_NN = vec(y_scale .* m(transpose(x_norm)))
       y_NN = transpose(y_NN .* transpose(B_unit))
    elseif model_type in [:m3v,:m3vc]
       y_NN = y_scale .* m(transpose(x_norm))  # vector-corrected 3xN, just rescale
    else
        error("Unsupported model designation for Model 3 explainability")
    end

    # compute aircraft or Earth field target
    vec_aircraft = TL_aircraft + y_NN
    if y_type_aircraft
        y_hat = vec(sum(vec_aircraft .* B_unit, dims=1))
    else
        Be = B_vec - vec_aircraft
        y_hat = vec(sqrt.(sum(Be.^2,dims=1)))
    end

    err = err_segs(y_hat,y,l_segs;silent=silent)
    @info("test  error: $(round(std(err),digits=2)) nT")

    print_time(time()-t0,1)

    return (TL_perm, TL_induced, TL_eddy, TL_aircraft,
            B_vec, B_unit, y_NN, vec_aircraft, y, y_hat, err, features)
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
- `silent`:      (optional) if true, no print outs

**Returns:**
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct
- `y_train`:     training observed data
- `y_train_hat`: training predicted data
- `err_train`:   training compensation error
- `y_test`:      testing observed data
- `y_test_hat`:  testing predicted data
- `err_test`:    testing compensation error
- `features`:    full list of features (including components of TL `A`, etc.)
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
                    silent::Bool=true)

Train and evaluate aeromagnetic compensation model performance.

**Arguments:**
- `lines_train`: selected line number(s) for training
- `lines_test`:  selected line number(s) for testing
- `df_line`:     lookup table (DataFrame) of `lines`
- `df_flight`:   lookup table (DataFrame) of flight files
- `df_map`:      lookup table (DataFrame) of map files
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct, either:
    - `NNCompParams`:  neural network-based aeromagnetic compensation parameters struct
    - `LinCompParams`: linear aeromagnetic compensation parameters struct
- `silent`:      (optional) if true, no print outs

**Returns:**
- `comp_params`: `CompParams` aeromagnetic compensation parameters struct
- `y_train`:     training observed data
- `y_train_hat`: training predicted data
- `err_train`:   mean-corrected (per line) training compensation error
- `y_test`:      testing observed data
- `y_test_hat`:  testing predicted data
- `err_test`:    mean-corrected (per line) testing compensation error
- `features`:    full list of features (including components of TL `A`, etc.)
"""
function comp_train_test(lines_train, lines_test,
                         df_line::DataFrame, df_flight::DataFrame,
                         df_map::DataFrame, comp_params::CompParams=NNCompParams();
                         silent::Bool=true)

    (comp_params,y_train,y_train_hat,err_train,features) =
        comp_train(lines_train,df_line,df_flight,df_map,comp_params;silent=silent)

    (y_test,y_test_hat,err_test,_) =
        comp_test(lines_test,df_line,df_flight,df_map,comp_params;silent=silent)

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
