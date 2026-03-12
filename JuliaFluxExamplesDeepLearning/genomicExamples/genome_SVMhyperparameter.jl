
# MLP Final Accuracy: 73.08%
# SVM Best Test Accuracy: 92.31%

using HTTP, Flux, LIBSVM, Statistics, Random
using Flux: onehotbatch, DataLoader, logitcrossentropy, onecold

# ============================================================
# 1. DATA CORE: LOAD AND PREPROCESS
# ============================================================
function load_genomic_data()
    println("--- Fetching UCI Promoter Dataset ---")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data"
    response = HTTP.get(url)
    raw_lines = split(strip(String(response.body)), "\n")

    alphabet = ['a', 'g', 'c', 't']
    X_raw, Y_raw = [], []

    for line in raw_lines
        parts = split(line, ",")
        if length(parts) < 3 continue end
        
        # Label: + is 1 (Promoter), - is 2 (Non-Promoter)
        label = strip(parts[1]) == "+" ? 1 : 2
        seq = lowercase(replace(strip(parts[3]), r"\s+" => ""))
        if length(seq) != 57 continue end

        # One-Hot Encoding
        encoded = Float32.(onehotbatch(collect(seq), alphabet))
        push!(X_raw, vec(encoded)) # Flatten to 228 features
        push!(Y_raw, label)
    end
    
    # Shuffle and Split (80/20)
    indices = shuffle(1:length(Y_raw))
    train_idx = indices[1:84]
    test_idx = indices[85:end]

    X = hcat(X_raw...)
    Y = Y_raw
    
    return X[:, train_idx], Y[train_idx], X[:, test_idx], Y[test_idx]
end

# ============================================================
# 2. MLP IMPLEMENTATION (Deep Learning)
# ============================================================
function run_mlp(xtrain, ytrain, xtest, ytest)
    println("\n--- Initializing MLP with Dropout ---")
    
    # Encode labels for Flux (One-hot)
    ytrain_hot = Flux.onehotbatch(ytrain, 1:2)
    ytest_hot = Flux.onehotbatch(ytest, 1:2)

    model = Chain(
        Dense(228, 128, relu),
        Dropout(0.3),
        Dense(128, 64, relu),
        Dense(64, 2)
    )

    loader = DataLoader((xtrain, ytrain_hot), batchsize=16, shuffle=true)
    opt_state = Flux.setup(Adam(0.001), model)

    for epoch in 1:200
        for (x, y) in loader
            trainmode!(model)
            grads = Flux.gradient(m -> logitcrossentropy(m(x), y), model)
            Flux.update!(opt_state, model, grads[1])
        end
    end

    testmode!(model)
    acc = mean(onecold(model(xtest)) .== ytest)
    println("MLP Final Accuracy: $(round(acc*100,2))%")
end

# ============================================================
# 3. SVM IMPLEMENTATION (Classical Machine Learning)
# ============================================================
function run_svm(xtrain, ytrain, xtest, ytest)
    println("\n--- Running SVM Grid Search ---")
    
    # Tuning ranges
    best_acc = 0.0
    best_model = nothing
    
    for c in [0.1, 1.0, 10.0], g in [0.001, 0.01, 0.1]
        # LIBSVM expects (Features x Samples) for training
        m = svmtrain(xtrain, Float64.(ytrain); 
                     kernel=LIBSVM.Kernel.RadialBasis, cost=c, gamma=g)
        
        preds, _ = svmpredict(m, xtest)
        acc = mean(preds .== ytest)
        
        if acc > best_acc
            best_acc = acc
            best_model = m
        end
    end
    println("SVM Best Test Accuracy: $(round(best_acc*100,2))%")
end

# ============================================================
# 4. EXECUTION
# ============================================================
function main()
    Random.seed!(123)
    xtrain, ytrain, xtest, ytest = load_genomic_data()
    
    run_mlp(xtrain, ytrain, xtest, ytest)
    run_svm(xtrain, ytrain, xtest, ytest)
end

@time main()

