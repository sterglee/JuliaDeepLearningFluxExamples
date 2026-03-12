
# MLP Test Accuracy: 76.92%
#  0.757349 seconds (979.01 k allocations: 283.569 MiB, 12.62% gc time, 1 lock conflict, 39.48% compilation time)

using HTTP, Flux, Statistics, Random
using Flux: onehotbatch, DataLoader, logitcrossentropy, onecold

# 1. DATA PREPROCESSING
function load_data()
    println("Fetching dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data"
    response = HTTP.get(url)
    raw_lines = split(strip(String(response.body)), "\n")

    alphabet = ['a', 'g', 'c', 't']
    X, Y = [], []

    for line in raw_lines
        parts = split(line, ",")
        if length(parts) < 3 continue end

        # Label: + is 1, - is 2
        label = strip(parts[1]) == "+" ? 1 : 2
        seq = lowercase(replace(strip(parts[3]), r"\s+" => ""))
        if length(seq) != 57 continue end

        # Flattened One-Hot Encoding (228 features)
        encoded = Float32.(onehotbatch(collect(seq), alphabet))
        push!(X, vec(encoded))
        push!(Y, Flux.onehot(label, 1:2))
    end
    return hcat(X...), hcat(Y...)
end

# 2. MLP ARCHITECTURE
function build_mlp()
    # Ensure the Chain starts immediately or use parentheses to wrap the return
    return Chain(
        Dense(228, 128, relu),
        Dropout(0.3), 
        Dense(128, 64, relu),
        Dense(64, 2)
    )
end

# 3. TRAINING
function train_model()
    Random.seed!(42)
    X, Y = load_data()

    # Simple Train/Test Split
    train_idx = 1:80
    test_idx = 81:106

    model = build_mlp()
    loader = DataLoader((X[:, train_idx], Y[:, train_idx]), batchsize=16, shuffle=true)
    opt_state = Flux.setup(Adam(0.001), model)

    println("Training MLP...")
    for epoch in 1:200 # 1000 might be overkill, usually stabilizes by 200
        for (x, y) in loader
            # Ensure model is in train mode (default, but good to be explicit)
            trainmode!(model) 
            grads = Flux.gradient(m -> logitcrossentropy(m(x), y), model)
            Flux.update!(opt_state, model, grads[1])
        end
    end

    # 4. EVALUATION
    # CRITICAL: Disable Dropout for testing
    testmode!(model)
    y_hat = model(X[:, test_idx])
    accuracy = mean(onecold(y_hat) .== onecold(Y[:, test_idx]))
    println("MLP Test Accuracy: $(round(accuracy * 100, digits=2))%")
end

@time train_model()

