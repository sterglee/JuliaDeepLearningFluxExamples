#time  68.4
#accuracy 0.0
using HTTP, Flux, Statistics, MLUtils, Random
using Flux: onehotbatch, onehot, DataLoader, reset!, onecold

# 1. DATA LOADING
function download_and_preprocess()
    println("Fetching dataset from UCI...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data"
    response = HTTP.get(url)
    raw_lines = split(strip(String(response.body)), "\n")

    alphabet = ['a','g','c','t']
    X = Vector{Array{Float32,2}}()
    Y = Vector{Flux.OneHotVector}() # Specialized type for performance

    for line in raw_lines
        parts = split(line, ",")
        if length(parts) < 3 continue end

        label = strip(parts[1]) == "+" ? 1 : 2
        seq = lowercase(replace(strip(parts[3]), r"\s+" => ""))

        if length(seq) != 57 continue end

        # X: (4, 57) | Y: One-hot vector of length 2
        push!(X, Float32.(onehotbatch(collect(seq), alphabet)))
        push!(Y, onehot(label, 1:2))
    end

    # X_tensor: (4, 57, N)
    # Y_tensor: (2, N)
    return cat(X..., dims=3), cat(Y..., dims=2)
end

# 2. MODEL
function build_lstm_model()
    hidden_dim = 128
    return Chain(
        LSTM(4 => hidden_dim),
        x -> x[:, end, :],   # Extract final hidden state: (hidden_dim, Batch)
        Dense(hidden_dim, 16, relu),
        Dense(16, 2),
        softmax
    )
end

# 3. ACCURACY
function calculate_accuracy(model, x, y)
    Flux.reset!(model)
    # Compare indices (1 or 2)
    return mean(onecold(model(x)) .== onecold(y))
end

# 4. MAIN
function main()
    Random.seed!(42)
    X_data, Y_data = download_and_preprocess()

    # Split into 80% train, 20% test
    (x_train, y_train), (x_test, y_test) = splitobs((X_data, Y_data), at = 0.8)

    train_loader = DataLoader((x_train, y_train), batchsize = 16, shuffle = true)

    model = build_lstm_model()
    opt_state = Flux.setup(Adam(0.001), model)

    println("\n--- Training ---")
    for epoch in 1:1000 
         for (x, y) in train_loader
            # We use a do-block for the gradient for cleaner syntax
            grads = Flux.gradient(model) do m
                Flux.reset!(m)
                Flux.crossentropy(m(x), y)
            end
            Flux.update!(opt_state, model, grads[1])
        end

        if epoch % 20 == 0
            acc = calculate_accuracy(model, x_test, y_test)
            println("Epoch $epoch | Test Accuracy: $(round(acc*100, digits=2))%")
            if acc >= 0.95 # Early stopping if we hit high accuracy
                println("Target accuracy reached.")
                break
            end
        end
    end

    final_acc = calculate_accuracy(model, x_test, y_test)
    println("---------------------------")
    println("Final Accuracy: $(round(final_acc*100, digits=2))%")
end

@time main()


