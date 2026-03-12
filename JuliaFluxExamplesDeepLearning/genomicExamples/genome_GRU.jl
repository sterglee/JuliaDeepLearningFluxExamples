#time 13.7
# acuuracy 0.0%
using HTTP, Flux, Statistics, MLUtils, Random
using Flux: onehotbatch, onehot, DataLoader, reset!, onecold

# ------------------------------------------------------------
# 1. DATA LOADING
# ------------------------------------------------------------
function download_and_preprocess()
    println("Fetching dataset from UCI...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data"
    response = HTTP.get(url)
    raw_lines = split(strip(String(response.body)), "\n")

    alphabet = ['a','g','c','t']
    X = Vector{Array{Float32,2}}()
    Y = Vector{Flux.OneHotVector}()

    for line in raw_lines
        parts = split(line, ",")
        if length(parts) < 3 continue end

        label = strip(parts[1]) == "+" ? 1 : 2
        seq = lowercase(replace(strip(parts[3]), r"\s+" => ""))

        if length(seq) != 57 continue end

        push!(X, Float32.(onehotbatch(collect(seq), alphabet)))
        push!(Y, onehot(label, 1:2))
    end

    return cat(X..., dims=3), cat(Y..., dims=2)
end

# ------------------------------------------------------------
# 2. GRU MODEL
# ------------------------------------------------------------
function build_gru_model()
    hidden_dim = 32
    return Chain(
        # GRU is often more efficient for short genomic sequences
        GRU(4 => hidden_dim),

        # Capture the final state after the 57th base pair
        x -> x[:, end, :], 

        Dense(hidden_dim, 16, relu),
        Dense(16, 2),
        softmax
    )
end

# ------------------------------------------------------------
# 3. UTILITIES
# ------------------------------------------------------------
function calculate_accuracy(model, x, y)
    Flux.reset!(model)
    return mean(onecold(model(x)) .== onecold(y))
end

# ------------------------------------------------------------
# 4. MAIN
# ------------------------------------------------------------
function main()
    Random.seed!(42)
    X_data, Y_data = download_and_preprocess()

    (x_train, y_train), (x_test, y_test) = splitobs((X_data, Y_data), at = 0.8)

    train_loader = DataLoader((x_train, y_train), batchsize = 16, shuffle = true)

    model = build_gru_model()
    opt_state = Flux.setup(Adam(0.001), model)

    println("\n--- Training GRU ---")
    for epoch in 1:1000
        for (x, y) in train_loader
            grads = Flux.gradient(model) do m
                Flux.reset!(m)
                Flux.crossentropy(m(x), y)
            end
            Flux.update!(opt_state, model, grads[1])
        end

        if epoch % 20 == 0
            acc = calculate_accuracy(model, x_test, y_test)
            println("Epoch $epoch | Test Accuracy: $(round(acc*100, digits=2))%")
            if acc >= 0.98 break end
        end
    end

    final_acc = calculate_accuracy(model, x_test, y_test)
    println("---------------------------")
    println("Final GRU Accuracy: $(round(final_acc*100, digits=2))%")
end

@time main()