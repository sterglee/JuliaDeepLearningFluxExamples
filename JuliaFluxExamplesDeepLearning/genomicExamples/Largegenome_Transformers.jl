using Flux
using Flux: onehotbatch, DataLoader, onecold, logitcrossentropy
using HTTP
using Statistics
using MLUtils: splitobs
using Printf
using Random

# -------------------------
# CONSTANTS
# -------------------------
const SEQ_LEN = 57
const VOCAB = Dict('a'=>1, 'c'=>2, 'g'=>3, 't'=>4)

# -------------------------
# 1. DATA LOADING
# -------------------------
function encode_seq(seq)
    seq = filter(c -> haskey(VOCAB, c), lowercase(seq))
    return [VOCAB[c] for c in seq]
end

function download_data()
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data"
    println("Downloading from UCI Repository...")
    response = HTTP.get(url)
    
    # Convert body bytes to string and split into lines
    raw_data = String(response.body)
    lines = split(raw_data, "\n")

    X_list = Vector{Vector{Int}}()
    Y_list = Vector{Int}()

    for line in lines
        parts = split(line, ",")
        if length(parts) < 3
            continue
        end

        # UCI Format: part[1] is +/- , part[3] is the sequence
        label = strip(parts[1]) == "+" ? 1 : 2
        seq = replace(strip(parts[3]), " " => "") # Remove any internal whitespace

        encoded = encode_seq(seq)
        if length(encoded) == SEQ_LEN
            push!(X_list, encoded)
            push!(Y_list, label)
        end
    end

    X = hcat(X_list...)             # Shape: (57, N)
    Y = onehotbatch(Y_list, 1:2)    # Shape: (2, N)

    return X, Y
end

# -------------------------
# 2. MODEL (CNN for Sequences)
# -------------------------
function build_model()
    return Chain(
        # 1. Map indices to vectors: (57, Batch) -> (32, 57, Batch)
        Embedding(4, 32),

        # 2. Reshape for Conv1D: (32, 57, Batch) -> (57, 32, Batch)
        # Width (Time) = 57, Channels = 32
        x -> permutedims(x, (2, 1, 3)),

        # 3. Convolutional Layers
        Conv((5,), 32 => 64, relu, pad=SamePad()),
        Conv((3,), 64 => 128, relu, pad=SamePad()),

        # 4. Global Mean Pooling (along the sequence length)
        x -> mean(x, dims=1),

        # 5. Flatten and Dense
        Flux.flatten,
        Dense(128, 64, relu),
        Dropout(0.3),
        Dense(64, 2)
    )
end

# -------------------------
# 3. METRICS
# -------------------------
accuracy(ŷ, y) = mean(onecold(ŷ) .== onecold(y))

# -------------------------
# 4. TRAINING
# -------------------------
function train!(model, x_train, y_train, x_test, y_test)
    loader = DataLoader((x_train, y_train), batchsize=16, shuffle=true)
    opt_state = Flux.setup(Adam(1e-3), model)

    println("Starting Training...")
    for epoch in 1:1000
        total_loss = 0f0

        for (x, y) in loader
            loss, grads = Flux.withgradient(model) do m
                ŷ = m(x)
                logitcrossentropy(ŷ, y)
            end

            Flux.update!(opt_state, model, grads[1])
            total_loss += loss
        end

        if epoch % 20 == 0 || epoch == 1
            ŷ_test = model(x_test)
            acc = accuracy(ŷ_test, y_test)
            @printf("Epoch %3d | Loss %.4f | Test Acc %.2f%%\n",
                epoch, total_loss / length(loader), acc * 100)
        end
    end
end

# -------------------------
# 5. MAIN
# -------------------------
function main()
    Random.seed!(42)

    X, Y = download_data()
    println("Dataset loaded. Total samples: $(size(X, 2))")

    # Split data (80% Train, 20% Test)
    (x_train, y_train), (x_test, y_test) = splitobs((X, Y), at=0.8)

    model = build_model()

    train!(model, x_train, y_train, x_test, y_test)

    final_acc = accuracy(model(x_test), y_test)
    println("\n" * "="^30)
    println("Final Accuracy: ", round(final_acc * 100, digits=2), "%")
    println("="^30)
end

main()

