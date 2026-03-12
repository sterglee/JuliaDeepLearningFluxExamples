# 26.64 secs
# accuracy  66.67

using HTTP
using Flux
using Flux: onehotbatch, DataLoader, @epochs
using Statistics
using MLUtils # For splitobs

# 1. DATA LOADING & PREPROCESSING
function download_and_preprocess()
    println("Fetching dataset from UCI...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data"
    
    # HTTP request inside the function scope
    response = HTTP.get(url)
    raw_content = String(response.body)
    raw_lines = split(strip(raw_content), "\n")

    alphabet = ['a', 'g', 'c', 't']
    X = []
    Y = []

    for line in raw_lines
        parts = split(line, ",")
        if length(parts) < 3 continue end
        
        # Label: + is promoter (class 1), - is non-promoter (class 2)
        label = strip(parts[1]) == "+" ? 1 : 2
        
        # Sequence: clean up whitespace
        seq = replace(strip(parts[3]), r"\s+" => "")
        
        # One-hot encode: resulting in (4, 57)
        # We lowercase to match the alphabet ['a', 'g', 'c', 't']
        encoded = Float32.(onehotbatch(collect(lowercase(seq)), alphabet))
        
        # Correct dimensions for 1D Conv: (Width/Length, Channels)
        # Transposing (4, 57) to (57, 4)
        push!(X, collect(encoded'))
        push!(Y, onehotbatch(label, 1:2))
    end

    # Combine into 3D tensor: (57, 4, 106) -> (Length, Channels, Batch)
    X_tensor = cat(X..., dims=3)
    Y_tensor = cat(Y..., dims=2)
    
    return X_tensor, Y_tensor
end

# 2. MODEL DEFINITION
function build_model()
    return Chain(
        # Input shape: (57, 4, Batch)
        # Conv((filter_size,), in_channels => out_channels)
        Conv((7,), 4 => 16, relu, pad=SamePad()),
        MaxPool((2,)), # Reduces sequence length from 57 to 28
        
        Conv((3,), 16 => 32, relu, pad=SamePad()),
        MaxPool((2,)), # Reduces sequence length from 28 to 14
        
        Flux.flatten,
        # Flattened size: 14 (length) * 32 (channels) = 448
        Dense(448, 16, relu), 
        Dense(16, 2),
        softmax
    )
end

# 3. UTILITIES
accuracy(m, x, y) = mean(Flux.onecold(m(x)) .== Flux.onecold(y))

# 4. MAIN EXECUTION
function main()
    # Data Setup
    X_raw, Y_raw = download_and_preprocess()
    
    # Split: 80% Train, 20% Test
    (x_train, y_train), (x_test, y_test) = splitobs((X_raw, Y_raw), at=0.8)
    
    # Create DataLoaders
    train_loader = DataLoader((x_train, y_train), batchsize=16, shuffle=true)
    
    # Initialize Model and Optimizer
    model = build_model()
    # Note: Use logitcrossentropy if the model DOES NOT have softmax
    # Since we have softmax, we'll use crossentropy
    loss(m, x, y) = Flux.crossentropy(m(x), y)
    opt_state = Flux.setup(Adam(0.001), model)

    println("\n--- Starting Training ---")
    initial_acc = accuracy(model, x_test, y_test)
    println("Initial Test Accuracy: $(round(initial_acc*100, digits=2))%")

    for epoch in 1:1000
        for (x, y) in train_loader
            # Compute gradient and update weights
            grads = Flux.gradient(m -> loss(m, x, y), model)
            Flux.update!(opt_state, model, grads[1])
        end
        
        if epoch % 10 == 0
            acc = accuracy(model, x_test, y_test)
            println("Epoch $epoch | Test Accuracy: $(round(acc*100, digits=2))%")
        end
    end

    final_acc = accuracy(model, x_test, y_test)
    println("-----------------------")
    println("Final Test Accuracy: $(round(final_acc*100, digits=2))%")
end

# Final check: call the main function
@time main()