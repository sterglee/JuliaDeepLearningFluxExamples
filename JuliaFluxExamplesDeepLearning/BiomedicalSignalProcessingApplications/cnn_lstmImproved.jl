# Python time 35.7s per epoch
# Julia time 34.6s per epoch

using Flux
using Flux: onehotbatch, onecold, DataLoader
using Images
using CSV
using DataFrames
using MLUtils 
using Statistics
using CUDA
using ProgressMeter
using Dates

# --- 1. Preprocessing ---

function load_and_preprocess(path, target_size=(224, 224))
    # Load and resize
    img = load(path)
    img_resized = imresize(img, target_size)
    
    # Julia/Flux uses WHCN format: (Width, Height, Channels, Batch)
    # channelview is (C, H, W) -> we permute to (W, H, C)
    img_array = Float32.(channelview(RGB.(img_resized)))
    img_array = permutedims(img_array, (3, 2, 1)) 
    
    # Normalize to [-1, 1] using Float32 literals (f0)
    return (img_array .- 0.5f0) ./ 0.5f0
end

# --- 2. Model Definition ---

function create_model(num_classes)
    return Chain(
        # Feature Extraction
        Conv((3, 3), 3 => 32, relu, pad=1), BatchNorm(32), MaxPool((2, 2)),
        Conv((3, 3), 32 => 64, relu, pad=1), BatchNorm(64), MaxPool((2, 2)),
        Conv((3, 3), 64 => 128, relu, pad=1), BatchNorm(128), MaxPool((2, 2)),
        Conv((3, 3), 128 => 256, relu, pad=1), BatchNorm(256), MaxPool((2, 2)),
        
        # Classifier
        Flux.flatten,
        Dense(256 * 14 * 14, 512, relu),
        Dropout(0.5f0),
        Dense(512, num_classes)
    )
end

# --- 3. Main Logic ---

function main()
    script_start = now()
    device = CUDA.functional() ? gpu : cpu
    @info "Using device: $(device == gpu ? "CUDA GPU" : "CPU")"

    # Data Loading
    df = CSV.read("HAM10000_metadata.csv", DataFrame)
    
    # Resolve Image Paths
    paths = map(id -> begin
        p1 = "HAM10000_images_part_1/$id.jpg"
        isfile(p1) ? p1 : "HAM10000_images_part_2/$id.jpg"
    end, df.image_id)

    # Label Encoding
    labels_unique = unique(df.dx)
    label_map = Dict(label => i for (i, label) in enumerate(labels_unique))
    labels = [label_map[x] for x in df.dx]
    num_classes = length(labels_unique)

    # Data Splitting (70/15/15)
    data = (paths, labels)
    (train_data, tmp_data) = splitobs(data, at=0.7, shuffle=true)
    (val_data, test_data) = splitobs(tmp_data, at=0.5, shuffle=false)

    # Lazy DataLoaders
    function make_loader(obs; shuffle=false)
        mapped = mapobs(obs) do (p, l)
            return load_and_preprocess(p), onehotbatch(l, 1:num_classes)
        end
        return DataLoader(mapped, batchsize=32, shuffle=shuffle, collate=true)
    end

    train_loader = make_loader(train_data, shuffle=true)
    val_loader   = make_loader(val_data)
    test_loader  = make_loader(test_data)

    # Model and Optimizer
    model = create_model(num_classes) |> device
    opt_state = Flux.setup(Adam(1f-4), model)

    # Training Loop
    println("\nStarting Training (5 Epochs)...")
    for epoch in 1:5
        epoch_start = now()
        epoch_loss = 0.0f0
        
        Flux.trainmode!(model)
        @showprogress "Epoch $epoch: " for (x, y) in train_loader
            x, y = x |> device, y |> device
            
            l, grads = Flux.withgradient(model) do m
                y_hat = m(x)
                Flux.logitcrossentropy(y_hat, y)
            end
            
            Flux.update!(opt_state, model, grads[1])
            epoch_loss += l
        end

        # Validation
        Flux.testmode!(model)
        correct, total = 0, 0
        for (x, y) in val_loader
            x, y = x |> device, y |> device
            preds = onecold(model(x))
            actual = onecold(y)
            correct += sum(preds .== actual)
            total += length(actual)
        end
        
        val_acc = correct / total
        duration = (now() - epoch_start).value / 1000
        
        println("Epoch $epoch | Loss: $epoch_loss/length(train_loader) | Val Acc: $val_acc | Time: $duration)s")
        println("-"^50)
    end

    # Final Evaluation
    @info "Evaluating on Test Set..."
    test_start = now()
    Flux.testmode!(model)
    y_true, y_pred = Int[], Int[]
    
    for (x, y) in test_loader
        x = x |> device
        append!(y_pred, onecold(model(x)))
        append!(y_true, onecold(y))
    end
    
    test_acc = mean(y_true .== y_pred)
    total_time = (now() - script_start).value / 1000
    
    println("\n" * "="^20)
    println("Test Accuracy: $(round(test_acc, digits=4))")
    println("Total Execution Time: $(round(total_time, 2)) seconds")
    println("="^20)
end

main()

