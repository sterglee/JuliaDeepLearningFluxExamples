using Flux: @functor, DataLoader, logitcrossentropy, cpu, gpu, onecold, softmax, GRU
using Optimisers
using CSV
using DataFrames
using Statistics
using Images, FileIO, ColorTypes
using CUDA
using JLD2

# --- 1. GPU Check ---
if CUDA.functional()
    println("CUDA GPU is functional!")
    device = gpu
else
    println("CUDA GPU not found, using CPU.")
    device = cpu
end

# --- 2. Data Processing ---

function load_and_preprocess(path)
    try
        img = load(path)
        img = Gray.(img)
        img = imresize(img, (32, 32))
        return reshape(Float32.(img), 32, 32, 1, 1) # (W, H, C, 1)
    catch e
        @warn "Failed to load $path: $e"
        return zeros(Float32, 32, 32, 1, 1)
    end
end

function get_dataset(csv_path; batchsize=32)
    df = CSV.read(csv_path, DataFrame)
    # Concatenate all images into a 4D tensor (32, 32, 1, Batch)
    imgs = [load_and_preprocess(row.image) for row in eachrow(df)]
    X = reduce((a, b) -> cat(a, b, dims=4), imgs)
    
    # Labels 0, 1, 2 -> 1, 2, 3 for One-Hot
    Y = Flux.onehotbatch(df.label .+ 1, 1:3)
    
    return DataLoader((X, Y), batchsize=batchsize, shuffle=true)
end

# --- 3. GRU Model Architecture ---

# This replaces the PatchEmbedding used in ViT
struct ImageToSequence
    projection::Conv
end
@functor ImageToSequence

function ImageToSequence(patch_size=4, in_ch=1, embed_dim=128)
    # Divide 32x32 image into 4x4 patches, projecting each to embed_dim
    projection = Conv((patch_size, patch_size), in_ch => embed_dim, stride=patch_size)
    return ImageToSequence(projection)
end

function (m::ImageToSequence)(x)
    # Input x: (32, 32, 1, Batch)
    x = m.projection(x) # Output: (8, 8, 128, Batch)
    # Reshape to (Features, SequenceLength, Batch)
    # SequenceLength = (32/4) * (32/4) = 64
    return reshape(x, size(x, 3), :, size(x, 4))
end

function create_model()
    return Chain(
        ImageToSequence(4, 1, 128),
        GRU(128, 128),              # Recurrent layer
        x -> x[:, end, :],          # Extract the final hidden state (summary)
        Dense(128, 64, relu),
        Dense(64, 3)                # 3 Classes
    )
end

# --- 4. Evaluation Function ---

function evaluate_performance(model, loader)
    Flux.testmode!(model)
    all_preds = Int[]
    all_labels = Int[]

    for (x, y) in loader
        x_dev = x |> device
        # IMPORTANT: Reset GRU hidden state for every batch
        Flux.reset!(model)
        
        logits = model(x_dev) |> cpu
        preds = onecold(logits, 1:3)
        actuals = onecold(y, 1:3)

        append!(all_preds, preds)
        append!(all_labels, actuals)
    end

    acc = mean(all_preds .== all_labels)

    # Per-Class Precision, Recall, F1
    metrics = map(1:3) do c
        tp = sum((all_preds .== c) .& (all_labels .== c))
        fp = sum((all_preds .== c) .& (all_labels .!= c))
        fn = sum((all_preds .!= c) .& (all_labels .== c))

        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * (prec * rec) / (prec + rec + 1e-9)
        return (precision=prec, recall=rec, f1=f1)
    end

    return acc, metrics
end

# --- 5. Training Loop ---

function train_model(csv_path, model_path, epochs)
    csv_path = "data.csv";
    model_path =  "models/model_gru.jld2";
    epochs = 2;
    loader = get_dataset(csv_path)
    model = create_model() |> device
    opt_state = Flux.setup(Flux.Adam(0.001f0), model)

    println("Starting GRU training on $(device)...")
    
    for epoch in 1:epochs
        loss_total = 0.0f0
        Flux.trainmode!(model)
        
        for (x, y) in loader
            x_dev, y_dev = x |> device, y |> device
            
            # Reset hidden state for the new sequence batch
            Flux.reset!(model)
            
            loss, grads = Flux.withgradient(model) do m
                y_hat = m(x_dev)
                logitcrossentropy(y_hat, y_dev)
            end
            
            Flux.update!(opt_state, model, grads[1])
            loss_total += loss
        end
        
        if epoch % 5 == 0 || epoch == 1
            println("Epoch $epoch: Avg Loss = $(round(loss_total/length(loader), digits=4))")
        end
    end

    # Save to Disk
    mkpath(dirname(model_path))
    jldsave(model_path; state=Flux.state(cpu(model)))
    println("Model saved to $model_path")
    
    return model, loader
end

# --- 6. Main Execution ---

function main()
    filepath = "data.csv"
    epochs = 30
    model_filepath = "models/model_gru.jld2"

    if isfile(filepath)
        # Train
        model, loader = train_model(filepath, model_filepath, epochs)
        
        # Evaluate
        final_acc, class_stats = evaluate_performance(model, loader)

        println("\n" * "="^30)
        println("GRU EVALUATION SUMMARY")
        println("="^30)
        println("Overall Accuracy: $(round(final_acc*100, digits=2))%")

        for i in 1:3
            println("\nClass $i:")
            println("  Precision: $(round(class_stats[i].precision, digits=3))")
            println("  Recall:    $(round(class_stats[i].recall, digits=3))")
            println("  F1-Score:  $(round(class_stats[i].f1, digits=3))")
        end
    else
        println("Error: CSV file '$filepath' not found.")
    end
end

main()


