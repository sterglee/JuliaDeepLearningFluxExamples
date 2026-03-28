using Flux
using Flux: @functor, DataLoader, logitcrossentropy, onecold, reset!, trainmode!, testmode!
using Optimisers
using CSV
using DataFrames
using Statistics
using Images, FileIO, ColorTypes
using JLD2
using MLUtils # For efficient batch stacking

# --- 1. Device Setup ---
# Force CPU usage
const device = cpu
println("Device set to: CPU")

# --- 2. Data Processing ---

function load_and_preprocess(path)
    try
        if !isfile(path)
            return zeros(Float32, 32, 32, 1)
        end
        img = load(path)
        img = Gray.(img)
        img = imresize(img, (32, 32))
        return reshape(Float32.(img), 32, 32, 1) # (W, H, C)
        catch e
        @warn "Error loading $path: $e"
        return zeros(Float32, 32, 32, 1)
    end
end

function get_dataset(csv_path; batchsize=32, shuffle=true)
    df = CSV.read(csv_path, DataFrame)
    if isempty(df)
        error("The CSV file at $csv_path is empty.")
    end

    println("Loading $(nrow(df)) images...")
    # Load images into a vector of arrays
    img_list = [load_and_preprocess(row.image) for row in eachrow(df)]

        # MLUtils.stack is faster and safer than reduce(cat...)
        # Result: (32, 32, 1, BatchSize)
        X = MLUtils.stack(img_list)

        # One-hot encode labels (assuming 0, 1, 2 in CSV)
        Y = Flux.onehotbatch(df.label .+ 1, 1:3)

        return DataLoader((X, Y), batchsize=batchsize, shuffle=shuffle)
    end

    # --- 3. GRU Model Architecture ---

    struct ImageToSequence
        projection::Conv
    end
    @functor ImageToSequence

    function ImageToSequence(patch_size=4, in_ch=1, embed_dim=128)
        # Patch projection: extracts features from local patches
        projection = Conv((patch_size, patch_size), in_ch => embed_dim, stride=patch_size)
        return ImageToSequence(projection)
    end

    function (m::ImageToSequence)(x)
        # x: (32, 32, 1, Batch) -> (8, 8, 128, Batch)
        x = m.projection(x)
        # Reshape to (Features, SequenceLength, Batch) for the GRU
        # SeqLength = 64 (8x8 grid of patches)
        return reshape(x, size(x, 3), :, size(x, 4))
    end

    function create_model()
        return Chain(
            ImageToSequence(4, 1, 128),
            Flux.GRU(128 => 128),
            x -> x[:, end, :],    # Take the final hidden state of the sequence
            Dense(128, 64, relu),
            Dense(64, 3)
            )
    end

    # --- 4. Evaluation Logic ---

    function evaluate_performance(model, loader)
        testmode!(model)
        all_preds = Int[]
        all_labels = Int[]

        for (x, y) in loader
            # Ensure hidden state is cleared before each batch
            reset!(model)

            logits = model(x)
            append!(all_preds, onecold(logits, 1:3))
            append!(all_labels, onecold(y, 1:3))
        end

        acc = mean(all_preds .== all_labels)

        # Calculate Per-Class Metrics
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
        loader = get_dataset(csv_path)
        model = create_model()

        # Adam Optimizer
        opt_state = Flux.setup(Flux.Adam(0.001f0), model)

        println("Starting training on CPU for $epochs epochs...")

            for epoch in 1:epochs
                loss_total = 0.0f0
                trainmode!(model)

                for (x, y) in loader
                    reset!(model) # Critical for GRU

                    loss, grads = Flux.withgradient(model) do m
                        y_hat = m(x)
                        logitcrossentropy(y_hat, y)
                    end
                    Flux.update!(opt_state, model, grads[1])
                    loss_total += loss
                end

                # Log progress every epoch
                acc, _ = evaluate_performance(model, loader)
                println("Epoch $epoch | Loss: $(round(loss_total/length(loader), digits=4)) | Acc: $(round(acc*100, digits=2))%")
            end

            # Save weights
            mkpath(dirname(model_path))
            jldsave(model_path; state=Flux.state(model))
            println("Model weights saved to $model_path")

            return model, loader
        end

        # --- 6. Main Execution ---

        function main()
            # Configuration
            csv_filename = "data.csv"
            save_path = "models/gru_cpu_model.jld2"
            num_epochs = 1000

            if isfile(csv_filename)
                model, loader = train_model(csv_filename, save_path, num_epochs)

                # Final Detailed Report
                acc, stats = evaluate_performance(model, loader)

                println("\n" * "="^30)
                println("FINAL PERFORMANCE REPORT (CPU)")
                println("="^30)
                println("Accuracy: $(round(acc*100, digits=2))%")
                for i in 1:3
                    println("\nClass $i:")
                    println("  F1-Score: $(round(stats[i].f1, digits=3))")
                    println("  Precision: $(round(stats[i].precision, digits=3))")
                end
            else
                println("Error: CSV file '$csv_filename' not found.")
            end
        end

        main()

