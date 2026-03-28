using Flux, Optimisers, CSV, DataFrames, Statistics
using Images, FileIO, ColorTypes, CUDA, JLD2
using MLUtils 
using Flux: onecold, softmax, logitcrossentropy

# --- 1. GPU Check ---
# Using 'const' ensures the device is accessible globally within the script
const device = CUDA.functional() ? gpu : cpu
println("Using device: ", device)

# --- 2. Data Processing ---

function load_and_preprocess(path)
    try
        img = load(path)
        img = Gray.(img)
        img = imresize(img, (32, 32))
        # Ensure it returns Float32 for Flux compatibility
        return reshape(Float32.(img), 32, 32, 1, 1) 
    catch e
        @warn "Could not load $path: $e"
        return zeros(Float32, 32, 32, 1, 1)
    end
end

function get_datasets(csv_path; batchsize=32, split_ratio=0.8)
    df = CSV.read(csv_path, DataFrame)
    if isempty(df)
        error("CSV file at $csv_path is empty!")
    end

    println("Loading and preprocessing images...")
    imgs = [load_and_preprocess(row.image) for row in eachrow(df)]
    # Concatenate along the 4th dimension (Batch)
    X = reduce((a, b) -> cat(a, b, dims=4), imgs)

    # Labels 0, 1, 2 -> Map to 1, 2, 3 for OneHot
    Y = Flux.onehotbatch(df.label .+ 1, 1:3)

    # Split into Train and Validation
    (X_train, Y_train), (X_val, Y_val) = splitobs((X, Y); at=split_ratio)

    train_loader = DataLoader((X_train, Y_train), batchsize=batchsize, shuffle=true)
    val_loader = DataLoader((X_val, Y_val), batchsize=batchsize, shuffle=false)

    return train_loader, val_loader
end

# --- 3. Model Components (ViT) ---

struct PatchEmbedding
    projection::Conv
    class_token
    pos_embedding
end
@functor PatchEmbedding

function PatchEmbedding(img_size=32, patch_size=4, in_ch=1, embed_dim=128)
    n_patches = (img_size ÷ patch_size)^2
    projection = Conv((patch_size, patch_size), in_ch => embed_dim, stride=patch_size)
    class_token = randn(Float32, embed_dim, 1, 1)
    pos_embedding = randn(Float32, embed_dim, n_patches + 1, 1)
    return PatchEmbedding(projection, class_token, pos_embedding)
end

function (m::PatchEmbedding)(x)
    x = m.projection(x)
    B = size(x, 4)
    x = reshape(x, size(x, 3), :, B) # (Embed, Patches, Batch)
    cl_tok = repeat(m.class_token, 1, 1, B)
    x = cat(cl_tok, x, dims=2)
    return x .+ m.pos_embedding
end

struct MLPBlock
    norm1; attn; norm2; mlp
end
@functor MLPBlock

function MLPBlock(embed=128, heads=4, mlp_dim=256)
    return MLPBlock(
        Flux.LayerNorm(embed),
        Flux.MultiHeadAttention(embed; nheads = heads),
        Flux.LayerNorm(embed),
        Chain(Dense(embed, mlp_dim, Flux.gelu), Dense(mlp_dim, embed))
    )
end

function (m::MLPBlock)(x)
    h = m.norm1(x)
    attn_out, _ = m.attn(h, h, h)
    x = x .+ attn_out
    x = x .+ m.mlp(m.norm2(x))
    return x
end

function create_model()
    vit = Chain(
        PatchEmbedding(32, 4, 1, 128),
        MLPBlock(128, 4, 256),
        MLPBlock(128, 4, 256),
        MLPBlock(128, 4, 256),
        MLPBlock(128, 4, 256),
        Flux.LayerNorm(128),
        x -> x[:, 1, :] # Extract Class Token
    )
    return Chain(vit, Dense(128, 64, relu), Dense(64, 3))
end

# --- 4. Evaluation Logic ---

function evaluate_performance(model, loader)
    Flux.testmode!(model)
    all_preds = Int[]
    all_labels = Int[]

    for (x, y) in loader
        x_dev = x |> device
        logits = model(x_dev) |> cpu
        preds = onecold(logits, 1:3)
        actuals = onecold(y, 1:3)

        append!(all_preds, preds)
        append!(all_labels, actuals)
    end

    acc = mean(all_preds .== all_labels)

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

# --- 5. Training Logic ---

function train_model(csv_path, model_path, epochs)
    train_loader, val_loader = get_datasets(csv_path)
    model = create_model() |> device
    opt_state = Flux.setup(Flux.Adam(1f-4), model)

    println("Training starting for $epochs epochs...")

    for epoch in 1:epochs
        Flux.trainmode!(model)
        loss_total = 0.0f0

        for (x, y) in train_loader
            x_dev, y_dev = x |> device, y |> device
            loss, grads = Flux.withgradient(model) do m
                y_hat = m(x_dev)
                logitcrossentropy(y_hat, y_dev)
            end
            Flux.update!(opt_state, model, grads[1])
            loss_total += loss
        end

        if epoch % 5 == 0 || epoch == 1
            val_acc, _ = evaluate_performance(model, val_loader)
            println("Epoch $epoch | Loss: $(round(loss_total/length(train_loader), digits=4)) | Val Acc: $(round(val_acc*100, digits=2))%")
        end
    end

    mkpath(dirname(model_path))
    jldsave(model_path; state=Flux.state(cpu(model)))
    println("Model saved to $model_path")

    return model, val_loader
end

# --- 6. Main ---

function main()
    filepath = "data.csv"
    epochs = 50 
    model_filepath = "models/modelJulia.jld2"

    if isfile(filepath)
        model, val_loader = train_model(filepath, model_filepath, epochs)

        final_acc, class_stats = evaluate_performance(model, val_loader)

        println("\n" * "="^30)
        println("FINAL PERFORMANCE REPORT")
        println("="^30)
        println("Overall Accuracy: $(round(final_acc*100, digits=2))%")

        for i in 1:3
            println("\nClass $i:")
            println("  Precision: $(round(class_stats[i].precision, digits=3))")
            println("  Recall:    $(round(class_stats[i].recall, digits=3))")
            println("  F1-Score:  $(round(class_stats[i].f1, digits=3))")
        end
    else
        println("Error: $filepath not found.")
    end
end

main()

