using Flux
using Flux: @functor, DataLoader, logitcrossentropy
using Optimisers
using CSV
using DataFrames
using Statistics
using Images, FileIO, ColorTypes
using JLD2

# --- 1. Data Processing ---

function load_and_preprocess(path)
    img = load(path)              # Load image
    img = Gray.(img)              # Convert to grayscale
    img = imresize(img, (32, 32)) # Resize to match model input
    # Flux WHCN: (Width, Height, Channels, Batch)
    return reshape(Float32.(img), 32, 32, 1)
end

function get_dataset(csv_path; batchsize=32, is_train=true)
    df = CSV.read(csv_path, DataFrame)
    imgs = [load_and_preprocess(row.image) for row in eachrow(df)]
        X = cat(imgs..., dims=4) # Combine into 4D tensor (32, 32, 1, N)

        if is_train
            # One-hot encoding for 3 classes (assuming labels are 0, 1, 2)
            Y = Flux.onehotbatch(df.label .+ 1, 1:3)
            return DataLoader((X, Y), batchsize=batchsize, shuffle=is_train), df
        end
        return DataLoader(X, batchsize=batchsize, shuffle=false), df
    end

    # --- 2. Model Components ---

    struct PatchEmbedding
        projection::Conv
        class_token
        pos_embedding
    end
    @functor PatchEmbedding

    function PatchEmbedding(img_size=32, patch_size=4, in_ch=1, embed_dim=128)
        n_patches = (img_size ÷ patch_size)^2
        # projection turns (patch, patch) regions into embed_dim channels
        projection = Conv((patch_size, patch_size), in_ch => embed_dim, stride=patch_size)
        class_token = randn(Float32, embed_dim, 1, 1)
        pos_embedding = randn(Float32, embed_dim, n_patches + 1, 1)

        return PatchEmbedding(projection, class_token, pos_embedding)
    end

    function (m::PatchEmbedding)(x)
        # x: (32, 32, 1, Batch)
        x = m.projection(x) # (8, 8, 128, Batch)

        # Flatten spatial dims: (128, 64, Batch)
        B = size(x, 4)
        x = reshape(x, size(x, 3), :, B)

        # Add class token: (128, 1, Batch)
        cl_tok = repeat(m.class_token, 1, 1, B)
        x = cat(cl_tok, x, dims=2) # (128, 65, Batch)

        return x .+ m.pos_embedding
    end

    struct MLPBlock
        norm1
        attn
        norm2
        mlp
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
        # x shape: (embed_dim, seq_len, batch)

        # Attention path with residual connection
        # MultiHeadAttention returns (output, weights)
        h = m.norm1(x)
        attn_out, _ = m.attn(h, h, h)
        x = x .+ attn_out

        # MLP path with residual connection
        x = x .+ m.mlp(m.norm2(x))
        return x
    end

    function create_model()
        # ViT Backbone
        vit = Chain(
            PatchEmbedding(32, 4, 1, 128),
            MLPBlock(128, 4, 256),
            MLPBlock(128, 4, 256),
            MLPBlock(128, 4, 256),
            MLPBlock(128, 4, 256),
            Flux.LayerNorm(128),
            x -> x[:, 1, :] # Extract Class Token: (128, Batch)
            )
        # Classification Head
        return Chain(vit, Dense(128, 64, relu), Dense(64, 3))
    end

    # --- 3. Training Function ---

    function train_model(csv_path, model_path, epochs)
        loader, _ = get_dataset(csv_path, is_train=true)
        model = create_model()
        # Using Flux.setup and Optimisers.jl (Modern Flux style)
        opt_state = Flux.setup(Flux.Adam(0.001f0), model)

        println("Training starting for $epochs epochs...")
            for epoch in 1:epochs
                loss_total = 0.0f0
                for (x, y) in loader
                    val, grads = Flux.withgradient(model) do m
                        y_hat = m(x)
                        logitcrossentropy(y_hat, y)
                    end
                    Flux.update!(opt_state, model, grads[1])
                    loss_total += val
                end
                println("Epoch $epoch: Avg Loss = $(loss_total/length(loader))")
            end

            # Save the model state
            mkpath(dirname(model_path))
            jldsave(model_path; state=Flux.state(model))
            println("Model saved at $model_path")
        end

        # --- 4. Main ---

        function main()
            filepath = "data.csv"
            epochs = 100
            model_filepath = "models/modelJulia.jld2"

            if isfile(filepath)
                train_model(filepath, model_filepath, epochs)
            else
                println("Error: $filepath not found. Please ensure the CSV exists.")
            end
        end

        main()

