using Flux
using Flux: @functor, DataLoader, logitcrossentropy, cpu, gpu
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
    img = load(path)
    img = Gray.(img)
    img = imresize(img, (32, 32))
    return reshape(Float32.(img), 32, 32, 1)
end

function get_dataset(csv_path; batchsize=32, is_train=true)
    df = CSV.read(csv_path, DataFrame)
    imgs = [load_and_preprocess(row.image) for row in eachrow(df)]
    X = cat(imgs..., dims=4)

        if is_train
            Y = Flux.onehotbatch(df.label .+ 1, 1:3)
            # Note: Do NOT move to GPU here, move in the training loop
            return DataLoader((X, Y), batchsize=batchsize, shuffle=is_train), df
        end
        return DataLoader(X, batchsize=batchsize, shuffle=false), df
    end

    # --- 3. Model Components ---

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
        # Input x shape: (32, 32, 1, Batch)
        x = m.projection(x) # (8, 8, 128, Batch)

        B = size(x, 4)
        x = reshape(x, size(x, 3), :, B) # (128, 64, Batch)

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
            x -> x[:, 1, :] # Extract Class Token: (128, Batch)
            )
        return Chain(vit, Dense(128, 64, relu), Dense(64, 3))
    end

    # --- 4. Training Function ---

    function train_model(csv_path, model_path, epochs)
        loader, _ = get_dataset(csv_path, is_train=true)

        # 1. Move model to device (GPU)
        model = create_model() |> device

        # Setup optimizer
        opt_state = Flux.setup(Flux.Adam(0.001f0), model)

        println("Training starting on $(device) for $epochs epochs...")
            for epoch in 1:epochs
                loss_total = 0.0f0
                for (x, y) in loader
                    # 2. Move data batch to device (GPU)
                    x_dev, y_dev = x |> device, y |> device

                    val, grads = Flux.withgradient(model) do m
                        y_hat = m(x_dev)
                        logitcrossentropy(y_hat, y_dev)
                    end
                    Flux.update!(opt_state, model, grads[1])
                    loss_total += cpu(val) # Move loss back to CPU for logging
                end
                println("Epoch $epoch: Avg Loss = $(loss_total/length(loader))")
            end

            # 3. Move model to CPU before saving
            mkpath(dirname(model_path))
            jldsave(model_path; state=Flux.state(cpu(model)))
            println("Model saved at $model_path")
        end

        # --- 5. Main ---

        function main()
            filepath = "data.csv"
            epochs = 300
            model_filepath = "models/modelJulia.jld2"

            if isfile(filepath)
                train_model(filepath, model_filepath, epochs)
            else
                println("Error: $filepath not found.")
            end
        end

        main()
