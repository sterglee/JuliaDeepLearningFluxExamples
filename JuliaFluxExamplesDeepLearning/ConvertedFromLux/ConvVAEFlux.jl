# Lux time/epoch: 1.2s
# Flux time/epoch: 0.9s

using Flux,
    Random,
    Statistics,
    MLUtils,
    Images,
    Printf,
    Optimisers

# Ensure reproducibility and set device
Random.seed!(123)
device = cpu

## --- 1. Synthetic Data Generation ---

function generate_synthetic_data(num_samples=1000)
    # Create 28x28 grayscale images of random "blobs"
    X = zeros(Float32, 28, 28, 1, num_samples)
    for i in 1:num_samples
        cx, cy = rand(6:22, 2)
        r = rand(3:5)
        for x in 1:28, y in 1:28
            if (x - cx)^2 + (y - cy)^2 < r^2
                X[x, y, 1, i] = 1.0f0
            end
        end
    end
    return X .+ 0.05f0 .* randn(Float32, size(X))
end

## --- 2. Model Definition ---

# We define the CVAE as a mutable struct to hold the sub-networks
mutable struct CVAE
    encoder_backbone
    proj_mu
    proj_log_var
    decoder
end

# Make the struct "Flux-aware" so Optimisers can see the parameters
Flux.@functor CVAE

function build_cvae(latent_dims, max_filters)
    # Encoder: 28x28 -> 14x14 -> 7x7 -> 4x4
    encoder_backbone = Chain(
        Conv((3, 3), 1 => max_filters ÷ 4, leakyrelu; stride=2, pad=1),
        BatchNorm(max_filters ÷ 4),
        Conv((3, 3), max_filters ÷ 4 => max_filters ÷ 2, leakyrelu; stride=2, pad=1),
        BatchNorm(max_filters ÷ 2),
        Conv((3, 3), max_filters ÷ 2 => max_filters, leakyrelu; stride=2, pad=1),
        BatchNorm(max_filters),
        Flux.flatten
        )

    # 4*4*max_filters = 16 * 32 = 512
    flattened_dim = 16 * max_filters

    proj_mu = Dense(flattened_dim, latent_dims)
    proj_log_var = Dense(flattened_dim, latent_dims)

    # Decoder: 4x4 -> 8x8 -> 16x16 -> 32x32 -> Crop to 28x28
    decoder = Chain(
        Dense(latent_dims, flattened_dim, leakyrelu),
        x -> reshape(x, 4, 4, max_filters, :),
        Upsample(2),
        Conv((3, 3), max_filters => max_filters ÷ 2, leakyrelu; stride=1, pad=1),
        Upsample(2),
        Conv((3, 3), max_filters ÷ 2 => max_filters ÷ 4, leakyrelu; stride=1, pad=1),
        Upsample(2),
        # Final conv with pad=0 reduces 32x32 to 28x28
        Conv((5, 5), max_filters ÷ 4 => 1, sigmoid; stride=1, pad=0)
        )

    return CVAE(encoder_backbone, proj_mu, proj_log_var, decoder)
end

function (model::CVAE)(x)
    # Forward pass: Encode
    h = model.encoder_backbone(x)
    μ = model.proj_mu(h)
    logσ² = clamp.(model.proj_log_var(h), -20.0f0, 10.0f0)

    # Reparameterization trick
    σ = exp.(logσ² .* 0.5f0)
    z = μ .+ σ .* randn(Float32, size(μ))

    # Decode
    return model.decoder(z), μ, logσ²
end

## --- 3. Training Logic ---

function loss_function(model, x)
    x_hat, μ, logσ² = model(x)
    # Reconstruction loss (Mean Squared Error)
    recon_loss = Flux.Losses.mse(x_hat, x, agg=sum)
    # KL Divergence: pushes latent distribution toward N(0,1)
    kl_loss = -0.5f0 * sum(1 .+ logσ² .- μ.^2 .- exp.(logσ²))
    # Average loss over the batch
    return (recon_loss + kl_loss) / size(x, 4)
end

function train()
    # Configuration
    batchsize = 32
    latent_dims = 4
    epochs = 15
    lr = 1e-3

    # Data and Model
    X = generate_synthetic_data(1000)
    loader = DataLoader(X, batchsize=batchsize, shuffle=true)
    model = build_cvae(latent_dims, 32)

    # Optimizer setup (Optimisers.jl standard)
    opt_state = Optimisers.setup(Optimisers.Adam(lr), model)

    println("Starting Training Flux CVAE...")
    for epoch in 1:epochs
        total_loss = 0.0f0
        start_time = time_ns()

        for x in loader
            # Compute gradients
            val, grads = Flux.withgradient(model) do m
                loss_function(m, x)
            end
            # Apply updates
            opt_state, model = Optimisers.update!(opt_state, model, grads[1])
            total_loss += val
        end

        elapsed = (time_ns() - start_time) / 1e9
        @printf "Epoch %02d | Loss: %.4f | Time: %.2fs | Throughput: %.1f im/s\n" epoch (total_loss/length(loader)) elapsed (length(X)/elapsed)
    end

    return model, X
end

# Execute training
trained_model, raw_data = train()

## --- 4. Result Visualization ---

function visualize(model, data)
    # Get 8 samples and reconstruct them
    sample = data[:, :, :, 1:8]
    reconstructed, _, _ = model(sample)

    # Grid: Top row original, Bottom row reconstruction
    orig = mosaicview(Gray.(sample[:, :, 1, :]), nrow=1)
    recon = mosaicview(Gray.(reconstructed[:, :, 1, :]), nrow=1)
    return vcat(orig, recon)
end



comparison_img = visualize(trained_model, raw_data)
# To view in VSCode/Jupyter, simply call comparison_img
# To save: save("flux_reconstruction.png", comparison_img)


