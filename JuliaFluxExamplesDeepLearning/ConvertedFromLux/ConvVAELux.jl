using Lux, Random, Statistics, MLUtils, Images, Printf, Optimisers, Zygote

# Force CPU
dev = cpu_device()

## --- 1. Synthetic Data Generation ---
function generate_synthetic_data(num_samples=1000)
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

# Encoder: 28x28 -> 14x14 -> 7x7 -> 4x4
function cvae_encoder(num_latent_dims::Int, max_num_filters::Int)
    backbone = Chain(
        Conv((3, 3), 1 => max_num_filters ÷ 4, leakyrelu; stride=2, pad=1),
        BatchNorm(max_num_filters ÷ 4),
        Conv((3, 3), max_num_filters ÷ 4 => max_num_filters ÷ 2, leakyrelu; stride=2, pad=1),
        BatchNorm(max_num_filters ÷ 2),
        Conv((3, 3), max_num_filters ÷ 2 => max_num_filters, leakyrelu; stride=2, pad=1),
        BatchNorm(max_num_filters),
        FlattenLayer(),
        )

    # 4*4*max_filters = 16*32 = 512
    flattened_dim = 16 * max_num_filters

    return @compact(;
                    embed = backbone,
                    proj_mu = Dense(flattened_dim, num_latent_dims),
                    proj_log_var = Dense(flattened_dim, num_latent_dims),
                    ) do x
        y = embed(x)
        μ = proj_mu(y)
        logσ² = clamp.(proj_log_var(y), -20.0f0, 10.0f0)
        σ = exp.(logσ² .* 0.5f0)
        z = μ .+ σ .* randn_like(μ)
        @return z, μ, logσ²
                    end
end

# Decoder: 4x4 -> 8x8 -> 16x16 -> 32x32 -> Crop to 28x28
function cvae_decoder(num_latent_dims::Int, max_num_filters::Int)
    reshape_size = (4, 4, max_num_filters)
    flattened_dim = prod(reshape_size)

    return Chain(
        Dense(num_latent_dims, flattened_dim, leakyrelu),
        x -> reshape(x, reshape_size..., :),
        Upsample(2), # 4x4 -> 8x8
        Conv((3, 3), max_num_filters => max_num_filters ÷ 2, leakyrelu; stride=1, pad=1),
        Upsample(2), # 8x8 -> 16x16
        Conv((3, 3), max_num_filters ÷ 2 => max_num_filters ÷ 4, leakyrelu; stride=1, pad=1),
        Upsample(2), # 16x16 -> 32x32
        # Use stride=1 and no padding to shrink 32x32 down to 28x28
        Conv((5, 5), max_num_filters ÷ 4 => 1, sigmoid; stride=1, pad=0)
        )
end



struct CVAE <: Lux.AbstractLuxContainerLayer{(:encoder, :decoder)}
    encoder::Lux.AbstractLuxLayer
    decoder::Lux.AbstractLuxLayer
end

function (model::CVAE)(x, ps, st)
    (z, μ, logσ²), st_enc = model.encoder(x, ps.encoder, st.encoder)
    x_rec, st_dec = model.decoder(z, ps.decoder, st.decoder)
    return (x_rec, μ, logσ²), (; encoder=st_enc, decoder=st_dec)
end

## --- 3. Training ---

function loss_function(model, ps, st, x)
    (x_hat, μ, logσ²), st_new = model(x, ps, st)
    recon_loss = sum((x_hat .- x) .^ 2)
    kl_loss = -0.5f0 * sum(1 .+ logσ² .- μ .^ 2 .- exp.(logσ²))
    return (recon_loss + kl_loss) / size(x, 4), st_new, (;)
end

function train()
    rng = Random.default_rng()
    X = generate_synthetic_data(1000)
    loader = DataLoader(X, batchsize=32, shuffle=true)

    model = CVAE(cvae_encoder(4, 32), cvae_decoder(4, 32))
    ps, st = Lux.setup(rng, model)
    tstate = Lux.Training.TrainState(model, ps, st, Optimisers.Adam(1f-3))

    println("Training Lux CVAE...")
    for epoch in 1:10
        total_loss = 0.0f0
        start_t = time_ns()
        for x in loader
            _, loss, _, tstate = Lux.Training.single_train_step!(AutoZygote(), loss_function, x, tstate)
            total_loss += loss
        end
        elapsed = (time_ns() - start_t) / 1e9
        @printf "Epoch %02d | Loss: %.4f | Time: %.2fs\n" epoch (total_loss/length(loader)) elapsed
    end
    return tstate, X
end

tstate, raw_data = train()

# Visual Check
sample = raw_data[:, :, :, 1:8]
(res, _, _), _ = tstate.model(sample, tstate.parameters, tstate.states)
mosaicview(Gray.(vcat(sample[:,:,1,:], res[:,:,1,:])), nrow=2)


