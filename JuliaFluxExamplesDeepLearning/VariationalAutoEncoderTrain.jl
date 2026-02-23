using Flux, Optimisers, ProgressMeter, MLDatasets, Statistics

# 1. Define the VAE Structure
struct VAE
    encoder
    μ_layer
    logσ_layer
    decoder
end

Flux.@layer VAE

# The Reparameterization Trick: z = μ + ϵ * σ
function sample_z(μ, logσ)
    ϵ = randn(Float32, size(μ))
    return μ + ϵ .* exp.(logσ)
end

function (m::VAE)(x)
    h = m.encoder(x)
    μ = m.μ_layer(h)
    logσ = m.logσ_layer(h)
    z = sample_z(μ, logσ)
    return m.decoder(z), μ, logσ
end

# 2. Build the Model
latent_dim = 2
encoder_backbone = Chain(Dense(784 => 256, relu), Dense(256 => 128, relu))

model = VAE(
    encoder_backbone,
    Dense(128 => latent_dim), # μ
    Dense(128 => latent_dim), # logσ
    Chain(Dense(latent_dim => 128, relu), Dense(128 => 256, relu), Dense(256 => 784, sigmoid))
    )

# 3. Data Preparation
train_data = MNIST(split=:train)
X_train = reshape(Float32.(train_data.features), 28*28, :)
loader = Flux.DataLoader(X_train, batchsize=128, shuffle=true)

# 4. Modern Optimiser Setup
opt_state = Optimisers.setup(Optimisers.Adam(1e-3), model)

# 5. VAE Loss Function (Reconstruction + KL Divergence)
function loss_fn(m, x)
    x̂, μ, logσ = m(x)

    # Binary Cross Entropy for reconstruction
    recon_loss = Flux.binarycrossentropy(x̂, x, agg=sum)

    # KL Divergence: -0.5 * sum(1 + 2logσ - μ^2 - exp(2logσ))
    kl_loss = -0.5f0 * sum(1 .+ 2f0 .* logσ .- μ.^2 .- exp.(2f0 .* logσ))

    return (recon_loss + kl_loss) / size(x, 2)
end

# 6. Training Loop
epochs = 20
for epoch in 1:epochs
    @showprogress "Epoch $epoch: " for x in loader
        # Use mutating train! call
        # Since VAE loss only needs 'x', we pass [x] as the data batch
        Flux.train!(loss_fn, model, [x], opt_state)
    end
    println("Epoch $epoch Loss: $(loss_fn(model, X_train[:, 1:1000]))")
end

