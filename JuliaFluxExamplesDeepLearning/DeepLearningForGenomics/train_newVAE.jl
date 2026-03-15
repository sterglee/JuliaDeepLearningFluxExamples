# train loss  0.1869
using Flux
using Flux: DataLoader
using NPZ
using Statistics
using Random

# -----------------------------
# 1. LOAD DATA
# -----------------------------
X_raw = npzread("data/X_train.npy")[1:10000, :, :]
X_train = Float32.(permutedims(X_raw, (2, 3, 1)))

seq_len, feat_dim, n_samples = size(X_train)
input_dim = seq_len * feat_dim
latent_dim = 64
batch_size = 100
epochs = 5

# -----------------------------
# 2. MODEL DEFINITION
# -----------------------------
# We define a custom struct to hold our VAE components.
# This is much cleaner and avoids the NamedTuple ArgumentError.
struct VAE
    encoder::Chain
    μ_layer::Dense
    logσ_layer::Dense
    decoder::Chain
end

# Make the struct "trainable" so Flux can see its parameters
Flux.@functor VAE

# Initialize the model
model = VAE(
    Chain(Flux.flatten, Dense(input_dim, 512, relu), Dense(512, 256, relu)),
    Dense(256, latent_dim),
    Dense(256, latent_dim),
    Chain(
        Dense(latent_dim, 256, relu),
        Dense(256, 512, relu),
        Dense(512, input_dim),
        x -> reshape(x, (seq_len, feat_dim, :)),
        σ
    )
)

# -----------------------------
# 3. VAE LOGIC
# -----------------------------
function reparameterize(μ, logσ)
    ϵ = randn(Float32, size(μ))
    return μ .+ exp.(0.5f0 .* logσ) .* ϵ
end

function vae_loss(m::VAE, x)
    # Forward Pass
    h = m.encoder(x)
    μ = m.μ_layer(h)
    logσ = m.logσ_layer(h)
    
    z = reparameterize(μ, logσ)
    x_hat = m.decoder(z)
    
    # 1. Reconstruction Loss (MSE)
    rec_loss = Flux.Losses.mse(x_hat, x)
    
    # 2. KL Divergence
    # We use a small epsilon for logσ stability if needed, 
    # but the standard formula usually works fine.
    kl_loss = -0.5f0 * mean(sum(1 .+ logσ .- μ.^2 .- exp.(logσ), dims=1))
    
    return rec_loss + kl_loss
end

# -----------------------------
# 4. TRAINING SETUP
# -----------------------------
opt_state = Flux.setup(Flux.Adam(1e-3), model)
train_loader = DataLoader((X_train,), batchsize=batch_size, shuffle=true)

# -----------------------------
# 5. TRAINING LOOP
# -----------------------------
println("Starting VAE Training...")

for epoch in 1:epochs
    for (x_batch,) in train_loader
        grads = Flux.gradient(model) do m
            vae_loss(m, x_batch)
        end
        Flux.update!(opt_state, model, grads[1])
    end
    
    l = vae_loss(model, X_train[:, :, 1:200])
    println("Epoch $epoch - Loss = $(round(l, digits=4))")
end

# -----------------------------
# 6. GENERATION
# -----------------------------
z_new = randn(Float32, latent_dim, 10)
samples = model.decoder(z_new)
println("Generated samples shape: ", size(samples))


