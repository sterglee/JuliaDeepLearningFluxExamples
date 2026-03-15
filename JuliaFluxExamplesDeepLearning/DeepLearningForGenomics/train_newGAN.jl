using Flux
using Flux: DataLoader
using NPZ
using Statistics
using Random

# -----------------------------
# 1. LOAD DATA
# -----------------------------
X_raw = npzread("data/X_train.npy")[1:10000, :, :]
y_raw = npzread("data/y_train.npy")[1:10000, :]

X_train = Float32.(permutedims(X_raw, (2,3,1))) # (Time, Features, Batch)
y_train = Float32.(permutedims(y_raw))          # (Classes, Batch)

seq_len = size(X_train, 1)
feat_dim = size(X_train, 2)
batch_size = 100
latent_dim = 128
epochs = 5

# -----------------------------
# 2. GENERATOR
# -----------------------------
generator = Chain(
    Dense(latent_dim, 512, relu),
    Dense(512, 1024, relu),
    Dense(1024, seq_len * feat_dim),
    x -> reshape(x, (seq_len, feat_dim, :)),
    x -> tanh.(x)  # broadcast tanh for 3D arrays
)

# -----------------------------
# 3. DISCRIMINATOR
# -----------------------------
discriminator = Chain(
    x -> reshape(x, (seq_len * feat_dim, :)),
    Dense(seq_len * feat_dim, 512, relu),
    Dense(512, 256, relu),
    Dense(256, 1),
    σ
)

# -----------------------------
# 4. OPTIMIZERS
# -----------------------------
opt_gen = Flux.setup(Flux.Adam(1e-4), generator)
opt_disc = Flux.setup(Flux.Adam(1e-4), discriminator)

# -----------------------------
# 5. DATA LOADER
# -----------------------------
train_loader = DataLoader((X_train,), batchsize=batch_size, shuffle=true)

# -----------------------------
# 6. TRAINING LOOP
# -----------------------------
for epoch in 1:epochs
    println("Epoch ", epoch)
    for (x_batch,) in train_loader
        x_real = Float32.(x_batch)

        # Train discriminator
        z = randn(Float32, latent_dim, size(x_real,3))
        x_fake = generator(z)
        y_real = ones(Float32, 1, size(x_real,3))
        y_fake = zeros(Float32, 1, size(x_real,3))

        grads_disc = Flux.gradient(discriminator) do d
            loss_real = Flux.Losses.binarycrossentropy(d(x_real), y_real)
            loss_fake = Flux.Losses.binarycrossentropy(d(x_fake), y_fake)
            loss_real + loss_fake
        end
        Flux.Optimise.update!(opt_disc, discriminator, grads_disc)

        # Train generator
        z = randn(Float32, latent_dim, size(x_real,3))
        grads_gen = Flux.gradient(generator) do g
            x_fake = g(z)
            Flux.Losses.binarycrossentropy(discriminator(x_fake), y_real)
        end
        Flux.Optimise.update!(opt_gen, generator, grads_gen)
    end

    # Monitor
    z = randn(Float32, latent_dim, 10)
    sample = generator(z)
    println("Epoch $epoch complete. Sample shape: ", size(sample))
end

println("GAN training complete")



