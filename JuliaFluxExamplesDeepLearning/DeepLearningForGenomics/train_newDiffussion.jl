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

println("X_train shape = ", size(X_train))
println("y_train shape = ", size(y_train))

# -----------------------------
# 2. DIFFUSION NOISE SCHEDULE
# -----------------------------
# Simple linear schedule
T = 100
β = range(1e-4, 0.02, length=T) |> collect
α = 1 .- β
α_hat = cumprod(α) # cumulative product

# -----------------------------
# 3. DENOISING NETWORK (1D CNN)
# -----------------------------
# Predict noise added at each timestep
denoise_model = Chain(
    Conv((3,), 4=>128, relu, pad=1),
    Conv((3,), 128=>128, relu, pad=1),
    Conv((3,), 128=>4, pad=1) # predict noise of same shape as input
)

opt_state = Flux.setup(Flux.Adam(1e-3), denoise_model)

# -----------------------------
# 4. FORWARD DIFFUSION (q(x_t | x_0))
# -----------------------------
function add_noise(x0, t)
    ϵ = randn(Float32, size(x0))
    xt = sqrt(α_hat[t]) .* x0 .+ sqrt(1 - α_hat[t]) .* ϵ
    return xt, ϵ
end

# -----------------------------
# 5. TRAINING LOOP
# -----------------------------
epochs = 5
train_loader = DataLoader((X_train, y_train), batchsize=100, shuffle=true)

for epoch in 1:epochs
    println("Epoch ", epoch)
    for (x, _) in train_loader
        x = Float32.(x)

        # Sample random diffusion timestep for each batch
        t = rand(1:T)

        # Add noise
        xt, ϵ_true = add_noise(x, t)

        # Forward pass: predict noise
        ϵ_pred = denoise_model(xt)

        # Loss: MSE between predicted noise and true noise
        loss_val = mean((ϵ_pred .- ϵ_true).^2)

        # Backprop
        grads = Flux.gradient(denoise_model) do m
            ϵ_pred = m(xt)
            mean((ϵ_pred .- ϵ_true).^2)
        end
        Flux.Optimise.update!(opt_state, denoise_model, grads)
    end

    println("Epoch $epoch complete")
end

println("Diffusion model training done")


