using Flux, Statistics, Random, Plots

# ==========================================
# 1. Setup Data and Models
# ==========================================
Random.seed!(42)
n_data = 100
# Real data centered at 2.5
x_real = randn(Float32, 1, n_data) .+ 2.5f0
z = randn(Float32, 1, n_data)                # Latent noise

# Simple linear models for the toy example
# Generator: x = z + θ
generator(z, θ) = z .+ θ

# Discriminator: sig(ϕ0 + ϕ1*x)
discriminator(x, ϕ0, ϕ1) = sigmoid.(ϕ0 .+ ϕ1 .* x)

# ==========================================
# 2. Training Logic
# ==========================================

# Loss for discriminator: -E[log(D(x_real))] - E[log(1 - D(x_syn))]
function d_loss(x_real, x_syn, ϕ0, ϕ1)
    # Using small epsilon to prevent log(0)
    ϵ = 1f-7
    real_loss = -mean(log.(discriminator(x_real, ϕ0, ϕ1) .+ ϵ))
    syn_loss = -mean(log.(1f0 .- discriminator(x_syn, ϕ0, ϕ1) .+ ϵ))
    return real_loss + syn_loss
end

# Loss for generator: -E[log(D(G(z)))]
function g_loss(z, θ, ϕ0, ϕ1)
    ϵ = 1f-7
    return -mean(log.(discriminator(generator(z, θ), ϕ0, ϕ1) .+ ϵ))
end

# Optimization Loop
function train_gan(x_real, z; n_iters=5)
    # Initial parameters
    θ = [0.0f0]      # Vectorized for easier Flux handling
    ϕ0 = [-1.0f0]
    ϕ1 = [0.5f0]
    lr = 0.1f0



    for i in 1:n_iters
        # 1. Update Discriminator (300 steps as per notebook)
        for _ in 1:300
            # Generate synthetic data (don't track gradients through x_syn here)
            x_syn = generator(z, θ[1])

            # Compute gradients for ϕ0 and ϕ1
            grads = Flux.gradient(ϕ0, ϕ1) do p0, p1
                d_loss(x_real, x_syn, p0[1], p1[1])
            end

            # Apply updates
            ϕ0 .-= lr .* grads[1]
            ϕ1 .-= lr .* grads[2]
        end

        # 2. Update Generator (3 steps)
        for _ in 1:3
            # Compute gradients for θ
            grads = Flux.gradient(θ) do t
                g_loss(z, t[1], ϕ0[1], ϕ1[1])
            end

            # Apply update
            θ .-= lr .* grads[1]
        end

        println("Iter $i: Generator θ = $(round(θ[1], digits=3))")
    end
    return θ[1], ϕ0[1], ϕ1[1]
end

θ_final, ϕ0_f, ϕ1_f = train_gan(x_real, z)

