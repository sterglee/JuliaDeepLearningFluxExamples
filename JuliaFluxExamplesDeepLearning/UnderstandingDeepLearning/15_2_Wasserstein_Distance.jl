using Flux, Statistics, Random, Plots

# ==========================================
# 1. Setup Data and Models
# ==========================================
Random.seed!(42)
n_data = 100
# Real data centered at 2.5
x_real = randn(Float32, 1, n_data) .+ 2.5f0
z = randn(Float32, 1, n_data)                # Latent noise

# Simple linear models
generator(z, θ) = z .+ θ
discriminator(x, ϕ0, ϕ1) = sigmoid.(ϕ0 .+ ϕ1 .* x)

# ==========================================
# 2. Loss Functions
# ==========================================
function d_loss(x_real, x_syn, ϕ0, ϕ1)
    ϵ = 1f-7
    real_loss = -mean(log.(discriminator(x_real, ϕ0, ϕ1) .+ ϵ))
    syn_loss = -mean(log.(1f0 .- discriminator(x_syn, ϕ0, ϕ1) .+ ϵ))
    return real_loss + syn_loss
end # <--- End of d_loss

function g_loss(z, θ, ϕ0, ϕ1)
    ϵ = 1f-7
    return -mean(log.(discriminator(generator(z, θ), ϕ0, ϕ1) .+ ϵ))
end # <--- End of g_loss

# ==========================================
# 3. Training Loop
# ==========================================
function train_gan(x_real, z; n_iters=5)
    # Parameters wrapped in arrays to be mutable
    θ = [0.0f0]
    ϕ0 = [-1.0f0]
    ϕ1 = [0.5f0]
    lr = 0.1f0

    for i in 1:n_iters
        # 1. Update Discriminator (Inner Loop)
        for _ in 1:300
            x_syn = generator(z, θ[1])

            # Explicit gradient calculation
            grads = Flux.gradient(ϕ0, ϕ1) do p0, p1
                d_loss(x_real, x_syn, p0[1], p1[1])
            end # <--- End of gradient do-block

            # Update parameters
            ϕ0 .-= lr .* grads[1]
            ϕ1 .-= lr .* grads[2]
        end # <--- End of discriminator loop

        # 2. Update Generator (Inner Loop)
        for _ in 1:3
            grads = Flux.gradient(θ) do t
                g_loss(z, t[1], ϕ0[1], ϕ1[1])
            end # <--- End of gradient do-block

            θ .-= lr .* grads[1]
        end # <--- End of generator loop

        println("Iter $i: Generator θ = $(round(θ[1], digits=3))")
    end # <--- End of n_iters loop

    return θ[1], ϕ0[1], ϕ1[1]
end # <--- End of train_gan

# Execute
θ_final, ϕ0_f, ϕ1_f = train_gan(x_real, z)

