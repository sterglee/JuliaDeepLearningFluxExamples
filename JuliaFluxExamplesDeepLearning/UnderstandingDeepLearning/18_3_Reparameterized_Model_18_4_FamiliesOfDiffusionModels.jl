# 1. Reparameterized Loss (Algorithm 18.1)
# L = || eps - eps_theta(sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*eps, t) ||^2
#
function diffusion_loss(model, x0, beta)
    t = rand(1:length(beta))
    ϵ = randn(Float32, size(x0))
    α_bar = prod(1.0 .- beta[1:t])

    # Create noisy sample z_t
    zt = sqrt(α_bar) .* x0 .+ sqrt(1.0 - α_bar) .* ϵ

    # Predict noise and return MSE
    return mean((ϵ .- model(vcat(zt, Float32[t/length(beta)]))).^2)
end


"""
DDIM Sampling Step (Non-stochastic)

"""
function ddim_step(model, zt, t, beta)
    T_total = length(beta)
    α_bar_t = prod(1.0 .- beta[1:t])
    α_bar_prev = t > 1 ? prod(1.0 .- beta[1:t-1]) : 1.0f0

    # 1. Predict noise
    ϵ_pred = model(vcat(zt, Float32[t/T_total]))

    # 2. Estimate z_0 (Predicted x0)
    z0_pred = (zt .- sqrt(1.0 - α_bar_t) .* ϵ_pred) ./ sqrt(α_bar_t)

    # 3. Compute z_{t-1} directionally
    zt_prev = sqrt(α_bar_prev) .* z0_pred .+ sqrt(1.0 - α_bar_prev) .* ϵ_pred

    return zt_prev
end

# Sampling 10x faster (skipping steps)
function sample_accelerated(model, beta, steps=10)
    z = randn(Float32, 1)
    for t in range(length(beta), 1, step=-Int(length(beta)/steps))
        z = ddim_step(model, z, Int(t), beta)
    end
    return z
end

