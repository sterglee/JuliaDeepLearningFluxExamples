using Distributions, Plots, LinearAlgebra

# 1. Define the true data distribution (Mixture of Gaussians)
#
pr_x_true(x) = 0.5 * pdf(Normal(-1.5, 0.4), x) + 0.5 * pdf(Normal(1.0, 0.5), x)

# 2. Diffusion Marginal Distribution (Equation 18.7)
# Pr(z_t | z_0) = N(z_t; sqrt(1-beta_t)*z_0, beta_t*I)
#
function diffusion_marginal(x_vals, p_x0, t, beta)
    # Cumulative alpha (alpha_bar) is the product of (1 - beta) up to time t
    #
    alpha_bar = prod(1.0 .- beta[1:t])

    # Pr(z_t) = ∫ Pr(z_t | z_0) Pr(z_0) dz_0
    # Numerical integration over the provided x range
    pr_zt = zeros(length(x_vals))
    for (i, z0) in enumerate(x_vals)
        μ = sqrt(alpha_bar) * z0
        σ = sqrt(1.0 - alpha_bar)
        pr_zt .+= p_x0[i] .* pdf.(Normal(μ, σ), x_vals)
    end
    return pr_zt ./ sum(pr_zt) # Normalize
end

# Setup schedule
T = 100
beta = range(0.001, 0.02, length=T) # Simple linear schedule

x_plot = range(-3, 3, length=601)
p_initial = pr_x_true.(x_plot)

# Visualize the diffusion process (the "Encoder")
p1 = plot(x_plot, p_initial, title="t=0 (Data)", fill=(0, 0.5, :blue))
p2 = plot(x_plot, diffusion_marginal(x_plot, p_initial, 50, beta), title="t=50")
p3 = plot(x_plot, diffusion_marginal(x_plot, p_initial, 100, beta), title="t=100 (Noise)")
plot(p1, p2, p3, layout=(1,3))

