using Distributions, Plots, LinearAlgebra

# 1. Define the Latent Variable Model
# Prior: z ~ N(0, 1)
z_prior = Normal(0.0f0, 1.0f0)

# Non-linear mapping from latent space to data space
function mapping_func(z)
    x1 = 1.5 * sin(0.5 * z)
    x2 = 0.5 * cos(2.0 * z)
    return [x1, x2]
end

# Likelihood: x | z ~ N(f(z), σ²I)
σ = 0.2f0

# 2. Compute the Likelihood Pr(x1, x2 | z)
function get_likelihood(x1, x2, z_val)
    μ = mapping_func(z_val)
    # Using the Gaussian density formula
    term1 = 1.0 / (2.0 * π * σ^2)
    dist_sq = (x1 - μ[1])^2 + (x2 - μ[2])^2
    return term1 * exp(-dist_sq / (2.0 * σ^2))
end

# 3. Compute the Posterior Pr(z | x1, x2)
# Since we cannot integrate analytically, we use numerical integration (summation)
function get_posterior(x1, x2)
    z_range = range(-3, 3, length=601)

    # Numerator: Likelihood * Prior
    unnormalized_posterior = [get_likelihood(x1, x2, z) * pdf(z_prior, z) for z in z_range]

        # Normalize such that the posterior sums to 1
        posterior = unnormalized_posterior ./ sum(unnormalized_posterior)
        return z_range, posterior
    end

    # Execution
    x_obs = [0.9, -0.9]
    z_axis, pr_z = get_posterior(x_obs[1], x_obs[2])

    plot(z_axis, pr_z, title="Posterior Distribution Pr(z | x)", xlabel="z", ylabel="Density", lw=2)

