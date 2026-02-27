using Flux
using Distributions
using Plots
using LinearAlgebra
using Random

# ==========================================
# 1. Base Distribution
# ==========================================
# Normalizing flows start with a simple base distribution Pr(z)
# Typically a standard normal distribution N(0, 1)
base_dist = Normal(0.0f0, 1.0f0)

# ==========================================
# 2. Invertible Mappings (The "Flow")
# ==========================================
# A normalizing flow is a sequence of invertible functions g(z)
# Here we implement a simple linear transformation: x = ϕ₀ + ϕ₁z

"""
forward_flow(z, phi0, phi1)
Maps from latent space z to data space x: x = g(z)
"""
forward_flow(z, phi0, phi1) = phi0 .+ phi1 .* z

"""
inverse_flow(x, phi0, phi1)
Maps from data space x back to latent space z: z = f(x)
"""
inverse_flow(x, phi0, phi1) = (x .- phi0) ./ phi1

# ==========================================
# 3. Change of Variables Formula
# ==========================================
# To find the density Pr(x), we use the change of variables formula:
# Pr(x) = Pr(z) * |dz/dx|  where z = f(x)

function compute_density(x, phi0, phi1)
    # 1. Map back to latent space
    z = inverse_flow(x, phi0, phi1)

    # 2. Compute base density at z
    prob_z = pdf.(base_dist, z)

    # 3. Compute the absolute value of the Jacobian determinant |dz/dx|
    # For z = (x - phi0) / phi1, the derivative dz/dx is 1 / phi1
    abs_det_jacobian = abs(1.0f0 / phi1)

    # 4. Final density
    return prob_z .* abs_det_jacobian
end

# ==========================================
# 4. Execution and Visualization
# ==========================================
# Parameters from the notebook: shift by 0.5, scale by 0.8
phi0 = 0.5f0
phi1 = 0.8f0

x_range = range(-3, 3, length=200)
pr_x = compute_density(x_range, phi0, phi1)

# Plotting the result
plot(x_range, pr_x, lw=3, title="1D Normalizing Flow Result",
     xlabel="x", ylabel="Pr(x)", label="Transformed Density")
