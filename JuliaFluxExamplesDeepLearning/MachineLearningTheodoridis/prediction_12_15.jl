using LinearAlgebra, Random, Plots

# Set seed for reproducibility
Random.seed!(123)

# 1. True signal curve
x_range = collect(0:0.0001:2)
# y = 0.2 - x + 0.9x^2 + 0.7x^3 - 0.2x^5
y_true = @. 0.2 - x_range + 0.9*x_range^2 + 0.7*x_range^3 - 0.2*x_range^5

# 2. Training samples
N = 20
a, b = 0.0, 2.0
x1 = collect(range(a, stop=b - b/N, length=N))

# Noise generation (σ_n is variance)
sigma_n = 0.05
noise = sqrt(sigma_n) .* randn(N)

# True parameters
theta_true = [0.2, -1.0, 0.9, 0.7, -0.2]
l = length(theta_true)

# Measurement matrix Phi (Polynomial basis)
Phi = hcat(ones(N), x1, x1.^2, x1.^3, x1.^5)

# Generate noisy observations
y1 = Phi * theta_true + noise

# 3. Bayesian Inference (Gaussian Prior)
sigma_theta = 2.0
mu_theta_prior = theta_true # Prior mean

# Posterior Covariance: Σ = ( (1/σ_θ²)I + (1/σ_n)ΦᵀΦ )⁻¹
Sigma_theta_pos = inv((1/sigma_theta) * I(l) + (1/sigma_n) * (Phi' * Phi))

# Posterior Mean
mu_theta_pos = mu_theta_prior + (1/sigma_n) * Sigma_theta_pos * Phi' * (y1 - Phi * mu_theta_prior)

# 4. Linear Prediction
Np = 20
x2 = (b - a) .* rand(Np)
Phip = hcat(ones(Np), x2, x2.^2, x2.^3, x2.^5)

# Predicted mean and variance
mu_y_pred = Phip * mu_theta_pos
# Variance calculation (includes observation noise σ_n)
sigma_y_pred = diag(sigma_n .+ sigma_n * sigma_theta .* Phip * inv(sigma_n * I(l) + sigma_theta * (Phi' * Phi)) * Phip')

# 5. Visualization
p = plot(x_range, y_true, color=:black, label="True Curve", lw=2)
scatter!(x2, mu_y_pred, yerror=sigma_y_pred,
         color=:red, m=:x, label="Predictions w/ Error Bars",
         xlabel="x", ylabel="y", title="Linear Prediction with Gaussian Prior")
display(p)


