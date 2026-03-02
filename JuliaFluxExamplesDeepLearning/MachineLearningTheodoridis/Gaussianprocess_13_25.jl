using LinearAlgebra, Random, Statistics, Plots

# -----------------------------------------------------------------
# 1. Setup and Data Generation
# -----------------------------------------------------------------
Random.seed!(0)
N = 20
h = 0.5           # Length scale
noise_var = 0.1   # Noise variance

# Input samples
x = randn(N)

# Define Kernel Function (Squared Exponential / RBF)
# Using k(x, x') = exp(-0.5 * |x-x'|^2 / h^2)
kernel(x1, x2, h) = exp(-0.5 * (x1 - x2)^2 / h^2)

# Compute Training Kernel Matrix K
K = [kernel(i, j, h) for i in x, j in x]

    # Generate noisy data: y = L*z + noise
    # Note: K must be positive definite; we add a small jitter if needed
    L = cholesky(K + 1e-9*I).L
    y = L * randn(N) + sqrt(noise_var) * randn(N)

    # -----------------------------------------------------------------
    # 2. Prediction (Matrix Version)
    # -----------------------------------------------------------------
    D = 100
    xp = collect(range(-3, 4, length=D)) # Prediction points

    # Kxxp: Kernel between training points x and prediction points xp
    Kxxp = [kernel(i, j, h) for j in xp, i in x] # D x N matrix

        # Kxp: Kernel between prediction points xp and itself
        Kxp = [kernel(i, j, h) for i in xp, j in xp] # D x D matrix

            # Sigma_N: Training covariance with noise
            Sigma_N = K + noise_var * I

            # Solve for weights (alpha) using backslash (efficient Cholesky solver)
            alpha = Sigma_N \ y

            # Compute Predictive Mean: μ = K_xp_x * (K_x_x + σ²I)⁻¹ * y
            mu_f = Kxxp * alpha

            # Compute Predictive Covariance: Σ = K_xp_xp - K_xp_x * (K_x_x + σ²I)⁻¹ * K_x_xp
            # We use Sigma_N \ Kxxp' to avoid explicit matrix inversion
            Sigma_f = Kxp - Kxxp * (Sigma_N \ Kxxp')

            # Extract standard deviation for the 2σ (95%) confidence interval
            std_f = sqrt.(abs.(diag(Sigma_f)))
            upper = mu_f .+ 2 .* std_f
            lower = mu_f .- 2 .* std_f

