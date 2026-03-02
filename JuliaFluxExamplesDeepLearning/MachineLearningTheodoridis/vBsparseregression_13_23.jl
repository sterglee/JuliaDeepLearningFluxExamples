using LinearAlgebra, Random, Plots, Statistics

# -----------------------------
# 1. Parameters
# -----------------------------
Random.seed!(0)
M = 100
SNR = 15 # Increased SNR for clearer visualization of sparse recovery
K_indices = [22, 64]
x = collect(range(-10, 10, length=M))

# -----------------------------
# 2. Basis matrix (RBF Kernel)
# -----------------------------
# Each column is a basis function centered at x[i]
Phi = [exp(-0.5 * 10.0 * (x[i] - x[j])^2) for i in 1:M, j in 1:M]
    Phi_cov = Phi' * Phi

    # -----------------------------
    # 3. Original signal + noise
    # -----------------------------
    # Truth is a sparse combination of two basis functions
    y0 = Phi[:, K_indices[1]] + Phi[:, K_indices[2]]
    var_y = var(y0)
    var_n = var_y / (10^(SNR/10))
    n_vec = sqrt(var_n) .* randn(M)
    y = y0 + n_vec

    # -----------------------------
    # 4. Least Squares estimate (ML)
    # -----------------------------
    # Standard LS usually overfits significantly in this basis
    w_LS = (Phi_cov + 1e-6*I) \ (Phi' * y) # Added jitter for stability
    y_LS = Phi * w_LS

    # -----------------------------
    # 5. EM algorithm (Evidence Framework / Type-II ML)
    # -----------------------------

    EMiter = 100
    beta_EM = 1.0 / var_n  # Precision of noise
    alpha_EM = 1.0         # Initial precision of weights (scalar for EM version)
    mu_EM = zeros(M)

    for i in 1:EMiter
        # Posterior Covariance: Σ = (βΦᵀΦ + αI)⁻¹
        # Use Symmetric to ensure numerical stability
        Sigma_EM = inv(Symmetric(beta_EM * Phi_cov + alpha_EM * I(M)))
        mu_EM = beta_EM * Sigma_EM * (Phi' * y)

        # Update Hyperparameters (MacKay updates)
        gamma = M - alpha_EM * tr(Sigma_EM)
        alpha_EM = gamma / dot(mu_EM, mu_EM)
        beta_EM = (M - gamma) / norm(y - Phi * mu_EM)^2
    end
    y_EM = Phi * mu_EM

    # -----------------------------
    # 6. Variational Bayes (VB) - Sparse Bayesian Learning
    # -----------------------------

    VBiter = 200
    beta_VB = 1.0 / var_n
    alpha_VB = ones(M) .* 1e-2 # Individual precisions for each weight
    mu_VB = zeros(M)

    # Hyper-hyperparameters (Gamma prior on alpha)
    a_prior = 1e-4
    b_prior = 1e-4

    for i in 1:VBiter
        # Update Posterior Covariance with individual precisions (Automatic Relevance Determination)
        Sigma_VB = inv(Symmetric(beta_VB * Phi_cov + Diagonal(alpha_VB)))
        mu_VB = beta_VB * Sigma_VB * (Phi' * y)

        # Update alpha_j (Variational update for Gamma distribution)
        # This leads to sparsity as most alpha_j will go to infinity
        alpha_VB = (a_prior .+ 0.5) ./ (b_prior .+ 0.5 .* (mu_VB.^2 + diag(Sigma_VB)))

        # Update noise precision beta
        err_norm = norm(y - Phi * mu_VB)^2 + tr(Sigma_VB * Phi_cov)
        beta_VB = (M/2 + a_prior) / (0.5 * err_norm + b_prior)
    end
    y_VB = Phi * mu_VB

    # -----------------------------
    # 7. Plot results
    # -----------------------------

    plt = plot(x, y0, color=:red, lw=3, label="Ground Truth", alpha=0.5)
    scatter!(x, y, marker=:x, color=:black, label="Noisy Obs", markersize=3)
    plot!(x, y_LS, linestyle=:dash, color=:blue, label="Least Squares (Overfits)")
    plot!(x, y_EM, linestyle=:dot, color=:green, lw=2, label="EM (Evidence)")
    plot!(x, y_VB, color=:black, lw=2, label="VB (Sparse)")

    xlabel!("x")
    ylabel!("y")
    title!("Sparse Regression Recovery")
    display(plt)


