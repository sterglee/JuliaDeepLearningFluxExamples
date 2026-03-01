using LinearAlgebra, Random, Plots, Statistics, Distributions

# -----------------------------
# Exercise 12.17 - EM for GMM
# -----------------------------

Random.seed!(0)

# 1. Configuration
N1, N2, N3 = 100, 100, 100
l = 2      # dimension
K = 3      # number of Gaussians
N = N1 + N2 + N3
NofIter = 50

# 2. Generate Synthetic Data
mu1_true = [10.0, 3.0]; Sigma1_true = [1.0 0.2; 0.2 1.0]
mu2_true = [1.0, 1.0];  Sigma2_true = [1.5 -0.4; -0.4 1.5]
mu3_true = [5.0, 4.0];  Sigma3_true = [2.0 0.8; 0.8 2.0]

x1 = mu1_true .+ cholesky(Sigma1_true).L * randn(l, N1)
x2 = mu2_true .+ cholesky(Sigma2_true).L * randn(l, N2)
x3 = mu3_true .+ cholesky(Sigma3_true).L * randn(l, N3)
X = hcat(x1, x2, x3)

# 3. Initialization
Pk = fill(1.0/K, K)
# Corrected random mean initialization
mu = minimum(X, dims=2) .+ rand(l, K) .* (maximum(X, dims=2) .- minimum(X, dims=2))
Sigmak = [Matrix{Float64}(I, l, l) for _ in 1:K]

    gammakn = zeros(K, N)
    lg_lklhd = zeros(NofIter)

    # -----------------------------
    # 4. EM Loop
    # -----------------------------


    for iter = 1:NofIter
        # --- E-step: Evaluate responsibilities ---
        for k = 1:K
            # Vectorized PDF evaluation is faster
            d = MvNormal(mu[:, k], Symmetric(Sigmak[k]))
            gammakn[k, :] = Pk[k] .* pdf(d, X)
        end

        # Normalize responsibilities
        evidence = sum(gammakn, dims=1)
        gammakn ./= (evidence .+ 1e-12)

        # Log-Likelihood storage
        lg_lklhd[iter] = sum(log.(evidence))

        # --- M-step: Update Parameters ---
        Nk = vec(sum(gammakn, dims=2))
        for k = 1:K
            if Nk[k] > 1e-8
                # Update mean (X is 2xN, gammakn[k,:] is Nx1)
                mu[:, k] = (X * gammakn[k, :]) ./ Nk[k]

                # Update covariance
                diff = X .- mu[:, k]
                # Efficient weighted outer product
                Sigmak[k] = (diff .* gammakn[k, :]') * diff' ./ Nk[k] + 1e-6*I(l)
            end
        end

        # Update mixture weights
        Pk = Nk ./ N
    end

    # -----------------------------
    # 5. Visualization
    # -----------------------------
    function plot_ellipse!(mu, Sigma, conf=0.9, col=:black)
        k = sqrt(quantile(Chisq(2), conf))
        eg = eigen(Symmetric(Sigma))
        theta = range(0, 2π, length=100)
        circle = [cos.(theta)'; sin.(theta)']
        A = eg.vectors * Diagonal(sqrt.(max.(0, eg.values)))
        ellipse = (A * circle) .* k
        plot!(ellipse[1,:] .+ mu[1], ellipse[2,:] .+ mu[2], color=col, lw=2, label="")
    end

    # Plotting Results
    p1 = scatter(X[1, 1:N1], X[2, 1:N1], label="Cluster 1", alpha=0.5)
    scatter!(X[1, N1+1:N1+N2], X[2, N1+1:N1+N2], label="Cluster 2", alpha=0.5)
    scatter!(X[1, N1+N2+1:end], X[2, N1+N2+1:end], label="Cluster 3", alpha=0.5)

    for k in 1:K
        plot_ellipse!(mu[:, k], Sigmak[k], 0.9, :black)
        scatter!([mu[1,k]], [mu[2,k]], color=:white, markersize=5, markerstrokewidth=2, label="")
    end
    title!("GMM Recovery (EM)")


    p2 = plot(lg_lklhd, title="Convergence", xlabel="Iteration", ylabel="Log-Likelihood", lw=2, color=:red, legend=false)

    plot(p1, p2, layout=(1,2), size=(1000, 450), margin=5Plots.mm)


