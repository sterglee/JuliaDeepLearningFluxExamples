using LinearAlgebra, Random, Plots, Statistics, Distributions

# -----------------------------------------------------------------
# Helper Function: K-means (Custom implementation for comparison)
# -----------------------------------------------------------------
function k_means_custom(X, theta_init)
    l, N = size(X)
    l, m = size(theta_init)
    theta = copy(theta_init)
    bel = zeros(Int, N)
    e = 1.0
    J = 0.0

    while e > 1e-6
        theta_old = copy(theta)

        # 1. Assignment Step (Expectation)
        dist_all = zeros(m, N)
        for j in 1:m
            # Squared Euclidean distance
            dist_all[j, :] = sum((X .- theta[:, j]).^2, dims=1)
        end

        # Find index of minimum distance for each point
        for n in 1:N
            min_val, idx = findmin(dist_all[:, n])
            bel[n] = idx
        end

        # Calculate Cost J
        J = sum(minimum(dist_all, dims=1))

        # 2. Update Step (Maximization)
        for j in 1:m
            mask = (bel .== j)
            if sum(mask) > 0
                theta[:, j] = mean(X[:, mask], dims=2)
            end
        end

        # Check Convergence
        e = sum(abs.(theta .- theta_old))
    end
    return theta, bel, J
end

# -----------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------
Random.seed!(123)

# Data points
N1 = 100
N2 = 20
N = N1 + N2
K = 2
l = 2

# Generate Synthetic Data
mu_true = [0.9 1.02; -1.2 -1.3]
Sigma1 = [0.5 0.081; 0.081 0.7]
Sigma2 = [0.4 0.02; 0.02 0.3]

# Generate samples
x1 = rand(MvNormal(mu_true[:, 1], Sigma1), N1) # 2 x N1
x2 = rand(MvNormal(mu_true[:, 2], Sigma2), N2) # 2 x N2
X = hcat(x1, x2)

marker_colors = [:red, :black]

# --- K-means Algorithm ---
# Initialize centroids by picking random points from data
theta_init = X[:, randperm(N)[1:K]]
theta_km, bel_km, J_km = k_means_custom(X, theta_init)

p1 = scatter(ratio=:equal, title="K-means Clustering")
for k in 1:K
    scatter!(X[1, bel_km .== k], X[2, bel_km .== k], mc=marker_colors[k], label="Cluster $k", markerstrokewidth=0)
end

# --- EM Algorithm ---
NofIter = 100
conf = 0.8
Pk = fill(1/K, K)
mu_em = randn(l, K)
Sigmak = [Matrix(1.0I, l, l) for _ in 1:K]
    gammakn = zeros(K, N)



    for i in 1:NofIter
        # E-step
        for k in 1:K
            d = MvNormal(mu_em[:, k], Symmetric(Sigmak[k]))
            gammakn[k, :] = Pk[k] .* pdf(d, X)
        end
        evidence = sum(gammakn, dims=1)
        gammakn ./= (evidence .+ 1e-15) # Responsibilities

        # M-step
        Nk = vec(sum(gammakn, dims=2))
        for k in 1:K
            if Nk[k] > 1e-5
                mu_em[:, k] = (X * gammakn[k, :]) ./ Nk[k]
                diff = X .- mu_em[:, k]
                Sigmak[k] = ((diff .* gammakn[k, :]') * diff') ./ Nk[k] + 1e-5*I
            end
        end
        Pk = Nk ./ N
    end

    # --- Visualization ---
    p2 = scatter(x1[1,:], x1[2,:], mc=marker_colors[2], alpha=0.3, label="Original G1", ratio=:equal, title="EM Recovery")
    scatter!(x2[1,:], x2[2,:], mc=marker_colors[1], alpha=0.3, label="Original G2")

    function plot_ellipse!(mu, Sigma, conf=0.8)
        k = sqrt(quantile(Chisq(2), conf))
        eg = eigen(Symmetric(Sigma))
        theta = range(0, 2π, length=100)
        circle = [cos.(theta)'; sin.(theta)']
        A = eg.vectors * Diagonal(sqrt.(max.(0, eg.values)))
        ellipse = (A * circle) .* k
        plot!(ellipse[1,:] .+ mu[1], ellipse[2,:] .+ mu[2], color=:blue, lw=2, label="")
    end

    for k in 1:K
        plot_ellipse!(mu_em[:, k], Sigmak[k], conf)
    end

    # Combine Plots
    plot(p1, p2, layout=(1, 2), size=(900, 400))


