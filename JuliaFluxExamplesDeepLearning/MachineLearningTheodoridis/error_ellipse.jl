using LinearAlgebra, Random, Distributions, Plots, Clustering

# --- Data Configuration ---
N1, N2 = 100, 20
N = N1 + N2
K, l = 2, 2
Random.seed!(0)

# Define True Gaussians
mu_true = [0.9 1.02; -1.2 -1.3]
Sigma1 = [0.5 0.081; 0.081 0.7]
Sigma2 = [0.4 0.02; 0.02 0.3]

# Generate Samples
x1 = rand(MvNormal(mu_true[:, 1], Sigma1), N1)'
x2 = rand(MvNormal(mu_true[:, 2], Sigma2), N2)'
X = collect(hcat(x1', x2')) # Data matrix: l x N

# Colors
marker_color = [:red, :black, :gray]

# --- 1. Plot Generated Data ---
p1 = scatter(x1[:,1], x1[:,2], mc=marker_color[2], label="True G1", ratio=:equal, title="Raw Data")
scatter!(x2[:,1], x2[:,2], mc=marker_color[1], label="True G2")

# --- 2. K-means Algorithm ---
# Clustering.jl expects data where each column is an observation
result_km = kmeans(X, K)
bel = assignments(result_km) # Cluster labels

p2 = scatter(ratio=:equal, title="K-means Result")
for k in 1:K
  scatter!(X[1, bel.==k], X[2, bel.==k], mc=marker_color[k], label="Cluster $k")
end

# --- 3. EM Algorithm Setup ---


NofIter = 300
conf = 0.8
Pk = fill(1/K, K)
mu = randn(l, K)
Sigmak = [Matrix(1.0I, l, l) for _ in 1:K]
  gammakn = zeros(K, N)

  # Function to plot confidence ellipses (Helper)
  function plot_ellipse!(mu_vec, Sigma_mat; conf=0.8, color=:red)
    # Get quantile for chi-square with 2 DoF
    k_scale = sqrt(quantile(Chisq(2), conf))
    eg = eigen(Sigma_mat)
    θ = range(0, 2π, length=100)
    circle = [cos.(θ)'; sin.(θ)']
    # Transform: eigvec * sqrt(eigval) * unit_circle
    ellipse = (eg.vectors * Diagonal(sqrt.(max.(0, eg.values))) * circle) .* k_scale
    plot!(ellipse[1,:] .+ mu_vec[1], ellipse[2,:] .+ mu_vec[2], c=color, label="", lw=1.5)
  end

  # --- 4. EM Algorithm Iteration ---
  for i in 1:NofIter
    # E-step: Responsibilities
    for k in 1:K
      dist = MvNormal(mu[:, k], Symmetric(Sigmak[k]))
      gammakn[k, :] = Pk[k] .* pdf(dist, X)
    end
    # Normalize across clusters (columns)
    sum_gamma = sum(gammakn, dims=1)
    gammakn ./= (sum_gamma .+ 1e-15)

    # M-step: Update Parameters
    Nk = vec(sum(gammakn, dims=2))
    for k in 1:K
      if Nk[k] > 1e-5
        # Update Mean
        mu[:, k] = (X * gammakn[k, :]) ./ Nk[k]

        # Update Covariance
        diff = X .- mu[:, k]
        # Weighted outer product
        Sigmak[k] = ((diff .* reshape(gammakn[k, :], 1, N)) * diff') ./ Nk[k]
        Sigmak[k] += 1e-5 * I # Regularization
      end
    end
    Pk = Nk ./ N
  end

  # --- 5. Plot EM Results ---
  p3 = scatter(x1[:,1], x1[:,2], mc=marker_color[2], alpha=0.5, label="Raw G1", ratio=:equal, title="EM Result")
  scatter!(x2[:,1], x2[:,2], mc=marker_color[1], alpha=0.5, label="Raw G2")

  for k in 1:K
    plot_ellipse!(mu[:, k], Sigmak[k], conf=conf, color=:blue)
  end

  # Display all plots
  plot(p1, p2, p3, layout=(1, 3), size=(1000, 350))

