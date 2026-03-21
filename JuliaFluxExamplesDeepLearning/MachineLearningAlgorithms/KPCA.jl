using LinearAlgebra, Statistics, Plots
using MLDatasets, MultivariateStats, Distances

# 1. Generate Moon Data (Using a simple manual generator for self-containment)
function make_moons(n_samples; noise=0.05)
    t = range(0, stop=pi, length=div(n_samples, 2))
    outer_circ_x = cos.(t)
    outer_circ_y = sin.(t)
    inner_circ_x = 1 .- cos.(t)
    inner_circ_y = 1 .- sin.(t) .- 0.5

    X = vcat(hcat(outer_circ_x, outer_circ_y), hcat(inner_circ_x, inner_circ_y))
    y = vcat(zeros(div(n_samples, 2)), ones(div(n_samples, 2)))
    return X, y
end

X, y = make_moons(100)

# 2. Manual RBF Kernel PCA Function
function rbf_kernel_pca(X, γ, n_components)
    # Calculate pairwise squared Euclidean distances
    # Distances.pairwise expects features in columns, so we transpose X'
    sq_dists = pairwise(SqEuclidean(), X', dims=2)

    # RBF Kernel Matrix
    K = exp.(-γ .* sq_dists)

    # Center the Kernel Matrix
    N = size(K, 1)
    one_n = ones(N, N) ./ N
    K_centered = K - one_n * K - K * one_n + one_n * K * one_n

    # Eigenpair decomposition
    # eigen() returns eigenvalues in ascending order
    vals, vecs = eigen(Symmetric(K_centered))

    # Return the top n_components (last columns)
    return vecs[:, end:-1:end-n_components+1]
end

# 3. Execution and Visualization
X_kpca = rbf_kernel_pca(X, 15, 2)

# Plotting
p1 = scatter(X_kpca[y.==0, 1], X_kpca[y.==0, 2], color=:red, marker=:triangle, label="Class 0")
scatter!(X_kpca[y.==1, 1], X_kpca[y.==1, 2], color=:blue, marker=:circle, label="Class 1", title="Manual Kernel PCA")

# 1D projection plot
p2 = scatter(X_kpca[y.==0, 1], fill(0.02, 50), color=:red, marker=:triangle, alpha=0.5, label="")
scatter!(X_kpca[y.==1, 1], fill(-0.02, 50), color=:blue, marker=:circle, alpha=0.5, label="", ylims=(-1, 1), title="1D Projection")

plot(p1, p2, layout=(1, 2), size=(800, 350))

