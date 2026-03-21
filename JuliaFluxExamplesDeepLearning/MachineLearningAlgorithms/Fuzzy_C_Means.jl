using LinearAlgebra, Statistics, Random, Plots

# Define the structure to hold our FCM state
mutable struct FCM
    k::Int
    m::Float64
    tol::Float64
    max_iter::Int
    weights::Matrix{Float64}
    centroids::Matrix{Float64}

    # Constructor
    FCM(; k=1, m=2.0, tol=1e-5, max_iter=100) = new(k, m, tol, max_iter, Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0))
end

# Calculate centroids based on current membership weights
function calculate_centroids!(model::FCM, X::Matrix{Float64})
    # W^m: weight each point by the fuzziness coefficient
    Wm = model.weights .^ model.m

    # Centroid formula: Σ (w^m * x) / Σ w^m
    num = Wm' * X
    den = sum(Wm, dims=1)'
    model.centroids = num ./ den
end

# Update membership degrees
function update_members!(model::FCM, X::Matrix{Float64})
    N = size(X, 1)
    K = model.k
    new_W = zeros(N, K)

    # Calculate distances between all points and centroids
    # dists[i, j] is distance from point i to centroid j
    dists = [norm(X[i, :] - model.centroids[j, :]) for i in 1:N, j in 1:K]

    # Avoid division by zero
    dists = max.(dists, 1e-10)

    # Fuzzy membership update logic
    p = 2 / (model.m - 1)
    for i in 1:N
        for j in 1:K
            # W_ij = 1 / Σ (d_ij / d_is)^p
            ratio_sum = sum((dists[i, j] ./ dists[i, :]) .^ p)
            new_W[i, j] = 1.0 / ratio_sum
        end
    end
    model.weights = new_W
end

# The fit function
function fit!(model::FCM, X::Matrix{Float64})
    N, n_features = size(X)
    rng = MersenneTwister(5) # fixed seed

    # Initialize weights randomly and normalize rows
    W = rand(rng, N, model.k)
    model.weights = W ./ sum(W, dims=2)

    for i in 1:model.max_iter
        old_centroids = isempty(model.centroids) ? zeros(model.k, n_features) : copy(model.centroids)

        calculate_centroids!(model, X)
        update_members!(model, X)

        # Check convergence
        if i > 1 && norm(model.centroids - old_centroids) < model.tol
            println("Converged at iteration $i")
            break
        end
    end
    return model
end

# --- Main Logic ---

# 1. Generate synthetic data (5 blobs)
function make_blobs(n, centers)
    X = vcat([randn(n, 2) .+ center' for center in centers]...)
    y = vcat([fill(i, n) for i in 1:length(centers)]...)
    return X, y
end

centers = [[-5, -5], [5, 5], [-5, 5], [5, -5], [0, 0]]
X, y = make_blobs(30, centers)

# 2. Run FCM
km = FCM(k=5, m=2.0, max_iter=100)
fit!(km, X)

# 3. Visualization
# Determine hard clusters for plotting colors
y_km = [argmax(km.weights[i, :]) for i in 1:size(X, 1)]

colors = [:lightgreen, :orange, :lightblue, :yellow, :magenta]
p = scatter(X[:, 1], X[:, 2], group=y_km, color=reshape(colors, 1, 5),
            marker=[:rect :circle :utriangle :dtriangle :diamond],
            label=["Cluster $i" for i in 1:5], legend=:outerright, title="Fuzzy C-Means (Julia)")

# Plot centroids
scatter!(km.centroids[:, 1], km.centroids[:, 2],
         color=:red, marker=:star, markersize=10, label="Centroids")

display(p)


