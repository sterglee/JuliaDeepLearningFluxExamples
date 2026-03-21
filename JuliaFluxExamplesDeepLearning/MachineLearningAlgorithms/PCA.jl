using LinearAlgebra
using Statistics
using DataFrames
using CSV
using Plots
using MLUtils: splitobs # Modern way to handle train/test splits

# -----------------------------
# 1. Helper Function
# -----------------------------
function standardize(X)
    # dims=1 performs the operation across rows (standardizing each feature)
    m = mean(X, dims=1)
    s = std(X, dims=1)
    return (X .- m) ./ s
end

# -----------------------------
# 2. PCA Struct and Logic
# -----------------------------
mutable struct PCA
    k::Int
    eigen_vals::Vector{Float64}
    eigen_vecs::Matrix{Float64}
    w::Matrix{Float64}

    # Constructor
    PCA(k=2) = new(k, Float64[], Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0))
end

function fit!(pca::PCA, X)
    X_std = standardize(X)
    n_samples = size(X_std, 1)

    # Covariance matrix: (X^T * X) / (n - 1)
    cov_mat = (X_std' * X_std) ./ (n_samples - 1)

    # Eigendecomposition
    # eigen(cov_mat) returns a struct with .values and .vectors
    decomp = eigen(cov_mat)

    # Julia returns eigenvals in ASCENDING order.
    # We want DESCENDING (highest variance first).
    idx = sortperm(decomp.values, rev=true)
    pca.eigen_vals = decomp.values[idx]
    pca.eigen_vecs = decomp.vectors[:, idx]

    # Select the top k eigenvectors to form the transformation matrix W
    pca.w = pca.eigen_vecs[:, 1:pca.k]

    return pca
end

function project(pca::PCA, X)
    # Project data into the new subspace
    return X * pca.w
end

# -----------------------------
# 3. Main Execution
# -----------------------------
function main()
    # Load Data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    df_wine = CSV.read(download(url), DataFrame, header=false)

    # Assign names (optional, but good practice)
    rename!(df_wine, [:Class, :Alcohol, :Malic_acid, :Ash, :Alcalinity, :Magnesium,
                      :Phenols, :Flavanoids, :Nonflavanoid, :Proanthocyanins,
                      :Color, :Hue, :OD280, :Proline])

    # Separate Features and Labels
    X = Matrix(df_wine[:, 2:end])
    y = Vector(df_wine[:, 1])

    # Train/Test Split (70/30) using MLUtils
    # splitobs expects (Features, Samples), so we transpose X
    (X_tr_raw, y_train), (X_te_raw, y_test) = splitobs((X', y), at=0.7, shuffle=true)
    X_train = Matrix(X_tr_raw')

    # Run PCA
    pca = PCA(2)
    fit!(pca, X_train)

    # Standardize data before projecting
    X_train_std = standardize(X_train)
    X_projected = project(pca, X_train_std)

    # Visualization
    classes = unique(y_train)
    markers = [:circle, :star7, :rect]

    p = plot(title="PCA of Wine Dataset (Julia)", xlabel="PC1", ylabel="PC2", legend=:topright)

    for (i, cl) in enumerate(classes)
        mask = y_train .== cl
        scatter!(p, X_projected[mask, 1], X_projected[mask, 2],
                 marker=markers[i], label="Class $cl", alpha=0.7)
    end

    display(p)
end

    main()

