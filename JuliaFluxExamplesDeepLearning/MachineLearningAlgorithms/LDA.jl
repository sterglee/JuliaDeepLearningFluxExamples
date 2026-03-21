using LinearAlgebra
using Statistics
using DataFrames
using CSV
using Plots

# -----------------------------
# 1. LDA Structure and Logic
# -----------------------------
mutable struct LDA
    theta::Vector{Float64}
    b::Float64
    cls::Vector{Int}

    LDA() = new(Float64[], 0.0, Int[])
end

function fit!(lda::LDA, X, y)
    n_samples = length(y)
    lda.cls = sort(unique(y))

    # Split data by class
    X1 = X[y .== lda.cls[1], :]
    X2 = X[y .== lda.cls[2], :]

    # Class priors
    p1 = size(X1, 1) / n_samples
    p2 = size(X2, 1) / n_samples

    # Means
    mu1 = mean(X1, dims=1)
    mu2 = mean(X2, dims=1)

    # Within-class scatter (Covariance matrices)
    # Using ' (adjoint) for (X-mu)T * (X-mu)
    Q1 = (1 / (size(X1, 1) - 1)) * (X1 .- mu1)' * (X1 .- mu1)
    Q2 = (1 / (size(X2, 1) - 1)) * (X2 .- mu2)' * (X2 .- mu2)

    # Pooled Covariance
    Q = p1 .* Q1 .+ p2 .* Q2

    # Solve for theta: Q * theta = (mu1 - mu2)
    # Using \ is more stable than inv(Q) * (mu1 - mu2)
    # we use vec() to ensure it's a 1D Vector
    lda.theta = vec(Q \ (mu1 .- mu2)')

    # Calculate threshold b
    b1 = mean(X1 * lda.theta)
    b2 = mean(X2 * lda.theta)
    lda.b = -0.5 * (b1 + b2) # Adjusted sign for prediction logic

    return lda
end

function predict(lda::LDA, X)
    # Vectorized decision: theta * x + b
    # If > 0, return cls[1], else cls[2]
    decisions = X * lda.theta .+ lda.b
    return [d > 0 ? lda.cls[1] : lda.cls[2] for d in decisions]
    end

    # -----------------------------
    # 2. Main Execution
    # -----------------------------
    function main()
        # Load Iris Data
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        df = CSV.read(download(url), DataFrame, header=false)

        # Use first 100 samples (Setosa and Versicolor)
        df_binary = df[1:100, :]
        y = [row.Column5 == "Iris-setosa" ? 0 : 1 for row in eachrow(df_binary)]
            X = Matrix{Float64}(df_binary[:, [1, 3]])

            # Standardize
            X_std = (X .- mean(X, dims=1)) ./ std(X, dims=1)

            # Fit LDA
            lda = LDA()
            fit!(lda, X_std, y)

            # Visualization
            x_rng = range(minimum(X_std[:,1])-0.5, maximum(X_std[:,1])+0.5, length=100)
            y_rng = range(minimum(X_std[:,2])-0.5, maximum(X_std[:,2])+0.5, length=100)

            # Grid for decision regions
            z = [predict(lda, reshape([x, y], 1, 2))[1] for x in x_rng, y in y_rng]

                p = contourf(x_rng, y_rng, z', alpha=0.3, color=:coolwarm, title="Fisher's LDA")
                scatter!(p, X_std[y .== 0, 1], X_std[y .== 0, 2], label="Setosa", color=:blue)
                scatter!(p, X_std[y .== 1, 1], X_std[y .== 1, 2], label="Versicolor", color=:red)
                xlabel!(p, "Sepal length [std]")
                ylabel!(p, "Petal length [std]")

                display(p)
            end

            main()

