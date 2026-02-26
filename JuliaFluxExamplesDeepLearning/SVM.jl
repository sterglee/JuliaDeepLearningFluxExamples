using Optim
using LinearAlgebra
using Statistics
using DataFrames
using CSV
using Plots

# -----------------------------
# 1. SVM Structure and Logic
# -----------------------------
mutable struct SVM
    C::Float64
    opt::String
    # Results stored after fit
    lambdas_s::Vector{Float64}
    ys::Vector{Float64}
    Xs::Matrix{Float64}
    theta_hat::Vector{Float64}
    theta0::Float64

    SVM(; C=100.0, opt="linear") = new(C, opt, [], [], Matrix{Float64}(undef,0,0), [], 0.0)
end

function kernel_matrix(svm::SVM, X, Y)
    if svm.opt == "linear"
        return X * Y'
    else
        return X * Y' # Placeholder for other kernels
    end
end

function fit!(svm::SVM, X, y)
    n_samples = size(X, 1)
    K = kernel_matrix(svm, X, X)

    # Dual Objective: min 0.5 * λ' * (diag(y) * K * diag(y)) * λ - 1' * λ
    # We use a closure so the optimizer can see K and y
    function objective(λ)
        zz = λ .* y
        return -sum(λ) + 0.5 * (zz' * K * zz)
    end

    # Gradient (Jacobian) of the dual objective
    function g!(G, λ)
        zz = λ .* y
        G .= (K * zz) .* y .- 1.0
    end

    # Constraints: 0 <= λ <= C and sum(λ .* y) == 0
    # Optim.jl handles box constraints with Fminbox
    lower = zeros(n_samples)
    upper = fill(svm.C, n_samples)
    initial_λ = fill(svm.C / 2, n_samples) # Warm start

    # Solve using Fminbox with L-BFGS
    # Note: For strict equality constraints (sum(λy)=0),
    # one usually uses a specialized QP solver like JuMP or COPT,
    # but for this script, we'll focus on the box constraints.
    results = optimize(objective, g!, lower, upper, initial_λ, Fminbox(LBFGS()))

    lambdas_all = results.minimizer
    # Identify Support Vectors
    idx = findall(x -> x > 1e-5, lambdas_all)

    svm.lambdas_s = lambdas_all[idx]
    svm.ys = y[idx]
    svm.Xs = X[idx, :]

    # Use vec() to convert the 1x2 Matrix result into a Vector{Float64}
    svm.theta_hat = vec((svm.lambdas_s .* svm.ys)' * svm.Xs)

    # Ensure the subtraction works for the scalar theta0
    svm.theta0 = mean(svm.ys .- (svm.Xs * svm.theta_hat))

    return svm
end

function predict(svm::SVM, Xtest)
    k = kernel_matrix(svm, Xtest, svm.Xs)
    return k * (svm.lambdas_s .* svm.ys) .+ svm.theta0
end

# -----------------------------
# 2. Main Execution
# -----------------------------

function main()
    # Load Iris Data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    df = CSV.read(download(url), DataFrame, header=false)

    # Filter for first 100 samples (Setosa and Versicolor)
    df_binary = df[1:100, :]
    y = [row.Column5 == "Iris-setosa" ? -1.0 : 1.0 for row in eachrow(df_binary)]
        X = Matrix{Float64}(df_binary[:, [1, 3]]) # Sepal length, Petal length

        # Standardize
        X_std = (X .- mean(X, dims=1)) ./ std(X, dims=1)

        # Fit SVM
        svm = SVM(C=1.0)
        fit!(svm, X_std, y)

        # Plot Decision Regions
        x_rng = range(minimum(X_std[:,1])-0.5, maximum(X_std[:,1])+0.5, length=100)
        y_rng = range(minimum(X_std[:,2])-0.5, maximum(X_std[:,2])+0.5, length=100)

        # Create a heatmap of the decision function
        decision_surface = [predict(svm, [x y])[1] for x in x_rng, y in y_rng]

            p = contourf(x_rng, y_rng, sign.(decision_surface)', alpha=0.3, color=:coolwarm)
            scatter!(p, X_std[y .== -1, 1], X_std[y .== -1, 2], label="Setosa", color=:blue)
            scatter!(p, X_std[y .== 1, 1], X_std[y .== 1, 2], label="Versicolor", color=:red)
            plot!(p, title="SVM Decision Boundary (Iris)", xlabel="Sepal Length", ylabel="Petal Length")

            display(p)
        end

        main()

