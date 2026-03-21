using LinearAlgebra, Random, Plots, StatsBase

# Define the IST structure
mutable struct IST
    μ::Float64        # Learning rate (step size)
    c::Float64        # Regularization constant
    max_iter::Int
    err_tol::Float64
    k0::Int           # Sparsity level (for hard thresholding)
    alpha::Float64    # Threshold value (c * μ)
    theta::Vector{Float64}

    # Constructor
    IST(; μ=1.0, c=0.1, max_iter=100, err_tol=0.0001, k0=3) =
        new(μ, c, max_iter, err_tol, k0, c * μ, Float64[])
end

# Hard Thresholding: Keep only the k0 largest components
function hard_thresholding(model::IST, x::Vector{Float64})
    L = length(x)
    # Get indices of the k0 largest absolute values
    idx = partialsortperm(abs.(x), 1:model.k0, rev=true)

    res = zeros(L)
    res[idx] = x[idx]
    return res
end

# Soft Thresholding: prox operator for L1 norm
function soft_thresholding(model::IST, x::Vector{Float64})
    # Formula: sign(x) * max(0, |x| - α)
    return sign.(x) .* max.(0.0, abs.(x) .- model.alpha)
end

function estimate!(model::IST, X::Matrix{Float64}, y::Vector{Float64})
    N, L = size(X)
    theta = zeros(L)
    err_vec = copy(y)

    for ii in 1:model.max_iter
        if norm(err_vec) <= model.err_tol
            break
        end

        # Gradient descent step: θ + μ * X' * (y - Xθ)
        # We use X' * err_vec directly since err_vec = y - Xθ
        theta_tmp = theta + model.μ * (X' * err_vec)

        # Choose thresholding method (switching to hard_thresholding per your Python logic)
        theta = hard_thresholding(model, theta_tmp)

        # Update error
        err_vec = y - X * theta
    end

    model.theta = theta
    return model
end

# --- Main Simulation ---

function main()
    L = 20        # Dimension
    k0 = 3        # Sparsity
    rng = MersenneTwister(0)

    # Generate sparse ground truth w
    w_true = zeros(L)
    support = sample(rng, 1:L, k0, replace=false)
    w_true[support] = randn(rng, k0)

    N_max = 30
    start_N = 1
    l2_errors = Float64[]

    # Initialize IST
    # Note: mu (step size) must be < 2/||X'X|| for stability
    ist = IST(err_tol=0.001, μ=0.03, c=7.0, k0=k0)

    for N in start_N:N_max-1
        X = randn(rng, N, L)
        y = X * w_true

        estimate!(ist, X, y)
        push!(l2_errors, norm(w_true - ist.theta))
    end

    # Visualization
    plot(start_N:N_max-1, l2_errors,
         marker=:circle,
         lw=2,
         title="Performance of the IST Algorithm",
         xlabel="# of samples",
         ylabel="l2-norm error",
         label="Recovery Error")
end

main()
