using LinearAlgebra, Random, Plots
using StatsBase: sample # Specific import to resolve your previous error

# 1. Define the CSMP structure
mutable struct CSMP
    k::Int
    t::Int
    err_tol::Float64
    max_iter::Int
    theta::Vector{Float64}

    CSMP(; k=1, t=1, err_tol=0.001, max_iter=100) =
        new(k, t, err_tol, max_iter, Float64[])
end

# 2. The Estimation Logic (Optimized with @views)
function estimate!(model::CSMP, X::Matrix{Float64}, y::Vector{Float64})
    N, L = size(X)
    theta = zeros(L)
    error_vec = copy(y)
    S = Int[]
    ii = 0

    while norm(error_vec) > model.err_tol && ii < model.max_iter
        # Correlation Step
        corrs = abs.(X' * error_vec) ./ [norm(X[:, j]) for j in 1:L]

        # Select 't' best candidates
        new_indices = partialsortperm(corrs, 1:model.t, rev=true)
        S = union(S, new_indices)

        # Least Squares on Support
        # @views ensures X[:, S] doesn't allocate new memory
        @views begin
            X_active = X[:, S]
            theta_tilde = X_active \ y

            # Prune to 'k' sparse elements
            k_idx = partialsortperm(abs.(theta_tilde), 1:min(model.k, length(S)), rev=true)

            # Update global theta
            theta .= 0.0
            actual_indices = S[k_idx]
            theta[actual_indices] = theta_tilde[k_idx]
        end

        error_vec = y - X * theta
        ii += 1
    end

    model.theta = theta
    return model
end

# 3. Execution and Plotting
function run_simulation()
    # Parameters
    L, k0 = 100, 5 # Larger dimension for more interesting results
    rng = MersenneTwister(123)

    # Generate True Sparse Signal
    w_true = zeros(L)
    true_support = sample(rng, 1:L, k0, replace=false)
    w_true[true_support] = randn(rng, k0)

    # Sweep through number of samples N
    N_range = (2*k0):2:60
    l2_errors = Float64[]

    model = CSMP(k=k0, t=k0, err_tol=1e-4)

    for N in N_range
        X = randn(rng, N, L)
        y = X * w_true

        estimate!(model, X, y)
        push!(l2_errors, norm(w_true - model.theta))
    end

    # Visualization
    p = plot(N_range, l2_errors,
        yaxis=:log, # Log scale helps see the "phase transition"
        marker=:circle,
        lw=2,
        color=:crimson,
        label="Reconstruction Error",
        title="CSMP Phase Transition",
        xlabel="Number of Samples (N)",
        ylabel="L2 Error (Log Scale)",
        grid=:both,
        minorgrid=true
    )

    # Add a vertical line for the theoretical "k log(L/k)" threshold
    vline!([k0 * log(L/k0)], label="Theoretical Bound", ls=:dash, color=:black)

    return p
end

# Run it!
p = run_simulation()
display(p)


