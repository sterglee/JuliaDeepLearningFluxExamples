using LinearAlgebra
using Statistics
using Plots

function run_robbins_monro()
    # --- 1. Parameters ---
    L = 2               # Dimension of unknown vector
    N = 500             # Number of data points
    IterNo = 1000       # Monte Carlo iterations
    noisevar = 0.1f0    # Noise variance

    # Pre-allocate weight tracking (Time x Iterations)
    # We track the first component of w as per the original script
    wtot = zeros(Float32, N, IterNo)

    # "True" system parameter (fixed across iterations for comparison)
    theta = randn(Float32, L)

    println("Running $IterNo Monte Carlo iterations...")

    for it in 1:IterNo
        # Generate data for this realization
        X = randn(Float32, L, N)
        noise = randn(Float32, N) .* sqrt(noisevar)
        y = (X' * theta) .+ noise

        # Initialize weight vector (Flux-style parameter)
        w = zeros(Float32, L)

        for i in 1:N
            # 2. Robbins-Monro Update
            # Step size mu = 1/i satisfies the stochastic approximation conditions:
            # sum(mu) = inf AND sum(mu^2) < inf
            mu = 1.0f0 / i

            xi = X[:, i]
            prediction = dot(w, xi)
            error = y[i] - prediction

            # Stochastic Gradient Descent update
            w += mu * error * xi

            # Record the first component
            wtot[i, it] = w[1]
        end
    end

    # --- 3. Statistics & Plotting ---
    mean_w = mean(wtot, dims=2)[:] # Average across iterations
    std_w = std(wtot, dims=2)[:]   # Standard deviation across iterations

    # Baseline for the true parameter
    p = plot(ones(N) .* theta[1], color=:red, label="True Theta[1]", lw=2)

    # Plot the mean estimate
    plot!(mean_w, color=:black, label="Mean Estimate (w[1])", alpha=0.8)

    # Add error bars every 10 samples (starting after 30)
    indices = 40:10:N
    scatter!(indices, mean_w[indices],
             yerror=std_w[indices],
             marker=:none,
             color=:blue,
             label="Standard Deviation")

    title!("Robbins-Monro Convergence Analysis")
    xlabel!("Iterations (N)")
    ylabel!("Weight Value")
    display(p)
end

run_lms_corrected = run_robbins_monro()
\
