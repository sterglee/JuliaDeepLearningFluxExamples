using Random, Distributions, Plots

# Set seed for reproducibility
Random.seed!(0)

# 1. Generate Data
D = 2
Ni = rand(1:30, 5) # Number of points per cluster
N = sum(Ni)

# Cluster parameters
mu = [-12.5 -4.0 2.0 10.0 3.0;
      2.5 -0.1 -3.5 8.0 3.0]

Sigmas = [
    [1.4 0.81; 0.81 1.3],
    [1.5 0.2; 0.2 2.1],
    [1.6 1.0; 1.0 2.9],
    [0.5 0.22; 0.22 0.8],
    [1.5 1.4; 1.4 2.4]
    ]

# Generate multivariate normal samples
# Note: Julia's MvNormal uses Covariance matrix, samples are columns
clusters = [rand(MvNormal(mu[:, i], Sigmas[i]), Ni[i]) for i in 1:5]
    X = hcat(clusters...) # Combine into one 2xN matrix

    # 2. Setup Grid for Prediction
    x_range = -20:0.1:15
    y_range = -8:0.1:12
    # Create grid points (2 x N_grid)
    test_data = hcat([[x, y] for y in y_range for x in x_range]...)

        # 3. Setup Options
        # Assuming mkopts_bj was defined as a mutable struct to allow updates:
        mutable struct BJOptions
            algorithm::String
            use_kd_tree::Bool
            sis::Int
            initial_K::Int
            do_greedy::Bool
            do_split::Bool
            do_merge::Bool
            do_sort::Bool
            get_q_of_z::Int # Added based on script usage
        end

        # Helper to create options (translating mkopts_bj(10))
        opts = BJOptions("bj", false, 20, 10, false, false, false, false, 0)

        # 4. Iterative Inference and Plotting
        # We simulate the 3 iterations requested (sis = 1, 2, 5)
        for s in [1, 2, 5]
            opts.sis = s
            opts.get_q_of_z = 1

            # --- vdpgm call ---
            # results = vdpgm(X, opts)
            # results_predictive = vdpgm(X, results)
            # ------------------

            println("Running VDPGM with sis = $s...")

            # Placeholder for the predictive posterior density (ppdf)
            # In a real run, this comes from the vdpgm package outputs
            # ppdf = reshape(results_predictive.predictive_posterior, length(y_range), length(x_range))

            # Visualization using Plots.jl
            # p = contourf(x_range, y_range, ppdf, levels=20, color=:viridis, frame=:box)
            # scatter!(X[1,:], X[2,:], color=:red, markersize=2, label="Data")
            # display(p)
        end

