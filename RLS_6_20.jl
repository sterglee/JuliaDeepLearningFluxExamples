using LinearAlgebra
using Statistics
using Plots
using Random

"""
run_comparison()

Implements the comparison between APA, RLS, and NLMS for a system
    identification task with correlated inputs.
    """
    function run_comparison()
        # --- 1. Parameters ---
        L = 200             # Dimension of unknown vector
        N = 3500            # Number of Data points
        IterNo = 100        # Monte Carlo iterations
        noisevar = 0.01f0   # Noise variance (Float32 for performance)

        # Pre-allocate MSE matrices (N x Iterations)
        MSE_APA  = zeros(Float32, N, IterNo)
        MSE_RLS  = zeros(Float32, N, IterNo)
        MSE_NLMS = zeros(Float32, N, IterNo)

        # Unknown parameter to identify
        θ_true = randn(Float32, L)

        println("Starting Simulation: $IterNo iterations...")

        for it in 1:IterNo
            # Progress log every 10 iterations
            it % 10 == 0 && println("Iteration: $it")

            # --- 2. Data Generation ---
            # Generate correlated input via sliding window (mimics MATLAB convmtx)
            xcorrel = randn(Float32, N + L - 1)
            xcorrel ./= std(xcorrel)

            # X is an L x N matrix where each column is an input vector
            X = zeros(Float32, L, N)
            for i in 1:N
                X[:, i] = xcorrel[i .+ (L-1:-1:0)]
            end

            noise = randn(Float32, N) .* sqrt(noisevar)
            y = (X' * θ_true) .+ noise

            # --- 3. APA (Affine Projection Algorithm) ---
            w_apa = zeros(Float32, L)
            μ_apa = 0.2f0
            δ_apa = 0.001f0
            q = 30 # Projection order

            for i in (q + 1):N
                # Window indices
                idx = i:-1:(i - q + 1)
                Xq = X[:, idx]'    # q x L matrix
                yq = y[idx]        # q x 1 vector

                # Error vector
                e_vec = yq - Xq * w_apa

                # Update using stable backslash instead of inv()
                # w = w + μ * Xq' * (δI + XqXq')⁻¹ * e
                w_apa += μ_apa * Xq' * ((δ_apa * I + Xq * Xq') \ e_vec)

                # Instantaneous error
                e_inst = y[i] - dot(w_apa, X[:, i])
                MSE_APA[i, it] = e_inst^2
            end

            # --- 4. RLS (Recursive Least Squares) ---
            w_rls = zeros(Float32, L)
            δ_rls = 0.001f0
            P = (1.0f0 / δ_rls) * Matrix{Float32}(I, L, L)

            for i in 1:N
                xi = X[:, i]
                # Gain calculation
                Px = P * xi
                γ = 1.0f0 / (1.0f0 + dot(xi, Px))
                g = Px * γ

                e = y[i] - dot(w_rls, xi)
                w_rls += g * e

                # Rank-1 P update (Sherman-Morrison)
                # P = P - (g * g') / γ  == P - (P*x*x'*P)/(1 + x'*P*x)
                P -= (g * (xi' * P))

                MSE_RLS[i, it] = e^2
            end

            # --- 5. NLMS (Normalized LMS) ---
            w_nlms = zeros(Float32, L)
            δ_nlms = 0.001f0
            μ_nlms = 1.2f0

            for i in 1:N
                xi = X[:, i]
                e = y[i] - dot(w_nlms, xi)

                # Step size normalized by signal power
                μ_eff = μ_nlms / (δ_nlms + dot(xi, xi))
                w_nlms += μ_eff * e * xi

                MSE_3_inst = e^2
                MSE_NLMS[i, it] = MSE_3_inst
            end
        end

        # --- 6. Averaging & Visualization ---
        # Average across Monte Carlo trials
        avg_apa  = 10log10.(mean(MSE_APA, dims=2))
        avg_rls  = 10log10.(mean(MSE_RLS, dims=2))
        avg_nlms = 10log10.(mean(MSE_NLMS, dims=2))

        p = plot(avg_apa, label="APA (q=30)", color=:red, lw=1.5)
        plot!(avg_rls, label="RLS", color=:green, lw=1.5)
        plot!(avg_nlms, label="NLMS", color=:blue, lw=1.5)

        title!("LMS/RLS/APA Convergence Comparison")
        xlabel!("Iterations (N)")
        ylabel!("Mean Square Error (dB)")
        grid!(true)

        display(p)
    end

    run_comparison()

