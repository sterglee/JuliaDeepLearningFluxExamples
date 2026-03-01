using LinearAlgebra
using Statistics
using Plots
using Random

function run_rls_apa_comparison()
    # --- 1. Parameters ---
    L = 200             # Dimension of unknown vector
    N = 3500            # Number of Data points
    IterNo = 100        # Monte Carlo iterations
    noisevar = 0.01f0

    # Pre-allocate MSE matrices (Time x Iterations)
    MSE_APA  = zeros(Float32, N, IterNo)
    MSE_RLS  = zeros(Float32, N, IterNo)
    MSE_NLMS = zeros(Float32, N, IterNo)

    println("Running $IterNo Monte Carlo iterations...")

    for it in 1:IterNo
        # --- 2. Data Generation ---
        # Generate white noise input and normalize
        xcorrel = randn(Float32, N + L - 1)
        xcorrel ./= std(xcorrel)

        # Construct Input Matrix X (L x N)
        X = zeros(Float32, L, N)
        for i in 1:N
            X[:, i] = xcorrel[i .+ (L-1:-1:0)]
        end

        theta = randn(Float32, L) # Unknown system parameter
        noise = randn(Float32, N) .* sqrt(noisevar)
        y = (X' * theta) .+ noise

        # --- 3. APA Recursion ---
        w_apa = zeros(Float32, L)
        μ_apa = 0.2f0
        δ_apa = 0.001f0
        q = 30 # Projection order (window size)

        for i in q:N
            # Data window for APA
            Xq = X[:, i:-1:i-q+1]' # (q x L)
            yvec = y[i:-1:i-q+1]   # (q x 1)

            e_vec = yvec - Xq * w_apa
            # Instantaneous error for tracking
            e_inst = y[i] - dot(w_apa, X[:, i])

            # APA Update: w = w + μ * Xq' * inv(δI + XqXq') * e
            # Using \ for numerical stability instead of inv()
            w_apa += μ_apa * Xq' * ((δ_apa * I + Xq * Xq') \ e_vec)
            MSE_APA[i, it] = e_inst^2
        end

        # --- 4. RLS Recursion ---
        w_rls = zeros(Float32, L)
        δ_rls = 0.001f0
        P = (1.0f0 / δ_rls) * Matrix{Float32}(I, L, L)

        for i in 1:N
            xi = X[:, i]
            # Intermediate gain calculations
            Px = P * xi
            gamma = 1.0f0 / (1.0f0 + dot(xi, Px))
            gi = Px * gamma

            e = y[i] - dot(w_rls, xi)
            w_rls += gi * e

            # Rank-1 update of the inverse correlation matrix (Sherman-Morrison)
            P -= gi * (xi' * P)
            MSE_RLS[i, it] = e^2
        end

        # --- 5. NLMS Recursion ---
        w_nlms = zeros(Float32, L)
        δ_nlms = 0.001f0
        μ_nlms = 1.2f0

        for i in 1:N
            xi = X[:, i]
            e = y[i] - dot(w_nlms, xi)
            # Normalized step size
            μ_n = μ_nlms / (δ_nlms + dot(xi, xi))
            w_nlms += μ_n * e * xi
            MSE_NLMS[i, it] = e^2
        end
    end

    # --- 6. Results & Plotting ---
    avg_apa = 10log10.(mean(MSE_APA, dims=2))
    avg_rls = 10log10.(mean(MSE_RLS, dims=2))
    avg_nlms = 10log10.(mean(MSE_NLMS, dims=2))

    p = plot(avg_apa, label="APA (q=30)", color=:red)
    plot!(avg_rls, label="RLS", color=:green)
    plot!(avg_nlms, label="NLMS", color=:blue)

    title!("Algorithm Comparison: APA vs RLS vs NLMS")
    xlabel!("Iterations"); ylabel!("MSE (dB)")
    display(p)
end

run_rls_apa_comparison()


