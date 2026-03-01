using LinearAlgebra
using Statistics
using Plots
using DSP  # For filtering

# --- 1. DCT Matrix ---
function dct_matrix(L)
    T = zeros(Float32, L, L)
    for k in 0:L-1
        for n in 0:L-1
            T[k+1, n+1] = cos(π * k * (2n + 1) / (2L))
        end
        if k == 0
            T[k+1, :] .*= sqrt(1/L)
        else
            T[k+1, :] .*= sqrt(2/L)
        end
    end
    return T
end

# --- 2. TDLMS Simulation ---
function run_tdlms_simulation()
    # Parameters
    L = 10
    N = 1500
    IterNo = 100
    noisevar = 0.01f0
    correlcoeff = 0.85f0
    delta = 0.01f0
    beta = 0.5f0
    μ = 0.01f0

    MSE_lms = zeros(Float32, N, IterNo)
    MSE_tdlms = zeros(Float32, N, IterNo)
    T_mat = dct_matrix(L)

    # Filter coefficients for AR(1): H(z) = 1 / (1 + 0.85 z^-1)
    b = [1.0f0]                  # numerator
    a = [1.0f0, correlcoeff]     # denominator

    println("Running TDLMS Simulation...")

    for It in 1:IterNo
        # Correlated input
        raw_noise = randn(Float32, N + L - 1)
        xcorrel = filt(b, a, raw_noise)      # filtered AR(1) input
        xcorrel ./= std(xcorrel)

        # Input matrix X (L x N)
        X = zeros(Float32, L, N)
        for i in 1:N
            X[:, i] = xcorrel[i .+ (L-1:-1:0)]
        end

        # Unknown system
        theta = randn(Float32, L)
        y = X' * theta .+ randn(Float32, N) .* sqrt(noisevar)

        # --- Standard LMS ---
        w_lms = zeros(Float32, L)
        for i in 1:N
            xi = X[:, i]
            e = y[i] - dot(w_lms, xi)
            w_lms += μ * e * xi
            MSE_lms[i, It] = e^2
        end

        # --- Transform Domain LMS (TDLMS) ---
        w_tdlms = zeros(Float32, L)
        sigmasquare = delta .* ones(Float32, L)
        for i in 1:N
            xi = X[:, i]
            intran = T_mat * xi
            e = y[i] - dot(w_tdlms, intran)
            sigmasquare = beta .* sigmasquare .+ (1.0f0 - beta) .* abs.(intran).^2
            Dinv = 1.0f0 ./ sigmasquare
            w_tdlms += μ * e .* (Dinv .* intran)
            MSE_tdlms[i, It] = e^2
        end
    end

    # --- Average and plot ---
    avg_lms = 10 .* log10.(mean(MSE_lms, dims=2))
    avg_tdlms = 10 .* log10.(mean(MSE_tdlms, dims=2))

    p = plot(avg_lms, label="Standard LMS", color=:red, lw=1.5)
    plot!(avg_tdlms, label="TDLMS (DCT)", color=:green, lw=1.5)
    title!("LMS vs TDLMS: Convergence Comparison")
    xlabel!("Iterations")
    ylabel!("MSE (dB)")
    display(p)
end

# --- Run simulation ---
run_tdlms_simulation()

