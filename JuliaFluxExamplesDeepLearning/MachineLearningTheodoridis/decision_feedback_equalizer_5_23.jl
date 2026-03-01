using Flux
using LinearAlgebra
using Statistics
using Plots

# --- Helper to generate Synthetic Data ---
function generate_data(L, N, theta, noisevar)
    xc = randn(N + L - 1)
    xc ./= std(xc)
    # Efficiently construct input matrix (L x N)
    X = zeros(Float32, L, N)
    for i in 1:N
        X[:, i] = xc[i .+ (L-1:-1:0)]
    end
    noise = randn(Float32, N) .* sqrt(Float32(noisevar))
    y = (X' * theta) .+ noise
    return X, y
end

# --- The Simulation Function ---
function run_simulation()
    # Setup
    L, N, IterNo = 60, 3500, 100
    theta = randn(Float32, L)
    noisevar = 0.01f0

    # Storage for MSE (Time x Iteration)
    results = [zeros(Float32, N, IterNo) for _ in 1:4]

        for it in 1:IterNo
            X, y = generate_data(L, N, theta, noisevar)

            # Initialize weights for 4 different models
            w1, w2, w3, w4 = [zeros(Float32, L) for _ in 1:4]

                # Hyperparameters
                delta = 0.001f0
                mu_apa = 0.1f0
                mu_nlms = 0.35f0
                mu_lms = 0.025f0

                for i in 1:N
                    # 1. APA (q=30)
                    if i >= 30
                        q = 30
                        idx = i:-1:i-q+1
                        Xq, yq = X[:, idx]', y[idx]
                        e_vec = yq - Xq * w1
                        # Flux-style update logic: w = w + step
                        # We use \ for the inverse to maintain numerical stability
                        w1 += mu_apa * Xq' * ((delta * I + Xq * Xq') \ e_vec)
                        results[1][i, it] = (y[i] - dot(w1, X[:, i]))^2
                    end

                    # 2. APA (q=10)
                    if i >= 10
                        q = 10
                        idx = i:-1:i-q+1
                        Xq, yq = X[:, idx]', y[idx]
                        e_vec = yq - Xq * w2
                        w2 += mu_apa * Xq' * ((delta * I + Xq * Xq') \ e_vec)
                        results[2][i, it] = (y[i] - dot(w2, X[:, i]))^2
                    end

                    # 3. NLMS
                    x_i = X[:, i]
                    e_nlms = y[i] - dot(w3, x_i)
                    norm_sq = dot(x_i, x_i)
                    w3 += (mu_nlms / (delta + norm_sq)) * e_nlms * x_i
                    results[3][i, it] = e_nlms^2

                    # 4. LMS
                    e_lms = y[i] - dot(w4, x_i)
                    w4 += mu_lms * e_lms * x_i
                    results[4][i, it] = e_lms^2
                end
            end

            # Average and Plot
            means = [10log10.(mean(r, dims=2)) for r in results]

                p = plot(title="Adaptive Algorithm Convergence", ylabel="MSE (dB)", xlabel="Iterations")
                plot!(means[1], label="APA (q=30)", color=:red)
                plot!(means[2], label="APA (q=10)", color=:green)
                plot!(means[3], label="NLMS", color=:black)
                plot!(means[4], label="LMS", color=:blue)
                display(p)
            end

            run_simulation()

