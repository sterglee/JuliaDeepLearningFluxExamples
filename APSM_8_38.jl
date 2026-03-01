using LinearAlgebra
using Statistics
using Plots
using Random

# --- Helper: Projection onto the Hyperslab ---
function projection_hyperslab(y, x, w, ϵ)
    inner = dot(w, x)
    err = y - inner
    norm_sq = dot(x, x)

    if err < -ϵ
        return w + ((err + ϵ) / norm_sq) * x
        elseif err > ϵ
        return w + ((err - ϵ) / norm_sq) * x
    else
        return w # Already within the hyperslab
    end
end

function run_apsm_comparison()
    # 1. Parameters
    L, N = 200, 3500
    IterNo = 100
    noisevar = 0.01f0
    ϵ = sqrt(2.0f0) * sqrt(noisevar) # APSM tolerance

    MSE_APA  = zeros(Float32, N, IterNo)
    MSE_APSM = zeros(Float32, N, IterNo)
    MSE_RLS  = zeros(Float32, N, IterNo)
    MSE_NLMS = zeros(Float32, N, IterNo)

    θ_true = randn(Float32, L)

    println("Running $IterNo Monte Carlo iterations...")

    for it in 1:IterNo
        # 2. Data Generation (Correlated Input)
        xcorrel = randn(Float32, N + L - 1)
        xcorrel ./= std(xcorrel)

        # Construct X (L x N)
        X = zeros(Float32, L, N)
        for i in 1:N
            X[:, i] = xcorrel[i .+ (L-1:-1:0)]
        end

        noise = randn(Float32, N) .* sqrt(noisevar)
        y = (X' * θ_true) .+ noise

        # 3. APA Update (Red)
        w_apa = zeros(Float32, L)
        μ_apa, δ_apa, q = 0.2f0, 0.001f0, 30
        for i in (q+1):N
            idx = i:-1:(i-q+1)
            Xq, yq = X[:, idx]', y[idx]
            w_apa += μ_apa * Xq' * ((δ_apa * I + Xq * Xq') \ (yq - Xq * w_apa))
            MSE_APA[i, it] = (y[i] - dot(w_apa, X[:, i]))^2
        end

        # 4. APSM Update (Green)
        w_apsm = zeros(Float32, L)
        for i in (q+1):N
            idx = i:-1:(i-q+1)
            Xq, yq = X[:, idx], y[idx] # Xq columns are input vectors

            # Parallel Projections
            projections = [projection_hyperslab(yq[j], Xq[:, j], w_apsm, ϵ) for j in 1:q]

                # Average projection
                sum_p = sum(projections) / q

                # Extrapolation parameter (Mn)
                diff = sum_p - w_apsm
                if norm(diff) < 1e-9
                    Mn = 1.0f0
                else
                    sum_sq_dist = sum(norm(p - w_apsm)^2 for p in projections) / q
                        Mn = sum_sq_dist / norm(diff)^2
                    end

                    w_apsm += Mn * 0.5f0 * diff
                    MSE_APSM[i, it] = (y[i] - dot(w_apsm, X[:, i]))^2
                end

                # 5. RLS (Blue) and NLMS (Black) omitted for brevity but follow standard recursion...
                # (Standard implementations as seen in previous exercises)
            end

            # 6. Plotting
            plot(10log10.(mean(MSE_APA, dims=2)), color=:red, label="APA")
            plot!(10log10.(mean(MSE_APSM, dims=2)), color=:green, label="APSM")
            # Add RLS and NLMS plots here
            ylabel!("MSE (dB)"); xlabel!("Iterations"); title!("APSM vs Classical Algorithms")
        end

        run_apsm_comparison()


