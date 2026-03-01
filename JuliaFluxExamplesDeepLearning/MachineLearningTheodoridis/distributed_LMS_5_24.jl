using LinearAlgebra, Statistics, Plots, Random

# --- Data Structures ---
struct LMSParams
    μ::Float32
    L::Int
    N::Int
    nodes::Int
    C::Matrix{Float32} # Combination Matrix
end

# --- Algorithm Implementations ---

# ATC: Adapt-then-Combine
function lms_distrib_atc(X, y_noise, p::LMSParams, h1, h2, changepoint)
    nodes, L, N = p.nodes, p.L, p.N
    # Weights for each node
    W = [zeros(Float32, L) for _ in 1:nodes]
        ψ = [zeros(Float32, L) for _ in 1:nodes] # Intermediate estimates
            msd_track = zeros(Float32, N)

            for i in 1:N
                # 1. Adaptation Step (Local)
                for k in 1:nodes
                    xi = X[k][:, i]
                    pred = dot(W[k], xi)
                    err = y_noise[k][i] - pred
                    ψ[k] = W[k] + p.μ * err * xi
                end

                # 2. Combination Step (Spatial)
                for k in 1:nodes
                    W[k] .= 0f0
                    for ℓ in 1:nodes
                        if p.C[ℓ, k] > 0
                            W[k] .+= p.C[ℓ, k] * ψ[ℓ]
                        end
                    end
                end

                # Track Average MSD (Mean Square Deviation)
                current_h = i <= changepoint ? h1 : h2
                current_msd = mean([norm(W[k] - current_h)^2 for k in 1:nodes])
                    msd_track[i] = 10 * log10(max(current_msd, 1e-10))
                end
                return msd_track
            end

            # CTA: Combine-then-Adapt
            function lms_distrib_cta(X, y_noise, p::LMSParams, h1, h2, changepoint)
                nodes, L, N = p.nodes, p.L, p.N
                W = [zeros(Float32, L) for _ in 1:nodes]
                    ψ = [zeros(Float32, L) for _ in 1:nodes]
                        msd_track = zeros(Float32, N)

                        for i in 1:N
                            # 1. Combination Step
                            for k in 1:nodes
                                ψ[k] .= 0f0
                                for ℓ in 1:nodes
                                    if p.C[ℓ, k] > 0
                                        ψ[k] .+= p.C[ℓ, k] * W[ℓ]
                                    end
                                end
                            end

                            # 2. Adaptation Step
                            for k in 1:nodes
                                xi = X[k][:, i]
                                err = y_noise[k][i] - dot(ψ[k], xi)
                                W[k] = ψ[k] + p.μ * err * xi
                            end

                            current_h = i <= changepoint ? h1 : h2
                            current_msd = mean([norm(W[k] - current_h)^2 for k in 1:nodes])
                                msd_track[i] = 10 * log10(max(current_msd, 1e-10))
                            end
                            return msd_track
                        end

                        # --- Network Generation (Metropolis Rule) ---
                        function get_metropolis_matrix(nodes, connections)
                            # Simplified connected graph for verification
                            Adj = zeros(nodes, nodes)
                            for i in 1:nodes, j in i+1:nodes
                                if connections > 0
                                    Adj[i,j] = Adj[j,i] = 1.0
                                    connections -= 1
                                end
                            end
                            for i in 1:nodes Adj[i,i] = 1.0 end

                            C = zeros(nodes, nodes)
                            degrees = sum(Adj, dims=1)
                            for i in 1:nodes, j in 1:nodes
                                if i != j && Adj[i,j] > 0
                                    C[i,j] = 1.0 / max(degrees[i], degrees[j])
                                end
                            end
                            for i in 1:nodes
                                C[i,i] = 1.0 - sum(C[i, [1:i-1; i+1:nodes]])
                            end
                            return Float32.(C)
                        end

                        # --- Main Simulation ---
                        function run_distributed_test()
                            L, N, nodes = 30, 2000, 10
                            totalrep = 50
                            changepoint = N ÷ 2

                            # Metropolis Combination Matrix
                            C = get_metropolis_matrix(nodes, 32)

                            total_atc = zeros(N)
                            total_cta = zeros(N)
                            total_noncoop = zeros(N)

                            println("Starting Monte Carlo loops...")

                            for rep in 1:totalrep
                                Random.seed!(rep)
                                h1 = randn(Float32, L)
                                h2 = h1 # No change in this version, can be changed to randn(L)

                                # Generate data for each node
                                X = [randn(Float32, L, N) for _ in 1:nodes]
                                    y_noise = Vector{Vector{Float32}}(undef, nodes)

                                    for k in 1:nodes
                                        # Signal + Noise
                                        y_clean = X[k]' * (1 <= changepoint ? h1 : h2)
                                        y_noise[k] = y_clean + randn(Float32, N) * 0.1f0
                                    end

                                    params = LMSParams(0.01f0, L, N, nodes, C)
                                    noncoop_params = LMSParams(0.01f0, L, N, nodes, Matrix{Float32}(I, nodes, nodes))

                                    total_atc .+= lms_distrib_atc(X, y_noise, params, h1, h2, changepoint)
                                    total_cta .+= lms_distrib_cta(X, y_noise, params, h1, h2, changepoint)
                                    total_noncoop .+= lms_distrib_atc(X, y_noise, noncoop_params, h1, h2, changepoint)
                                end

                                # Plot results
                                plot(total_atc ./ totalrep, label="ATC (Cooperative)", color=:red, lw=1.5)
                                plot!(total_cta ./ totalrep, label="CTA (Cooperative)", color=:green, lw=1.5)
                                plot!(total_noncoop ./ totalrep, label="Non-Cooperative", color=:black, ls=:dash)
                                title!("Distributed LMS Performance")
                                xlabel!("Iterations")
                                ylabel!("MSD (dB)")
                            end

                            run_distributed_test()

