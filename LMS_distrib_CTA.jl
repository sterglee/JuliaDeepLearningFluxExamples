using LinearAlgebra, Random, Statistics, Plots

# --- 1. Define the CTA-LMS Algorithm ---
function cta_lms_distributed(inputvec, y, data, errfun)
    L, N, nodes = data.L, data.N, data.nodes
    μ, C = data.mu, data.C

    x = [zeros(Float64, L) for _ in 1:nodes]
        x_combined = [zeros(Float64, L) for _ in 1:nodes]
            errors = [zeros(Float64, N) for _ in 1:nodes]

                for i in 1:N
                    # --- Stage 1: COMBINE (Consensus) ---
                    for n in 1:nodes
                        fill!(x_combined[n], 0.0)
                        for j in 1:nodes
                            if C[j, n] != 0
                                x_combined[n] .+= C[j, n] .* x[j]
                            end
                        end
                    end

                    # --- Stage 2: ADAPT (Gradient Update) ---
                    for n in 1:nodes
                        a_n = inputvec(n, i)
                        # Error is calculated relative to the COMBINED estimate
                        er = y[n][i] - dot(a_n, x_combined[n])
                        x[n] = x_combined[n] + μ * er * a_n

                        # Log Mean Square Deviation (MSD)
                        errors[n][i] = errfun(i, x[n])
                    end
                end
                return errors, x
            end

            # --- 2. Main Simulation Call ---
            function main_cta_call()
                # Network Parameters
                L, N, nodes = 50, 2500, 10
                μ = 0.01

                # Generate a Metropolis-Hastings Combination Matrix (C)
                # This creates a doubly-stochastic matrix for a random graph
                Random.seed!(42)
                adj = rand(nodes, nodes) .< 0.4
                adj = (adj .| adj') .| Matrix(I, nodes, nodes)
                degrees = [sum(adj[:, i]) - 1 for i in 1:nodes]
                    C = zeros(nodes, nodes)
                    for i in 1:nodes, j in 1:nodes
                        if i != j && adj[i, j]
                            C[i, j] = 1.0 / (max(degrees[i], degrees[j]) + 1.0)
                        end
                    end
                    for i in 1:nodes; C[i, i] = 1.0 - sum(C[i, :]); end

                    # True System to Identify
                    θ_true = randn(L)

                    # Synthetic Data Generation
                    node_X = [randn(L, N) for _ in 1:nodes]
                        node_y = [(node_X[n]' * θ_true) .+ randn(N)*0.01 for n in 1:nodes]

                            # Interface Functions
                            input_h(idx, iter) = node_X[idx][:, iter]
                            err_h(iter, x_est) = 10log10(norm(θ_true - x_est)^2 + 1e-12)

                            # Call Algorithm
                            data_struct = (L=L, N=N, nodes=nodes, mu=μ, C=C)
                            println("Simulating CTA-LMS Network...")
                            errors, final_weights = cta_lms_distributed(input_h, node_y, data_struct, err_h)

                            # Plot Average Performance across all nodes
                            avg_error = mean(hcat(errors...), dims=2)
                            plt = plot(avg_error, title="Global Network Convergence (CTA-LMS)",
                                       xlabel="Iterations", ylabel="Network MSD (dB)", lw=2, color=:green)
                            display(plt)
                        end

                        main_cta_call()

