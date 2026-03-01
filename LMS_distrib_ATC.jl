using LinearAlgebra, Random, Statistics, Plots

# --- 1. Define the ATC-LMS Function ---
function atc_lms_distributed(inputvec, y, data, errfun)
    L, N, nodes = data.L, data.N, data.nodes
    μ, C = data.mu, data.C

    x = [zeros(Float64, L) for _ in 1:nodes]
        x_adapted = [zeros(Float64, L) for _ in 1:nodes]
            errors = [zeros(Float64, N) for _ in 1:nodes]

                for i in 1:N
                    # Adaptation Stage
                    for n in 1:nodes
                        a_n = inputvec(n, i)
                        er = y[n][i] - dot(a_n, x[n])
                        x_adapted[n] = x[n] + μ * er * a_n
                    end

                    # Combination Stage
                    for n in 1:nodes
                        x[n] = zeros(L)
                        for j in 1:nodes
                            if C[j, n] != 0
                                x[n] .+= C[j, n] .* x_adapted[j]
                            end
                        end
                        errors[n][i] = errfun(i, x[n])
                    end
                end
                return errors, x
            end

            # --- 2. Setup and Execution ---
            function main_atc_call()
                # Parameters
                L, N, nodes = 40, 2000, 5
                μ = 0.01

                # Generate a simple Ring Topology Combination Matrix
                C = zeros(nodes, nodes)
                for i in 1:nodes
                    C[i, i] = 0.4
                    C[mod1(i-1, nodes), i] = 0.3
                    C[mod1(i+1, nodes), i] = 0.3
                end

                # Define the "True" System we want to identify
                Random.seed!(123)
                θ_true = randn(L)

                # Generate Data for each node
                node_X = [randn(L, N) for _ in 1:nodes]
                    node_y = [(node_X[n]' * θ_true) .+ randn(N)*0.01 for n in 1:nodes]

                        # Handle functions for the algorithm
                        input_h(idx, iter) = node_X[idx][:, iter]
                        err_h(iter, x_est) = 10log10(norm(θ_true - x_est)^2 + 1e-12)

                        # Data object
                        data_struct = (L=L, N=N, nodes=nodes, mu=μ, C=C)

                        # CALL THE FUNCTION
                        println("Calling ATC-LMS for $nodes nodes...")
                            errors, final_weights = atc_lms_distributed(input_h, node_y, data_struct, err_h)

                            # Plot result for Node 1
                            plt = plot(errors[1], title="ATC-LMS Convergence (Node 1)",
                                       xlabel="Iterations", ylabel="MSD (dB)", lw=2, color=:red)
                            display(plt)
                        end

                        main_atc_call()


