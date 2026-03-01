using LinearAlgebra, Statistics, Random, Plots

# --- 1. THE FUNCTION DEFINITIONS (Must come first) ---

"""
project_hyperslab(y, x, w, ϵ)
Geometric projection of vector w onto a hyperslab defined by input x and observation y.
"""
function project_hyperslab(y, x, w, ϵ)
    inner = dot(w, x)
    err = y - inner
    norm_sq = dot(x, x)

    if err < -ϵ
        return w + ((err + ϵ) / norm_sq) * x
        elseif err > ϵ
        return w + ((err - ϵ) / norm_sq) * x
    else
        return w
    end
end

"""
cta_apsm_distributed(input_handle, y, data, errfun)
Implements the Combine-Then-Adapt Distributed APSM algorithm for networks.
    """
    function cta_apsm_distributed(inputvec, y, data, errfun)
        L, N, nodes = data.L, data.N, data.nodes
        ϵ, q, C = data.epsilon, data.q, data.C

        # Initialization
        x = [zeros(Float64, L) for _ in 1:nodes]
            x_combined = [zeros(Float64, L) for _ in 1:nodes]
                errors = [zeros(Float64, N) for _ in 1:nodes]

                    for i in 1:N
                        # --- Stage 1: Combination (Socialize) ---
                        for inod in 1:nodes
                            fill!(x_combined[inod], 0.0)
                            for j in 1:nodes
                                if C[j, inod] != 0
                                    x_combined[inod] .+= C[j, inod] .* x[j]
                                end
                            end
                        end

                        # --- Stage 2: Adaptation (Local APSM) ---
                        for inod in 1:nodes
                            window = max(1, i - q + 1):i
                            Xq = inputvec(inod, window)
                            yv = y[inod][window]
                            sx = length(yv)

                            sum_p = zeros(Float64, L)
                            sum_sq_dist = 0.0

                            for jj in 1:sx
                                p = project_hyperslab(yv[jj], Xq[:, jj], x_combined[inod], ϵ)
                                sum_p .+= p ./ sx
                                sum_sq_dist += norm(p - x_combined[inod])^2 / sx
                            end

                            diff = sum_p - x_combined[inod]
                            Mn = norm(diff) < 1e-12 ? 1.0 : sum_sq_dist / norm(diff)^2

                            x[inod] = x_combined[inod] + 0.2 * Mn * diff
                            errors[inod][i] = errfun(i, x[inod])
                        end
                    end
                    return errors, x
                end

                # --- 2. THE MAIN CALL (Must come after definitions) ---

                function run_simulation()
                    # Setup Network (Ring Topology)
                    nodes = 10
                    C = zeros(nodes, nodes)
                    for i in 1:nodes
                        C[i, i] = 1/3
                        C[mod1(i-1, nodes), i] = 1/3
                        C[mod1(i+1, nodes), i] = 1/3
                    end

                    data = (L=50, N=2000, nodes=nodes, epsilon=0.05, q=10, C=C)

                    # Generate Synthetic Data
                    Random.seed!(42)
                    theta_true = randn(data.L)
                    node_X = [randn(data.L, data.N) for _ in 1:nodes]
                        node_y = [(node_X[n]' * theta_true) .+ randn(data.N)*0.01 for n in 1:nodes]

                            # Interface functions
                            input_h(idx, rng) = node_X[idx][:, rng]
                            err_h(iter, x_est) = norm(theta_true - x_est)^2

                            # CALL THE FUNCTION
                            errors, final_x = cta_apsm_distributed(input_h, node_y, data, err_h)

                            # Plot
                            p = plot(10log10.(errors[1]), title="Node 1 Convergence", ylabel="MSD (dB)", xlabel="Iter")
                            display(p)
                        end

                        run_simulation()

