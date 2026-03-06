using LinearAlgebra, Random, Statistics, Plots

# --- 1. Helper: Concrete Metropolis Matrix ---
function create_metropolis_matrix(nodes, connectivity_prob=0.4)
    # 1. Create Adjacency
    adj = rand(nodes, nodes) .< connectivity_prob
    adj = (adj .| adj') .| Matrix(I, nodes, nodes) # Ensure symmetric & self-connected

    # 2. Calculate degrees
    degrees = [sum(adj[:, i]) - 1 for i in 1:nodes]

        # 3. Apply Metropolis Rule
        C = zeros(nodes, nodes)
        for i in 1:nodes, j in 1:nodes
            if i != j && adj[i, j]
                C[i, j] = 1.0 / (max(degrees[i], degrees[j]) + 1.0)
            end
        end

        # 4. Set self-weights (Diagonal)
        for i in 1:nodes
            C[i, i] = 1.0 - sum(C[i, :])
        end
        return C
    end

    # --- 2. Adaptive Algorithms Functions ---

    function lms_step(w, x, y, μ)
        return w + μ * (y - dot(w, x)) * x
    end

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

    function apsm_step(w, X_window, y_window, ϵ, q)
        sum_p = zeros(length(w))
        sum_sq_dist = 0.0
        for j in 1:length(y_window)
            p = project_hyperslab(y_window[j], X_window[:, j], w, ϵ)
            sum_p .+= p / q
            sum_sq_dist += norm(p - w)^2 / q
        end
        diff = sum_p - w
        # Extrapolation parameter Mn
        Mn = norm(diff) < 1e-10 ? 1.0 : sum_sq_dist / norm(diff)^2
        return w + 0.2 * Mn * diff
    end

    # --- 3. Main Simulation Call ---

    function run_distributed_simulation()
        # Parameters
        N, L, nodes = 1500, 30, 10
        total_rep = 5
        changepoint = N ÷ 2
        μ_lms, ϵ, q = 0.02, 0.1, 15

        # IMPORTANT: Ensure C is a concrete Matrix, not UniformScaling
        C = create_metropolis_matrix(nodes)

        total_msd = [zeros(N) for _ in 1:4] # 1:ATC, 2:CTA, 3:Non-Coop, 4:APSM

            println("Running Ensemble Averages...")

            for rep in 1:total_rep
                Random.seed!(rep)
                θ1, θ2 = randn(L), randn(L)
                θ_true(i) = i <= changepoint ? θ1 : θ2

                node_X = [randn(L, N) for _ in 1:nodes]
                    node_y = [(node_X[n]' * θ_true(1)) .+ randn(N)*0.05 for n in 1:nodes]

                        # W[algorithm][Dimension, Node]
                        W = [zeros(L, nodes) for _ in 1:4]

                            for i in 1:N
                                curr_θ = θ_true(i)

                                # --- 1. ATC LMS (Adapt Then Combine) ---
                                W_adapted = copy(W[1])
                                for n in 1:nodes
                                    W_adapted[:, n] = lms_step(W[1][:, n], node_X[n][:, i], node_y[n][i], μ_lms)
                                end
                                W[1] = W_adapted * C' # Combine after adaptation

# --- 2. CTA LMS (Combine Then Adapt) ---
W_combined = W[2] * C'
    for n in 1:nodes
        W[2][:, n] = lms_step(W_combined[:, n], node_X[n][:, i], node_y[n][i], μ_lms)
    end

    # --- 3. Non-Cooperative LMS ---
    for n in 1:nodes
        W[3][:, n] = lms_step(W[3][:, n], node_X[n][:, i], node_y[n][i], μ_lms)
    end

    # --- 4. CTA APSM ---
    W_comb_apsm = W[4] * C'
        if i > q
            for n in 1:nodes
                win = i-q+1:i
                W[4][:, n] = apsm_step(W_comb_apsm[:, n], node_X[n][:, win], node_y[n][win], ϵ, q)
            end
        end

        # Calculate average MSD across nodes
        for alg in 1:4
            avg_msd = mean([norm(curr_θ - W[alg][:, n])^2 for n in 1:nodes])
                total_msd[alg][i] += 10log10(avg_msd + 1e-12) / total_rep
            end
        end
                            end

                            # Plotting
                            plt = plot(total_msd[1], label="ATC LMS", color=:red, lw=1.5)
                            plot!(total_msd[2], label="CTA LMS", color=:green, lw=1.5)
                            plot!(total_msd[3], label="Non-Cooperative", color=:black, ls=:dash)
                            plot!(total_msd[4], label="CTA APSM", color=:magenta, lw=2)

                            title!("Distributed Network Performance (MSD)")
                            xlabel!("Iterations"); ylabel!("MSD (dB)")
                            return plt
                        end

                        # CALL THE MAIN FUNCTION
                        plt = run_distributed_simulation()
                        display(plt)


