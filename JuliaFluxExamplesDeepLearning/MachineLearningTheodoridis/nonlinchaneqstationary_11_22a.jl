using LinearAlgebra, Random, Statistics, Plots

# ============================================================
# 1. Infrastructure (Kernels & Support)
# ============================================================
abstract type Kernel end
struct Gaussian <: Kernel sigma::Float64 end
kappa(x, y, k::Gaussian) = exp(-norm(x - y)^2 / k.sigma^2)

# Helper: Sliding window mean (replaces sum_mean_val2)
function smooth(v, K)
    N = length(v)
    res = zeros(N)
    for i in 1:N
        win = max(1, i-K+1):i
        res[i] = mean(v[win])
    end
    return res
end

# ============================================================
# 2. Adaptive Filter: QKNLMS (Quantized Kernel NLMS)
# ============================================================
function QKernel_NLMS(z, d, mu, kern, sparse_params)
    N1, L = size(z)
    epsilon_q = sparse_params[1] # Quantization threshold

    centers = [z[1, :]]
    a = [mu * d[1]]
    errors = zeros(N1)
    expansion_sizes = zeros(N1)

    for n in 2:N1
        x_n = z[n, :]
        # Compute current prediction: f(x_n) = sum(a_i * kappa(center_i, x_n))
        f_n = sum(a[i] * kappa(centers[i], x_n, kern) for i in 1:length(a))

            errors[n] = d[n] - f_n

            # Quantization Logic
            dists = [norm(x_n - c) for c in centers]
                min_dist, min_idx = findmin(dists)

                if min_dist <= epsilon_q
                    # Update existing closest center
                    a[min_idx] += mu * errors[n]
                else
                    # Add new center
                    push!(centers, x_n)
                    push!(a, mu * errors[n])
                end
                expansion_sizes[n] = length(a)
            end
            return a, centers, errors, expansion_sizes
        end

        # ============================================================
        # 3. Main Simulation Call
        # ============================================================
        function main_stationary_simulation()
            Random.seed!(0)
            N, NUMBER_OF_TESTS = 5000, 50 # Reduced tests for quick execution
            p, L, D = 5, 5, 2
            h = [-0.9, 0.6, -0.7, 0.2, 0.1]

            N1 = N - L + 1
            total_mse_knlms = zeros(N1)
            total_exp_knlms = zeros(N1)

            println("--- Starting Simulation ---")

            for t_idx in 1:NUMBER_OF_TESTS
                print("\rProgress: $(round(t_idx/NUMBER_OF_TESTS*100, digits=1))%")

                # Signal Generation
                s = 0.8 .* randn(N)
                noise = sqrt(0.8^2 / (10^(15/10))) .* randn(N)

                # Channel
                r = zeros(N)
                for n in p+1:N
                    tn = dot(h, s[n:-1:n-p+1]) + noise[n]
                    r[n] = tn + 0.15*tn^2 + 0.03*tn^3 + noise[n]
                end

                # Data Formatting
                z, d = zeros(N1, L), zeros(N1)
                for n in (L-D):(N-D)
                    idx = n - (L-D) + 1
                    z[idx, :] = r[(n + D - L + 1):(n + D)]
                    d[idx] = s[n]
                end

                # Run QKNLMS
                _, _, e, ex = QKernel_NLMS(z, d, 0.5, Gaussian(5.0), [0.1])

                total_mse_knlms .+= e .^ 2
                total_exp_knlms .+= ex
            end

            # Processing & Plotting
            mse_db = 10log10.(smooth(total_mse_knlms ./ NUMBER_OF_TESTS, 10))

            p1 = plot(mse_db, title="Stationary Channel MSE", ylabel="MSE (dB)", label="QKNLMS", color=:red)
            p2 = plot(total_exp_knlms ./ NUMBER_OF_TESTS, title="Expansion Size", ylabel="M", label="Size", color=:blue)

            println("\nSimulation Finished.")
            display(plot(p1, p2, layout=(2,1)))
        end

        # EXECUTE MAIN
        main_stationary_simulation()


