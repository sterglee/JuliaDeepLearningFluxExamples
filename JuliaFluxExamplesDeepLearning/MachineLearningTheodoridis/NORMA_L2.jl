using LinearAlgebra, Random, Plots, Statistics

# ------------------------------
# Kernel
# ------------------------------
abstract type Kernel end
struct Gaussian <: Kernel
    sigma::Float64
end

function kappa(x, y, k::Gaussian)
    return exp(-norm(x - y)^2 / (2*k.sigma^2))
end

# ------------------------------
# Loss Functions
# ------------------------------
abstract type Loss end
struct L2Loss <: Loss end

# Compute subgradient coefficient
function compute_subgrad_coef(error, epsilon, ::L2Loss)
    # Simple L2: subgradient = error
    return error
end

# ------------------------------
# APSM Function
# ------------------------------
function QKernel_APSM(z, d, epsilon, Q, loss_obj, kern, sparse_params)
    N1, L = size(z)
    Delta = Int(sparse_params[1])
    q_threshold = sparse_params[2]

    centers = [z[1, :]]
    a = [0.0]
    errors = zeros(N1)
    expansion_size = zeros(Int, N1)
    w = fill(1/Q, Q)

    for n in 1:N1
        x_n = z[n, :]
        f_n = sum(a[i]*kappa(centers[i], x_n, kern) for i in 1:length(a))
            errors[n] = d[n] - f_n

            g_coef = compute_subgrad_coef(errors[n], epsilon, loss_obj)
            if abs(g_coef) > 0
                norm_sq = kappa(x_n, x_n, kern)
                step = (errors[n]*w[1]) / (norm_sq + 1e-6)

                # Quantization / sparsification
                dists = [norm(x_n - c) for c in centers]
                    min_dist, min_idx = findmin(dists)

                    if min_dist <= q_threshold
                        a[min_idx] += step
                        elseif length(a) < Delta
                        push!(centers, x_n)
                        push!(a, step)
                    end
                end
                expansion_size[n] = length(a)
            end

            return a, centers, errors, expansion_size
        end

        # ------------------------------
        # Example Usage
        # ------------------------------
        Random.seed!(123)
        N = 200
        z = randn(N,2)
        d = sin.(z[:,1]) + 0.1*randn(N)

        epsilon_apsm = 1e-5
        Q_subspace = 5
        sigma_kern = 1.0
        Delta_max = 1000
        q_threshold = 0.2

        kernel_obj = Gaussian(sigma_kern)
        loss_obj = L2Loss()

        a_apsm, centers_apsm, e_apsm, exp_size_apsm = QKernel_APSM(
            z, d, epsilon_apsm, Q_subspace, loss_obj, kernel_obj, [Delta_max, q_threshold]
            )

        mse_apsm = e_apsm .^ 2
        smoothed_mse_db = 10*log10.(filter(x->!isnan(x), [mean(mse_apsm[max(1,i-9):i]) for i in 1:length(mse_apsm)]))

            println("Final Dictionary Size: ", exp_size_apsm[end])
            println("Final Error (dB): ", round(smoothed_mse_db[end], digits=2))

            # Plot
            plot(smoothed_mse_db, label="QAPSM", color=:blue)
            xlabel!("Iteration")
            ylabel!("MSE (dB)")
            title!("QKernel APSM Performance")

