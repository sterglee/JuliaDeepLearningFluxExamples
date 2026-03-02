using LinearAlgebra, Random, Plots, Statistics

# -------------------------------
# 1. Gaussian Kernel
# -------------------------------
struct GaussianKernel
    sigma::Float64
end

function kappa(x::AbstractVector, y::AbstractVector, kern::GaussianKernel)
    return exp(-norm(x - y)^2 / (2*kern.sigma^2))
end

# -------------------------------
# 2. L2 Loss Subgradient
# -------------------------------
struct L2Loss end
compute_subgrad_coef(error::Float64, epsilon::Float64, ::L2Loss) = error

# -------------------------------
# 3. APSM Kernel Function
# -------------------------------
function QKernel_APSM(z::Matrix{Float64}, d::Vector{Float64}, epsilon::Float64,
                      Q::Int, loss_obj, kern::GaussianKernel, sparse_params::Vector{Float64})

    N, L = size(z)
    Delta = Int(sparse_params[1])       # max dictionary size
    q_threshold = sparse_params[2]      # quantization threshold

    centers = [z[1, :]]                 # initialize with first sample
    a = [0.0]                           # coefficient vector
    errors = zeros(N)
    expansion_size = zeros(Int, N)
    w = fill(1/Q, Q)                    # projection weights

    mu_n = 0.5
    minimum_mu = 1e-3
    weight = 1.0

    for n in 1:N
        x_n = z[n, :]
        # current output
        f_n = sum(a[i] * kappa(centers[i], x_n, kern) for i in 1:length(a))
            errors[n] = d[n] - f_n

            # APSM update
            g_coef = compute_subgrad_coef(errors[n], epsilon, loss_obj)
            if abs(g_coef) > 0
                norm_sq = kappa(x_n, x_n, kern)
                step = (errors[n] * w[1]) / (norm_sq + 1e-6)

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

                # Extrapolation step
                mu_n = max(mu_n * weight, minimum_mu)
            end

            return a, centers, errors, expansion_size
        end

        # -------------------------------
        # 4. Example Usage
        # -------------------------------
        Random.seed!(123)
        N = 200
        z = randn(N,2)
        d = sin.(z[:,1]) + 0.1*randn(N)

        epsilon = 1e-5
        Q = 5
        sigma_kern = 1.0
        Delta_max = 1000
        q_threshold = 0.2

        kernel_obj = GaussianKernel(sigma_kern)
        loss_obj = L2Loss()

        a_apsm, centers_apsm, e_apsm, exp_size_apsm = QKernel_APSM(
            z, d, epsilon, Q, loss_obj, kernel_obj, [Delta_max, q_threshold]
            )

        # Smoothed MSE (dB)
        mse_apsm = e_apsm .^ 2
        smoothed_mse_db = 10 .* log10.( [mean(mse_apsm[max(1,i-9):i]) for i in 1:N] )

            println("Final Dictionary Size: ", exp_size_apsm[end])
            println("Final Error (dB): ", round(smoothed_mse_db[end], digits=2))

            # Plot
            plot(smoothed_mse_db, label="QAPSM", color=:blue)
            xlabel!("Iteration")
            ylabel!("MSE (dB)")
            title!("QKernel APSM Performance")


