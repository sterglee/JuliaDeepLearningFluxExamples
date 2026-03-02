using LinearAlgebra, Random, Plots, Statistics

# -------------------------------
# 1. Gaussian Kernel
# -------------------------------
struct GaussianKernel
    sigma::Float64
end

function kappa(x::AbstractVector, y::AbstractVector, kern::GaussianKernel)
    return exp(-norm(x - y)^2 / (2 * kern.sigma^2))
end

# -------------------------------
# 2. QKernel NLMS (Kernel LMS)
# -------------------------------
function QKernel_NLMS(z::Matrix{Float64}, d::Vector{Float64}, mu::Float64,
                      sparse_flag::Int, print_flag::Int, kern::GaussianKernel,
                      sparse_params::Vector{Float64})

    epsilon = 1e-6
    q_size = sparse_params[1]       # quantization threshold
    weight = 1.0
    N, L = size(z)

    a = Float64[]                   # coefficients
    centers = Array{Float64,2}(undef, 0, L)
    e = zeros(N)
    expansion_size = zeros(Int, N)
    N0 = 0                           # number of centers

    for n in 1:N
        # Compute current output y_n
        R_sum = 0.0
        if N0 > 0
            for k in 1:N0
                R_sum += a[k] * kappa(centers[k, :], z[n, :], kern)
            end
        end
        y_n = R_sum
        e[n] = d[n] - y_n

        # Quantization / sparsification
        add_this_center = true
        min_dist = Inf
        thesis = 0
        for k in 1:N0
            dist = norm(z[n, :] - centers[k, :])^2
            if dist < min_dist
                min_dist = dist
                thesis = k
            end
        end
        if min_dist < q_size
            add_this_center = false
        end

        # Update coefficients
        if add_this_center
            N0 += 1
            centers = vcat(centers, z[n, :]')         # add as new row
            push!(a, mu * e[n] / (kappa(z[n, :], z[n, :], kern) + epsilon))
        else
            a[thesis] += mu * e[n] / (kappa(z[n, :], z[n, :], kern) + epsilon)
        end

        expansion_size[n] = N0
        mu *= weight
    end

    # Plot MSE in dB
    if print_flag == 1
        mse = e .^ 2
        smoothed_mse_db = 10 .* log10.( [mean(mse[max(1,i-9):i]) for i in 1:N] )
            plot(smoothed_mse_db, label="QKNLMS", color=:red)
            xlabel!("Iteration")
            ylabel!("MSE (dB)")
            title!("QKernel NLMS Performance")
        end

        return a, centers, e, expansion_size
    end

    # -------------------------------
    # 3. Example Usage
    # -------------------------------
    Random.seed!(123)
    N = 200
    z = randn(N,2)
    d = sin.(z[:,1]) + 0.1*randn(N)

    mu = 0.5
    sparse_flag = 1
    print_flag = 1
    sigma_kern = 1.0
    kernel_obj = GaussianKernel(sigma_kern)
    sparse_params = [0.2]  # quantization threshold

    a_knlms, centers_knlms, e_knlms, exp_size_knlms = QKernel_NLMS(
        z, d, mu, sparse_flag, print_flag, kernel_obj, sparse_params
        )

    println("Final dictionary size: ", exp_size_knlms[end])


