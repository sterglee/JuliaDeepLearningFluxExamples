using Plots
using DelimitedFiles
using LinearAlgebra
using Statistics
using Random
using Printf

# --- 1. Supporting & Diagnostic Functions ---

function sb1_diagnostic(level::Int, fmt::String, args...)
    # level 1 is standard, level 2+ is verbose
    @printf(fmt, args...)
end

"""
Simplified Posterior Mode for Classification.
    Real implementation would involve iteratively reweighted least squares (IRLS).
    """
    function sb1_posterior_mode(PHI, t, w_init, alpha, max_its)
        # Placeholder using a regularized least squares approximation
        n_basis = size(PHI, 2)
        Hessian = (PHI' * PHI) + Diagonal(alpha)
        u_fact = cholesky(Hermitian(Hessian)).U
        ui = inv(u_fact)
        w = (ui * (ui' * (PHI' * t)))
        # Dummy Gaussian likelihood approximation
        data_likely = -0.5 * sum((t - PHI*w).^2)
        return w, ui, data_likely
    end

    # --- 2. Kernel Engine ---

    function dist_sqrd(X, Y)
        # ||x - y||² = ||x||² + ||y||² - 2xᵀy
        sum_x2 = sum(abs2, X, dims=2)
        sum_y2 = sum(abs2, Y, dims=2)
        D2 = sum_x2 .+ sum_y2' .- 2 .* (X * Y')
        return max.(0.0, D2)
    end

    function sb1_kernel_function(X1::AbstractMatrix, X2::AbstractMatrix, kernel_str::String, length_scale::Real)
        N1, d = size(X1)
        N2, _ = size(X2)
        p = 0
        kernel_type = lowercase(kernel_str)

        # Parse polynomial order
        if startswith(kernel_type, "poly")
            p = parse(Int, kernel_type[5:end])
            kernel_type = "poly"
            elseif startswith(kernel_type, "hpoly")
            p = parse(Int, kernel_type[6:end])
            kernel_type = "hpoly"
        end

        eta = 1 / length_scale^2

        if kernel_type == "gauss"
            return exp.(-eta .* dist_sqrd(X1, X2))
            elseif kernel_type == "tps" # Thin-plate spline
            r2 = eta .* dist_sqrd(X1, X2)
            return 0.5 .* r2 .* log.(r2 .+ (r2 .== 0))
            elseif kernel_type == "cauchy"
            return 1 ./ (1 .+ eta .* dist_sqrd(X1, X2))
            elseif kernel_type == "laplace"
            return exp.(-sqrt.(eta .* dist_sqrd(X1, X2)))
            elseif kernel_type == "poly"
            return (X1 * (eta .* X2') .+ 1) .^ p
        else
            error("Unrecognised kernel: $kernel_str")
        end
    end

    # --- 3. The Sparse Bayesian Learning Engine ---

    function sb1_estimate(PHI, t, alpha_val, beta_val, max_its, mon_its=0)
        MIN_DELTA_LOGALPHA = 1e-3
        ALPHA_MAX = 1e9

        regression = (beta_val != 0)
        fixed_beta = beta_val < 0
        beta = Float64(abs(beta_val))

        n_samples, m_total = size(PHI)
        w = zeros(m_total)
        alpha = fill(Float64(alpha_val), m_total)
        gamma = ones(m_total)
        phit_t = PHI' * t

        useful = trues(m_total)
        marginal = -Inf

        for i in 1:max_its
            # 1. Pruning
            useful = (alpha .< ALPHA_MAX)
            m_curr = sum(useful)
            if m_curr == 0
                @warn "All basis functions pruned."
                break
            end

            alpha_used = alpha[useful]
            phi_used = PHI[:, useful]
            w[.!useful] .= 0

            # 2. Compute Weights
            if regression
                hessian = (phi_used' * phi_used) .* beta + Diagonal(alpha_used)
                u_fact = cholesky(Hermitian(hessian)).U
                ui = inv(u_fact)
                w_useful = (ui * (ui' * phit_t[useful])) * beta
                w[useful] .= w_useful
                ed = sum((t .- phi_used * w_useful).^2)
                data_likely = (n_samples * log(beta) - beta * ed) / 2
            else
                w_useful, ui, data_likely = sb1_posterior_mode(phi_used, t, w[useful], alpha_used, 25)
                w[useful] .= w_useful
            end

            # 3. Covariance & Well-determinedness
            log_det_h = -2.0 * sum(log.(diag(ui)))
            diag_sig = sum(ui.^2, dims=2)[:]
            gamma_used = 1.0 .- alpha_used .* diag_sig
            gamma[useful] .= gamma_used

            # 4. Marginal Likelihood
            marginal = data_likely - 0.5 * (log_det_h - sum(log.(alpha_used)) + (w[useful]' * (alpha_used .* w[useful])))

            # 5. Convergence check & Re-estimation
            log_alpha_old = log.(alpha[useful])
            alpha[useful] .= gamma_used ./ (w[useful].^2 .+ eps())

            if regression && !fixed_beta
                ed = sum((t .- phi_used * w[useful]).^2)
                beta = (n_samples - sum(gamma_used)) / (ed + eps())
            end

            # Monitoring
            if (mon_its > 0 && i % mon_its == 0)
                sb1_diagnostic(1, "%5d> L = %.3f\t RVs = %d\n", i, marginal, m_curr)
            end

            if i > 1 && maximum(abs.(log_alpha_old .- log.(alpha[useful]))) < MIN_DELTA_LOGALPHA
                break
            end
        end

        final_useful_idx = findall(alpha .< ALPHA_MAX)
        return w[final_useful_idx], final_useful_idx, marginal, alpha[final_useful_idx], beta, gamma[final_useful_idx]
    end

    # --- 4. RVM Wrapper ---

    function SB1_RVM(X, t, init_alpha, init_beta, kernel_, length_scale, use_bias, max_its, mon_its)
        phi = sb1_kernel_function(X, X, kernel_, length_scale)
        n_samples = size(X, 1)

        if use_bias
            phi = hcat(phi, ones(n_samples, 1))
        end

        weights, used, marginal, alpha, beta, gamma = sb1_estimate(
            phi, t, init_alpha, init_beta, max_its, mon_its
            )

        bias = 0.0
        if use_bias
            bias_idx = findfirst(==(n_samples + 1), used)
            if bias_idx !== nothing
                bias = weights[bias_idx]
                deleteat!(weights, bias_idx)
                deleteat!(used, bias_idx)
            end
        end

        return weights, used, bias, marginal, alpha, beta, gamma
    end

    # --- 5. Classification Demo ---

    function sb1_example_classify(N=100, kernel_="gauss", width=0.5, max_its=500)
        Random.seed!(42)

        # Check for Ripley data; create dummy if missing
        if !isfile("synth.tr")
            @info "synth.tr not found. Creating synthetic dummy data..."
            dummy_data = [randn(N, 2) .+ 2.0 rand(0:1, N)]
            dummy_data[1:Int(N/2), 1:2] .-= 4.0
            writedlm("synth.tr", dummy_data)
            writedlm("synth.te", dummy_data)
        end

        synth_tr = readdlm("synth.tr")
        X, t = synth_tr[1:N, 1:2], synth_tr[1:N, 3]

        weights, used, bias, ml, alpha, beta, gamma = SB1_RVM(
            X, t, 1e-3, 0.0, kernel_, width, true, max_its, 50
            )

        # Visualize Decision Boundary
        x_range = range(minimum(X[:,1])-1, maximum(X[:,1])+1, length=50)
        y_range = range(minimum(X[:,2])-1, maximum(X[:,2])+1, length=50)
        grid_pts = hcat([x for x in x_range, y in y_range][:], [y for x in x_range, y in y_range][:])

            phi_grid = sb1_kernel_function(grid_pts, X[used, :], kernel_, width)
            p_grid = 1 ./ (1 .+ exp.(-(phi_grid * weights .+ bias)))
            P = reshape(p_grid, length(x_range), length(y_range))'

            plt = scatter(X[t .== 0, 1], X[t .== 0, 2], label="Class 0", mc=:black, ms=4)
            scatter!(X[t .== 1, 1], X[t .== 1, 2], label="Class 1", mc=:green, ms=4)
            contour!(x_range, y_range, P, levels=[0.5], color=:red, lw=2, label="Boundary")
            scatter!(X[used, 1], X[used, 2], m=:circle, ms=8, mc=:transparent, msc=:red, label="RVs")
            title!("RVM Classification")

            display(plt)
        end

        # Run the demo
        sb1_example_classify()

