using LinearAlgebra
using Random
using Statistics
using Plots

# Note: This script assumes you have implemented:
# QKernel_NLMS, NORMA_L2, QKernel_APSM, and sum_mean_val2 (sliding window average)

function main_simulation()
    Random.seed!(0)

    # --- Simulation Parameters ---
    NA = 5000
    N = 10000
    NUMBER_OF_TESTS = 100 # Reduced from 1000 for faster demonstration
    p = 5
    L = 5
    D = 2

    # Channels
    h1 = [-0.9, 0.6, -0.7, 0.2, 0.1]
    h2 = [0.8, -0.7, 0.6, -0.2, -0.2]

    # Preallocate Accumulators
    N_out = N - L + 1
    total_mse_knlms = zeros(N_out)
    total_mse_norma = zeros(N_out)
    total_mse_apsm  = zeros(N_out)

    total_exp_knlms = zeros(N_out)
    total_exp_norma = zeros(N_out)
    total_exp_apsm  = zeros(N_out)

    for i in 1:NUMBER_OF_TESTS
        println("-------------------------------------------------------------------")
        println("TEST NUMBER : $i")

        # 1. Signal and Noise Generation
        var_signal = 0.8
        s = var_signal .* randn(N)

        cur_snr = 15
        var_noise = sqrt(var_signal^2 / (10^(cur_snr/10)))
        noise = var_noise .* randn(N)

        # 2. Non-stationary Channel Generation
        r = zeros(N)
        t = zeros(N)

        # Channel 1 (n=1 to NA)
        for n in p+1:NA
            xi_n = s[n:-1:n-p+1]
            t[n] = dot(h1, xi_n) + noise[n]
            r[n] = t[n] + 0.15*t[n]^2 + 0.03*t[n]^3 + noise[n]
        end

        # Channel 2 (n=NA+1 to N)
        for n in NA+1:N
            xi_n = s[n:-1:n-p+1]
            t[n] = dot(h2, xi_n) + noise[n]
            r[n] = t[n] + 0.12*t[n]^2 + 0.02*t[n]^3 + noise[n]
        end

        # 3. Construct Input Matrix Z and Desired Vector D
        # Matches MATLAB: z(n-N0+1,:) = r(n+D-L+1:n+D)
        N0 = -D + L
        N1 = N - L + 1
        z = zeros(N1, L)
        d = zeros(N1)

        for n in (L-D):(N-D)
            idx = n - (L-D) + 1
            z[idx, :] = r[(n + D - L + 1):(n + D)]
            d[idx] = s[n]
        end

        # 4. Run Kernel Algorithms
        sigma = 5.0

        # QKNLMS
        # [a, centers, e, expansion_size] = QKernel_NLMS(...)
        _, _, e_knlms, exp_knlms = QKernel_NLMS(z, d, 0.5, Gaussian(sigma), [7])

        # NORMA L2
        # [a, centers, e, expansion_size] = NORMA_L2(...)
        _, _, e_norma, exp_norma = NORMA_L2(z, d, 0.25, 0.01, Gaussian(sigma), [80])

        # QAPSM
        # [a, centers, e, expansion_size] = QKernel_APSM(...)
        _, _, e_apsm, exp_apsm = QKernel_APSM(z, d, 1e-5, 5, L2Loss(), Gaussian(sigma), [10000, 7])

        # 5. Accumulate Results
        total_mse_knlms .+= e_knlms .^ 2
        total_mse_norma .+= e_norma .^ 2
        total_mse_apsm  .+= e_apsm .^ 2

        total_exp_knlms .+= exp_knlms
        total_exp_norma .+= exp_norma
        total_exp_apsm  .+= exp_apsm
    end

    # 6. Averaging
    mean_mse_knlms = total_mse_knlms ./ NUMBER_OF_TESTS
    mean_mse_norma = total_mse_norma ./ NUMBER_OF_TESTS
    mean_mse_apsm  = total_mse_apsm ./ NUMBER_OF_TESTS

    # --- Plotting ---
    p1 = plot(10log10.(mean_mse_norma), label="NORMA", color=:black)
    plot!(p1, 10log10.(mean_mse_knlms), label="QKNLMS", color=:red)
    plot!(p1, 10log10.(mean_mse_apsm), label="QAPSM", color=:gray)
    title!(p1, "Non-linear Channel Equalization (MSE)")
    ylabel!(p1, "MSE (dB)")

    p2 = plot(total_exp_norma ./ NUMBER_OF_TESTS, label="QNORMA", lw=2, color=:black)
    plot!(p2, total_exp_knlms ./ NUMBER_OF_TESTS, label="QKNLMS", lw=2, color=:blue)
    plot!(p2, total_exp_apsm ./ NUMBER_OF_TESTS, label="QAPSM", lw=2, color=:green)
    title!(p2, "Evolution of Expansion Size")
    ylabel!(p2, "Size (M)")

    display(plot(p1, p2, layout=(2,1), size=(800, 800)))
end

