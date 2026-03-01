using LinearAlgebra
using Statistics
using Plots
using Random

function LMS_white_two_stepsize()

    # =========================
    # 1. Parameters
    # =========================
    L       = 10         # Dimension of unknown vector
    N       = 2500       # Number of samples
    IterNo  = 100        # Monte Carlo runs
    noisevar = 0.01

    θ = randn(L)         # Unknown system

    MSE1 = zeros(N, IterNo)
    MSE2 = zeros(N, IterNo)

    println("Running $IterNo Monte Carlo simulations for two step sizes...")

        # =========================
        # 2. Monte Carlo loop
        # =========================
        for It in 1:IterNo

            # Generate white input
            xvec = randn(L, N)

            # Additive noise
            noise = sqrt(noisevar) .* randn(N)
            y = xvec' * θ .+ noise  # Desired signal

            # --- LMS with small step μ1 = 0.01 ---
            w = zeros(L)
            μ1 = 0.01
            for n in 1:N
                x = xvec[:, n]
                e = y[n] - dot(w, x)
                w += μ1 * e * x
                MSE1[n, It] = e^2
            end

            # --- LMS with larger step μ2 = 0.075 ---
            w = zeros(L)
            μ2 = 0.075
            for n in 1:N
                x = xvec[:, n]
                e = y[n] - dot(w, x)
                w += μ2 * e * x
                MSE2[n, It] = e^2
            end
        end

        # =========================
        # 3. Monte Carlo averaging
        # =========================
        MSEav1 = vec(mean(MSE1, dims=2))
        MSEav2 = vec(mean(MSE2, dims=2))

        epsval = 1e-12
        MSEdB1 = 10 .* log10.(MSEav1 .+ epsval)
        MSEdB2 = 10 .* log10.(MSEav2 .+ epsval)

        # =========================
        # 4. Plot
        # =========================
        plot(MSEdB1,
             color=:red,
             linewidth=2,
             label="μ = 0.01",
             xlabel="Iteration",
             ylabel="MSE (dB)",
             title="LMS Convergence with White Input (Two Step Sizes)"
             )
        plot!(MSEdB2,
              color=:black,
              linewidth=2,
              label="μ = 0.075")

    end

    # Run
    LMS_white_two_stepsize()

