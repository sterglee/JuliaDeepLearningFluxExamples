using LinearAlgebra
using Statistics
using DSP
using Plots
using Random

function LMS()

    # =====================================
    # 1. Parameters
    # =====================================
    L       = 10
    N       = 2500
    IterNo  = 100
    noisevar = 0.01

    θ = randn(L)

    MSE1 = zeros(N, IterNo)
    MSE2 = zeros(N, IterNo)

    println("Running $IterNo Monte Carlo simulations...")

    # =====================================
    # 2. WHITE INPUT CASE
    # =====================================
    for It in 1:IterNo

        X = randn(L, N)

        noise = sqrt(noisevar) .* randn(N)
        y = X' * θ .+ noise

        w = zeros(L)
        μ = 0.01

        for n in 1:N
            x = X[:, n]
            e = y[n] - dot(w, x)
            w += μ * e * x
            MSE1[n, It] = e^2
        end
    end

    # =====================================
    # 3. AR(1) CORRELATED INPUT CASE
    # =====================================
    for It in 1:IterNo

        # Generate AR(1) input
        xcorrel = randn(N + L - 1)
        xcorrel = filt([1.0], [1.0, 0.85], xcorrel)
        xcorrel ./= std(xcorrel)

        # Build regression matrix manually (L × N)
        X = zeros(L, N)
        for n in 1:N
            X[:, n] = xcorrel[n + L - 1:-1:n]
        end

        noise = sqrt(noisevar) .* randn(N)
        y = X' * θ .+ noise

        w = zeros(L)
        μ = 0.01

        for n in 1:N
            x = X[:, n]
            e = y[n] - dot(w, x)
            w += μ * e * x
            MSE2[n, It] = e^2
        end
    end

    # =====================================
    # 4. Monte Carlo Average
    # =====================================
    MSEav1 = vec(mean(MSE1, dims=2))
    MSEav2 = vec(mean(MSE2, dims=2))

    epsval = 1e-12
    MSEdB1 = 10 .* log10.(MSEav1 .+ epsval)
    MSEdB2 = 10 .* log10.(MSEav2 .+ epsval)

    # =====================================
    # 5. Plot
    # =====================================
    plot(MSEdB1,
         color=:red,
         linewidth=2,
         label="White Input",
         xlabel="Iteration",
         ylabel="MSE (dB)",
         title="LMS Convergence: White vs AR(1) Input"
         )

    plot!(MSEdB2,
          color=:black,
          linewidth=2,
          label="AR(1) Input")

end

# Run
LMS()

