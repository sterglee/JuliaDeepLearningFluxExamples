using LinearAlgebra
using Statistics
using Plots
using DSP
using Random

function run_lms_correlated()

  # ==============================
  # 1. Simulation Parameters
  # ==============================
  L       = 10            # Adaptive filter length
  N       = 2500          # Number of samples
  IterNo  = 100           # Monte Carlo runs
  noisevar = 0.01f0       # Measurement noise variance
  μ1      = 0.01f0        # Step size 1
  μ2      = 0.001f0       # Step size 2

  # Storage for Monte Carlo averaging
  MSE1 = zeros(Float32, N, IterNo)
  MSE2 = zeros(Float32, N, IterNo)

  println("Running $IterNo Monte Carlo iterations...")

  # ==============================
  # 2. Monte Carlo Loop
  # ==============================
  for it in 1:IterNo

    # ---- Generate Correlated AR(1) Input ----
    # AR(1): x[n] = -0.85x[n-1] + w[n]
    raw_noise = randn(Float32, N + L - 1)

    b = [1.0f0]              # Numerator
    a = [1.0f0, 0.85f0]      # Denominator
    x = filt(b, a, raw_noise)

    # Normalize to unit variance
    x ./= std(x)

    # ---- Construct Regressor Matrix X (L x N) ----
    X = zeros(Float32, L, N)
    for n in 1:N
      X[:, n] = x[n + L - 1:-1:n]
    end

    # ---- Unknown System ----
    θ = randn(Float32, L)
    v = sqrt(noisevar) .* randn(Float32, N)
    y = X' * θ .+ v

    # ---- LMS Function ----
    function lms(μ)
      w = zeros(Float32, L)
      mse = zeros(Float32, N)

      for n in 1:N
        xn = X[:, n]
        e  = y[n] - dot(w, xn)
        w .+= μ * e .* xn
        mse[n] = e^2
      end

      return mse
    end

    MSE1[:, it] = lms(μ1)
    MSE2[:, it] = lms(μ2)
  end

  # ==============================
  # 3. Average Learning Curves
  # ==============================
  MSEavg1 = vec(mean(MSE1, dims=2))
  MSEavg2 = vec(mean(MSE2, dims=2))

  # Prevent log(0)
  epsval = Float32(1e-12)

  MSEdB1 = 10 .* log10.(MSEavg1 .+ epsval)
  MSEdB2 = 10 .* log10.(MSEavg2 .+ epsval)

  # ==============================
  # 4. Plot
  # ==============================
  p = plot(MSEdB1,
           label="μ = 0.01",
           linewidth=2,
           title="LMS Convergence with Correlated AR(1) Input",
           xlabel="Iteration",
           ylabel="MSE (dB)"
           )

  plot!(p, MSEdB2, label="μ = 0.001", linewidth=2)

  display(p)

  println("Simulation complete.")
end

# Run
run_lms_correlated()

