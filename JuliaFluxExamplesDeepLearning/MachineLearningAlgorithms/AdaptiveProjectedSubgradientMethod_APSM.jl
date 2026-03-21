using LinearAlgebra, Random, Statistics, Plots

"""
projection(y, x, ϵ, θ)
Project point θ onto the hyperslab defined by y, x, and ϵ.
"""
function projection(y, x, ϵ, θ)
      v = dot(x, θ) - y
      norm_x_sq = sum(abs2, x)

      if v + ϵ < 0
            β = (-v - ϵ) / norm_x_sq
            elseif v - ϵ > 0
            β = (-v + ϵ) / norm_x_sq
      else
            β = 0.0
      end
      return θ + β * x
end

# --- Setup and Data Generation ---
rng = MersenneTwister(1)
L = 100               # Dimensionality
θ_true = randn(rng, L) # True model parameters
q = 30                # Window length
N = 3500              # Number of samples

σ = 0.1               # Noise std dev
factor = 0.5          # mu = factor * M_n
ϵ = sqrt(1) * σ       # Sensitivity

# Generate Observations
η = randn(rng, N) .* σ
X = randn(rng, N, L)
y = X * θ_true + η

θ_init = randn(rng, L)
mse = Float64[]

# --- Main APSM Loop ---
θ_hat = copy(θ_init)

for n in 1:(N-1)
      # Define the active sliding window indices
      if n < q
            idx = 1:n
            w = fill(1.0/n, n)
      else
            idx = (n - q + 1):n
            w = fill(1.0/q, q)
      end

      X_active = X[idx, :]
      y_active = y[idx]

      # Calculate projections for each constraint in the window
      # We use a list comprehension for the projections
      Ps = [projection(y_active[i], X_active[i, :], ϵ, θ_hat) for i in 1:length(idx)]

            # Weighted sum of projections
            sum_w_Ps = sum(w[i] * Ps[i] for i in 1:length(Ps))

                  # Calculate M_n (the step size control)
                  diff_norm_sq = norm(θ_hat - sum_w_Ps)^2

                  # Calculate M as weighted average of squared distances
                  distances_sq = [norm(Ps[i] - θ_hat)^2 for i in 1:length(Ps)]
                        M = dot(w, distances_sq) / (diff_norm_sq + 1e-12) # Added small constant to prevent div by 0

                        # Update Rule
                        μ = factor * M
                        θ_hat = θ_hat + μ * (sum_w_Ps - θ_hat)

                        # Track MSE in dB
                        push!(mse, 10 * log10(mean((θ_true .- θ_hat).^2)))
                  end

                  # --- Plotting ---
                  plot(mse, color=:red, label="APSM", lw=1.5,
                       ylabel="MSE [dB]", xlabel="Iterations",
                       title="Adaptive Projected Subgradient Method",
                       grid=:both)


