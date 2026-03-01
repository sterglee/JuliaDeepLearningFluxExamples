using Flux
using LinearAlgebra
using Plots

function run_exercise_518()
  # --- 1. Problem Definition ---
  Σ = Float32[1.0 0.0; 0.0 0.1]
  p = Float32[0.05, 0.03]
  x0 = Float32[-2.0, -1.0]

  # The optimum of the quadratic surface Σx = p
  x_opt = Σ \ p

  # Theoretical optimum step sizes (eigenvalue based)
  λ_max, λ_min = 1.0f0, 0.1f0
  α_opt = 2.0f0 / (λ_max + λ_min)
  α_half = α_opt / 2.0f0

  # --- 2. Optimization Function ---
  function train_model(start_x, step_size, iterations)
    # In Flux, we wrap parameters in a 'Params' object or a 'Leaf'
    # Here we use a simple vector but manual gradient application
    # to match the Exercise's specific update rule.
    w = copy(start_x)
    mse_history = Float32[]

    for i in 1:iterations
      # Calculate Mean Squared Error relative to the optimum
      push!(mse_history, norm(w - x_opt)^2)

      # Gradient calculation:
      # The original exercise uses g = Σw - p
      # Note: The true gradient of ||p - Σw||² is 2Σ(Σw - p)
      # We follow the manual gradient definition from your script:
      g = Σ * w - p

      # Apply Gradient Descent update
      w -= step_size * g

      if !all(isfinite.(w))
        @error "Diverged at iteration $i"
        return mse_history
      end
    end
    return mse_history
  end

  # --- 3. Execution ---
  max_iter = 200
  err_opt = train_model(x0, α_opt, max_iter)
  err_half = train_model(x0, α_half, max_iter)

  # --- 4. Visualization ---
  plot(err_opt,
       label="Optimum Step-size (α = $(round(α_opt, digits=2)))",
       lc=:blue, lw=2, yaxis=:log)
  plot!(err_half,
        label="Half Optimum Step-size (α = $(round(α_half, digits=2)))",
        lc=:red, lw=2)

  title!("Convergence Curves (Exercise 5.18)")
  xlabel!("Iterations")
  ylabel!("Squared Error: ||x - x_opt||²")
end

run_exercise_518()

