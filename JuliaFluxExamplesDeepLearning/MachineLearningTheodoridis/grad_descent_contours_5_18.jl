using LinearAlgebra, Plots, Zygote

function safe_grad_descent()
    # 1. Problem Setup
    Σ = [1.0 0.0; 0.0 0.1]
    p = [0.05, 0.03]
    x = [-2.0, -1.0]

    # 2. Stability Check
    # The condition for convergence is 0 < α < 2/λ_max
    λ_max = maximum(eigvals(Σ))
    α_max = 2.0 / λ_max

    # Recommended step size for this specific exercise
    α = 2.0 / (1.0 + 0.1)

    if α >= α_max
        @warn "Step size α ($α) is at or above the stability limit ($α_max). Reducing for safety."
            α = 0.9 * α_max
        end

        # 3. Objective (Consistent with the original sum of squares)
        # J(x) = (p₁ - Σ₁₁x₁)² + (p₂ - Σ₂₂x₂)²
        objective(x) = sum((p .- diag(Σ) .* x).^2)

        history = [copy(x)]
        max_iter = 1000
        tol = 1e-8

        # 4. Iteration with NaN Protection
        for i in 1:max_iter
            # Automatic Differentiation for the gradient
            g = gradient(objective, x)[1]

            x_new = x - α .* g

            # Check for divergence before updating
            if any(isnan.(x_new)) || any(isinf.(x_new))
                @error "Divergence detected at iteration $i. Gradient was $g."
                break
            end

            if norm(x_new - x) < tol
                println("Converged at iteration $i")
                break
            end

            x = x_new
            push!(history, copy(x))
        end

        # 5. Visualization
        xr = -3:0.1:3; yr = -3:0.1:3
        z = [objective([xi, yi]) for xi in xr, yi in yr]

            p1 = contour(xr, yr, z', levels=20, title="Stable GD Path", aspect_ratio=:equal)
            plot!(first.(history), last.(history), marker=:circle, line=:red, label="Path")
            display(p1)
        end

        safe_grad_descent()


