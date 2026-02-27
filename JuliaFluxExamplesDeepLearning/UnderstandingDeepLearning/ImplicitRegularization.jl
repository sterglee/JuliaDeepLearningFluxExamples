using Flux
using Statistics
using Plots
using LinearAlgebra
using Zygote

# ---------------------------------------------------------
# 1. Define the Loss Function Surface
# ---------------------------------------------------------
function loss_surface(phi)
    x, y = phi[1], phi[2]
    # Simple non-convex surface: sin(x)cos(y) + quadratic bowl
    return sin(x) * cos(y) + 0.1f0 * (x^2 + y^2)
end

# Helper to get gradient as a vector
get_grad(p) = Flux.gradient(loss_surface, p)[1]

# ---------------------------------------------------------
# 2. Gradient Descent Functions
# ---------------------------------------------------------

# Standard GD: Follows L(phi)
function run_standard_gd(start_phi, lr, steps)
    phi = copy(start_phi)
    path = [copy(phi)]
    for _ in 1:steps
        phi .-= lr .* get_grad(phi)
        push!(path, copy(phi))
    end
    return path
end

# CORRECTED Implicit Reg GD
# We use Zygote.hessian or a manual second-order approximation 
# to avoid the "gradient within gradient" error.
function run_implicit_reg_gd(start_phi, lr, steps)
    phi = copy(start_phi)
    path = [copy(phi)]
    alpha_reg = lr / 4.0f0
    
    for _ in 1:steps
        # Instead of nested Flux.gradient, we compute the explicit 
        # update rule for the modified loss: 
        # Grad_modified = Grad_L + alpha_reg * Hessian_L * Grad_L
        
        g = get_grad(phi)
        # Use Zygote.hessian to get the 2nd derivatives
        H = Zygote.hessian(loss_surface, phi)
        
        # The update direction for the implicitly regularized path
        modified_grad = g + alpha_reg .* (H * g)
        
        phi .-= lr .* modified_grad
        push!(path, copy(phi))
    end
    return path
end

# ---------------------------------------------------------
# 3. Execution
# ---------------------------------------------------------
start_pos = Float32.([-0.7, -0.75])

# 1. Small Step (Reference "True" Path)
path_small_lr = run_standard_gd(start_pos, 0.001f0, 10000)

# 2. Large Step (Implicitly Regularized Path)
# This zig-zags and deviates from the true path
path_typical_lr = run_standard_gd(start_pos, 0.05f0, 200)

# 3. Explicitly Regularized Path (Calculated with tiny steps)
# This should match the large-step path
path_explicit_reg = run_implicit_reg_gd(start_pos, 0.001f0, 10000)

# ---------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------
x_range = -2:0.1:2
y_range = -2:0.1:2
z = [loss_surface([x, y]) for y in y_range, x in x_range]

p = contour(x_range, y_range, z, title="Implicit Regularization Fix", fill=true, color=:viridis)

# Small LR
plot!(p, [p[1] for p in path_small_lr], [p[2] for p in path_small_lr], 
      label="Small LR (True Path)", color=:black, lw=2)

# Large LR (Standard GD)
plot!(p, [p[1] for p in path_typical_lr], [p[2] for p in path_typical_lr], 
      label="Large LR (Actual path)", color=:red, lw=2)

# Explicitly Regularized (The theory prediction)
plot!(p, [p[1] for p in path_explicit_reg], [p[2] for p in path_explicit_reg], 
      label="Modified Loss Prediction", color=:white, ls=:dash, lw=2)

      