using Flux, Plots, LinearAlgebra

# ==========================================
# 1. Define the Residual Function f[z]
# ==========================================
# A contraction mapping g[z] = z + f[z] is invertible if the Lipschitz
# constant of f[z] is less than 1.

# Let's define a simple f[z] as a small neural network with constrained weights
# or a simple sine function as used in the notebook logic.
f(z, ϕ) = ϕ .* sin.(2π .* z) ./ 10.0f0

# ==========================================
# 2. Forward Mapping: x = g[z] = z + f[z]
# ==========================================
forward_mapping(z, ϕ) = z .+ f(z, ϕ)

# ==========================================
# 3. Fixed-Point Iteration (Inversion)
# ==========================================
# To find z from x, we rearrange x = z + f[z] to z = x - f[z].
# We then iterate: z_{k+1} = x - f[z_k]

function solve_fixed_point(x, ϕ; n_iters=20)
    # Initial guess: z_0 = x
    z_k = copy(x)

    # Store history for visualization
    history = [z_k]

    for _ in 1:n_iters
        # Fixed point update step
        z_k = x .- f(z_k, ϕ)
        push!(history, z_k)
    end

    return z_k, history
end

# ==========================================
# 4. Visualization and Verification
# ==========================================
ϕ = 0.8f0  # Lipschitz constant control
x_target = 0.5f0
z_final, path = solve_fixed_point(x_target, ϕ)

println("Target x: ", x_target)
println("Recovered z: ", z_final)
println("Verification (g[z]): ", forward_mapping(z_final, ϕ))

# Plotting the contraction mapping process
z_range = range(0, 1, length=100)
g_vals = forward_mapping(z_range, ϕ)



p = plot(z_range, g_vals, label="g[z] = z + f[z]", lw=2, title="Contraction Mapping Inversion")
plot!(z_range, z_range, label="Identity line", ls=:dash, color=:black)
scatter!([z_final], [x_target], color=:red, label="Fixed Point Solution")
xlabel!("Input z")
ylabel!("Output x")
display(p)

