using Flux
using Statistics
using Plots
using Optim  # For advanced line search and optimization

# ---------------------------------------------------------
# 1. Define a 1D Loss Function
# ---------------------------------------------------------
# The notebook uses a 1D function to demonstrate line search.
# We'll create a "loss surface" function similar to the notebook.
function loss_function(ϕ)
    return cos(3.0 * ϕ - 0.5) + 0.1 * ϕ^2 + 2.0
end

# ---------------------------------------------------------
# 2. Golden Section Search Implementation
# ---------------------------------------------------------
# This follows the logic of the "line_search" function in the notebook.
function golden_section_search(f, a, d; tol=0.001)
    # Golden ratio constant
    gr = (sqrt(5) + 1) / 2

    # Define initial internal points
    b = d - (d - a) / gr
    c = a + (d - a) / gr

    while abs(d - a) > tol
        if f(b) < f(c)
            d = c
        else
            a = b
        end
        # Recompute internal points
        b = d - (d - a) / gr
        c = a + (d - a) / gr
    end
    return (a + d) / 2
end

# ---------------------------------------------------------
# 3. Visualization of the Line Search
# ---------------------------------------------------------
ϕ_vals = collect(-1.0:0.01:4.0)
losses = loss_function.(ϕ_vals)

# Find the minimum using our line search
soln = golden_section_search(loss_function, -1.0, 4.0)
min_loss = loss_function(soln)



display(plot(ϕ_vals, losses, label="Loss Function", lw=2, color=:black))
scatter!([soln], [min_loss], label="Minimum Found", color=:red, markersize=6)
xlabel!("Step size (ϕ)")
ylabel!("Loss")
title!("Notebook 6.1: Golden Section Line Search")

# ---------------------------------------------------------
# 4. Modern Flux Integration (Gradient Descent)
# ---------------------------------------------------------
# In a real Flux model, we calculate gradients and move in that direction.
# Line search would determine exactly how far to move.

model = Chain(Dense(1 => 1)) # Simple model
x_data = [1.0f0;;]
y_target = [2.0f0;;]

# Instead of a manual line search, we use Flux's built-in optimizers
# which handle the "step" based on the gradient.
opt_state = Flux.setup(Flux.Adam(0.1), model)

println("Line Search Solution: ϕ = $soln, Loss = $min_loss")

