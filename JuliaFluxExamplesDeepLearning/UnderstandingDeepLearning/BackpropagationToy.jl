using Flux
using Statistics

# ---------------------------------------------------------
# 1. Define the Toy Model (1-1-1-1 Architecture)
# ---------------------------------------------------------
# x -> Layer 1 -> Layer 2 -> Layer 3 -> Layer 4 -> y
model = Chain(
    Dense(1 => 1, relu), # Layer 1
    Dense(1 => 1, relu), # Layer 2
    Dense(1 => 1, relu), # Layer 3
    Dense(1 => 1)        # Layer 4 (Output)
    )

# ---------------------------------------------------------
# 2. Set Specific Weights (Matching Notebook 7.1)
# ---------------------------------------------------------
model[1].weight .= [0.6];  model[1].bias .= [-0.2]
model[2].weight .= [0.5];  model[2].bias .= [0.4]
model[3].weight .= [-0.3]; model[3].bias .= [0.2]
model[4].weight .= [0.8];  model[4].bias .= [-0.1]

# ---------------------------------------------------------
# 3. Define Input and Loss
# ---------------------------------------------------------
x_input = Float32.([1.0;;])
y_target = Float32.([0.5;;])

# Least Squares Loss: L = 0.5 * (y_pred - y_target)^2
loss_fn(m, x, y) = 0.5f0 * sum((m(x) .- y).^2)

# ---------------------------------------------------------
# 4. Backpropagation (Automatic Differentiation)
# ---------------------------------------------------------
# Flux.withgradient returns a tuple: (loss_value, gradients)
val, grads = Flux.withgradient(model) do m
    loss_fn(m, x_input, y_target)
end

# ---------------------------------------------------------
# 5. Correct Gradient Access
# ---------------------------------------------------------
# The 'grads' object for a Chain is a NamedTuple containing :layers.
# We access it as grads[1].layers[index]
g = grads[1]

println("--- Results ---")
println("Loss Value: ", val)
println("dL/dOmega_4 (Last Layer Weight): ", g.layers[4].weight[1])
println("dL/dBeta_4  (Last Layer Bias):   ", g.layers[4].bias[1])
println("dL/dOmega_1 (First Layer Weight): ", g.layers[1].weight[1])

# ---------------------------------------------------------
# 6. Training Step
# ---------------------------------------------------------
opt_state = Flux.setup(Flux.Descent(0.01), model)

# Flux.train! takes (loss, model, data, opt_state)
# We wrap our single point in a vector to treat it as data
data = [(x_input, y_target)]
Flux.train!(loss_fn, model, data, opt_state)

println("\nUpdated Weight 1 after training step: ", model[1].weight[1])

