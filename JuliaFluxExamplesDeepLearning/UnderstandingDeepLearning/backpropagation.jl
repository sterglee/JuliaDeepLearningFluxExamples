using Flux
using Statistics
using Plots

# ---------------------------------------------------------
# 1. Define a General Deep Network
# ---------------------------------------------------------
# The notebook uses a network with several hidden layers.
# Let's define a 1-5-5-1 architecture as an example.
D_i, D_h, D_o = 1, 5, 1

model = Chain(
    Dense(D_i => D_h, relu), # Layer 1
    Dense(D_h => D_h, relu), # Layer 2
    Dense(D_h => D_o)        # Layer 3 (Output)
    )

# ---------------------------------------------------------
# 2. Data and Loss Function
# ---------------------------------------------------------
# Synthetic data for regression
x_train = Float32.(collect(-1:0.2:1)')
y_train = sin.(x_train)

# Least Squares Loss (Sum of Squared Errors / 2)
# The notebook specifically uses this form to simplify derivatives
loss_fn(m, x, y) = 0.5f0 * sum((m(x) .- y).^2)

# ---------------------------------------------------------
# 3. Automatic Backpropagation
# ---------------------------------------------------------
# In the notebook, you'd manually compute deltas.
# Here, withgradient does the entire backward pass automatically.
val, grads = Flux.withgradient(model) do m
    loss_fn(m, x_train, y_train)
end

# ---------------------------------------------------------
# 4. Training Loop with train!
# ---------------------------------------------------------
# Replicating the training process using the modern Flux API
opt_state = Flux.setup(Flux.Adam(0.01), model)
data_loader = Flux.DataLoader((x_train, y_train), batchsize=4, shuffle=true)

epochs = 100
for epoch in 1:epochs
    Flux.train!(loss_fn, model, data_loader, opt_state)
end

# ---------------------------------------------------------
# 5. Accessing Gradients for All Layers
# ---------------------------------------------------------
# To inspect gradients for weight matrices and bias vectors across the network:
for (i, layer_grad) in enumerate(grads[1].layers)
    println("Layer $i Weight Gradients: ", size(layer_grad.weight))
    println("Layer $i Bias Gradients:   ", size(layer_grad.bias))
end

# ---------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------
x_plot = Float32.(collect(-1.1:0.01:1.1)')
y_pred = model(x_plot)

scatter(x_train', y_train', label="Training Data", color=:red)
plot!(x_plot', y_pred', label="Backprop-trained Model", lw=2, color=:blue)
title!("Notebook 7.2: Full Backpropagation in Flux")

