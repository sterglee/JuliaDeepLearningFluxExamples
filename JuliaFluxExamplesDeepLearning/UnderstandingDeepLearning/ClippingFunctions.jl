using Flux
using Statistics
using Random
using Plots

# ---------------------------------------------------------
# 1. Define the Network Architecture
# ---------------------------------------------------------
# The notebook describes a 1-3-3-1 architecture
# Layer 1: 1 input -> 3 hidden units (ReLU)
# Layer 2: 3 hidden units -> 3 hidden units (ReLU)
# Layer 3: 3 hidden units -> 1 output (Linear)

model = Chain(
    Dense(1 => 3, relu), # Layer 1 (theta parameters)
    Dense(3 => 3, relu), # Layer 2 (psi parameters)
    Dense(3 => 1)        # Layer 3 (phi parameters)
    )

# ---------------------------------------------------------
# 2. Manual Parameter Initialization (Optional)
# ---------------------------------------------------------
# To match the notebook's specific manual values exactly:
# Note: Julia/Flux use Column-Major (Out x In) for weight matrices.

# Example: setting specific weights for the first layer (theta)
# model[1].weight .= [-1.0; 2.0; 0.65] # theta_11, theta_21, theta_31
# model[1].bias   .= [0.3; -1.0; -0.5] # theta_10, theta_20, theta_30

# ---------------------------------------------------------
# 3. Data Preparation
# ---------------------------------------------------------
# Range of input values [0, 1] as defined in the notebook
x_data = collect(0:0.01:1)'  # Shape (1, 101) - Flux expects (features, observations)

# Create some dummy target data for the training example
y_target = sin.(x_data .* 2π) .+ 0.5f0

# Create a DataLoader for batching
loader = Flux.DataLoader((x_data, y_target), batchsize=16, shuffle=true)

# ---------------------------------------------------------
# 4. Training Setup
# ---------------------------------------------------------
# Define the loss function (Mean Squared Error)
loss_fn(m, x, y) = Flux.mse(m(x), y)

# Choose the Adam optimizer
opt_state = Flux.setup(Flux.Adam(0.01), model)

# ---------------------------------------------------------
# 5. Training Loop
# ---------------------------------------------------------
epochs = 100
println("Starting training...")

for epoch in 1:epochs
    # Flux.train! automates the gradient calculation and update step
    Flux.train!(loss_fn, model, loader, opt_state)

    if epoch % 20 == 0
        current_loss = loss_fn(model, x_data, y_target)
        println("Epoch $epoch: Loss = $current_loss")
    end
end

# ---------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------
y_pred = model(x_data)

p=plot(x_data', y_target', label="Target", lw=2)
plot!(x_data', y_pred', label="Model Prediction", lw=2, linestyle=:dash)
xlabel!("Input x")
ylabel!("Output y")
title!("Deep Network (1-3-3-1) Flux Implementation")
display(p)

