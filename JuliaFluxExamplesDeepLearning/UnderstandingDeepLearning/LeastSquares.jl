using Flux
using Statistics
using Random
using Plots

# ---------------------------------------------------------
# 1. Define the Univariate Shallow Network
# ---------------------------------------------------------
# Architecture: 1 input, 10 hidden units (ReLU), 1 output
# This corresponds to the shallow_nn function in the notebook
D_i, D_h, D_o = 1, 10, 1

model = Chain(
    Dense(D_i => D_h, relu),
    Dense(D_h => D_o)
    )

# ---------------------------------------------------------
# 2. Data Generation
# ---------------------------------------------------------
# Create synthetic training data as seen in the notebook
x_train = Float32.(collect(0:0.1:1)')
# Generate y with some Gaussian noise (Least Squares assumes Gaussian noise)
y_train = sin.(x_train .* 5) .+ 0.2f0 .* randn(Float32, size(x_train))

loader = Flux.DataLoader((x_train, y_train), batchsize=4, shuffle=true)

# ---------------------------------------------------------
# 3. Least Squares Loss and Likelihood
# ---------------------------------------------------------
# The notebook shows that minimizing Least Squares is equivalent to
# maximizing the Log Likelihood of a Gaussian.

# Standard MSE Loss (Least Squares)
loss_fn(m, x, y) = Flux.mse(m(x), y)

# ---------------------------------------------------------
# 4. Training Loop using Flux.train!
# ---------------------------------------------------------
opt_state = Flux.setup(Flux.Adam(0.02), model)
epochs = 200

println("Training to minimize Least Squares loss...")
for epoch in 1:epochs
    Flux.train!(loss_fn, model, loader, opt_state)

    if epoch % 50 == 0
        current_loss = loss_fn(model, x_train, y_train)
        println("Epoch $epoch: MSE Loss = $current_loss")
    end
end

# ---------------------------------------------------------
# 5. Visualization of the Fit
# ---------------------------------------------------------
x_plot = Float32.(collect(0:0.01:1)')
y_pred = model(x_plot)

# Plotting the model fit and the training data
scatter(x_train', y_train', label="Training Data (noisy)", color=:red)
p1=plot!(x_plot', y_pred', label="Flux Model (Least Squares Fit)", lw=3, color=:blue)
xlabel!("Input x")
ylabel!("Output y")
title!("Notebook 5.1: Least Squares Loss in Flux")
display(p1)


