using Flux
using Statistics
using Plots
using Random

# ---------------------------------------------------------
# 1. Define the Model and Data
# ---------------------------------------------------------
# Simple linear model: y = wx + b
model = Chain(Dense(1 => 1))

# Training data from the notebook
x_train = Float32.([0.03, 0.19, 0.34, 0.46, 0.78, 0.81, 1.08, 1.18, 1.39, 1.60, 1.65, 1.90]')
y_train = Float32.([0.67, 0.85, 1.05, 1.10, 1.40, 1.50, 1.30, 1.60, 1.70, 2.10, 2.30, 2.20]')

# Adam is usually used with mini-batches
loader = Flux.DataLoader((x_train, y_train), batchsize=5, shuffle=true)

# ---------------------------------------------------------
# 2. Setup Adam Optimizer
# ---------------------------------------------------------
loss_fn(m, x, y) = Flux.mse(m(x), y)

# alpha (learning rate) = 0.01 (standard default)
# beta1 = 0.9 (momentum), beta2 = 0.999 (scaling)
learning_rate = 0.05f0
opt_adam = Flux.setup(Flux.Adam(learning_rate, (0.9, 0.999)), model)

# ---------------------------------------------------------
# 3. Training Loop with train!
# ---------------------------------------------------------
epochs = 100
history = []

println("Starting Gradient Descent with Adam...")

for epoch in 1:epochs
    Flux.train!(loss_fn, model, loader, opt_adam)

    # Track parameter trajectory
    push!(history, (model[1].bias[1], model[1].weight[1]))

    if epoch % 20 == 0
        current_loss = loss_fn(model, x_train, y_train)
        println("Epoch $epoch: Loss = $current_loss")
    end
end

# ---------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------
# Plot the adaptive path in parameter space
p1 = plot([h[1] for h in history], [h[2] for h in history],
              marker=:circle, color=:purple, label="Adam Path",
              title="Adam Trajectory",
              xlabel="Intercept (Bias)", ylabel="Slope (Weight)")



    # Plot the final fit
    x_plot = Float32.(collect(0:0.01:2)')
    p2 = scatter(x_train', y_train', label="Data", color=:red)
    plot!(x_plot', model(x_plot)', label="Model", lw=2, color=:black, title="Final Fit")

    plot(p1, p2, layout=(1, 2), size=(900, 400))

