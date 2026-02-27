using Flux
using Statistics
using Plots
using Random

# ---------------------------------------------------------
# 1. Define the Model and Data
# ---------------------------------------------------------
# Same linear model as before: y = wx + b
model = Chain(Dense(1 => 1))

# Training data from the notebook
x_train = Float32.([0.03, 0.19, 0.34, 0.46, 0.78, 0.81, 1.08, 1.18, 1.39, 1.60, 1.65, 1.90]')
y_train = Float32.([0.67, 0.85, 1.05, 1.10, 1.40, 1.50, 1.30, 1.60, 1.70, 2.10, 2.30, 2.20]')

# ---------------------------------------------------------
# 2. Stochastic Data Loading
# ---------------------------------------------------------
# batchsize=5 as suggested in the notebook TODO
# shuffle=true is what makes it "Stochastic" (randomized batches)
loader = Flux.DataLoader((x_train, y_train), batchsize=5, shuffle=true)

# ---------------------------------------------------------
# 3. Setup Training
# ---------------------------------------------------------
loss_fn(m, x, y) = Flux.mse(m(x), y)

# Flux.Descent is the standard SGD optimizer
learning_rate = 0.4f0
opt_state = Flux.setup(Flux.Descent(learning_rate), model)

# ---------------------------------------------------------
# 4. Training Loop with train!
# ---------------------------------------------------------
# We will track the parameter path to visualize the "noisy" convergence
history = []
epochs = 20

println("Starting Stochastic Gradient Descent...")

for epoch in 1:epochs
    # Flux.train! runs through the entire loader once (one epoch)
    # Since batchsize=5 and N=12, this will perform ~3 updates per epoch
    Flux.train!(loss_fn, model, loader, opt_state)

    # Track parameters after each epoch
    push!(history, (model[1].bias[1], model[1].weight[1]))

    if epoch % 5 == 0
        current_loss = loss_fn(model, x_train, y_train)
        println("Epoch $epoch: Loss = $current_loss")
    end
end

# ---------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------
# Plot A: The noisy path in parameter space
p1 = plot([h[1] for h in history], [h[2] for h in history],
              marker=:circle, color=:blue, label="SGD Path",
              title="SGD Parameter Trajectory",
              xlabel="Intercept (Bias)", ylabel="Slope (Weight)")

    # Plot B: Final Regression Fit
    x_plot = Float32.(collect(0:0.01:2)')
    p2 = scatter(x_train', y_train', label="Data", color=:red, title="Final SGD Fit")
    plot!(x_plot', model(x_plot)', label="Model", lw=2, color=:black)



    plot(p1, p2, layout=(1, 2), size=(900, 400))

