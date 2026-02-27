using Flux
using Statistics
using Plots

# ---------------------------------------------------------
# 1. Define the Model and Data
# ---------------------------------------------------------
# We use a Chain with one Dense layer to ensure model[1] works.
# Dense(1 => 1) represents y = wx + b (slope and intercept)
model = Chain(Dense(1 => 1))

# Synthetic data from the notebook
x_train = Float32.([0.03, 0.19, 0.34, 0.46, 0.78, 0.81, 1.08, 1.18, 1.39, 1.60, 1.65, 1.90]')
y_train = Float32.([0.67, 0.85, 1.05, 1.10, 1.40, 1.50, 1.30, 1.60, 1.70, 2.10, 2.30, 2.20]')

# ---------------------------------------------------------
# 2. Define Loss and Optimizer
# ---------------------------------------------------------
loss_fn(m, x, y) = Flux.mse(m(x), y)

# Descent(lr) is pure Stochastic Gradient Descent
learning_rate = 0.1
opt_state = Flux.setup(Flux.Descent(learning_rate), model)

# ---------------------------------------------------------
# 3. Training Loop with Parameter Tracking
# ---------------------------------------------------------
n_steps = 20
history = []

println("Starting Gradient Descent...")

for step in 1:n_steps
    # 1. Compute gradients
    val, grads = Flux.withgradient(m -> loss_fn(m, x_train, y_train), model)

    # 2. Update parameters
    Flux.update!(opt_state, model, grads[1])

    # 3. Save parameters (Fixing the MethodError by using the Chain index)
    # model[1] is the Dense layer.
    # .bias[1] is the intercept, .weight[1] is the slope.
    push!(history, (model[1].bias[1], model[1].weight[1]))

    if step % 5 == 0
        println("Step $step: Loss = $val")
    end
end

# ---------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------
# Plot A: Final Fit
x_plot = Float32.(collect(0:0.01:2)')
y_pred = model(x_plot)

p1 = scatter(x_train', y_train', label="Data", color=:red, title="Linear Regression Fit")
plot!(x_plot', y_pred', label="Flux Prediction", lw=2, color=:blue)
xlabel!("x")
ylabel!("y")

# Plot B: Optimization Path
# This visualizes how the weights moved toward the minimum
p2 = plot([h[1] for h in history], [h[2] for h in history],
              marker=:circle, label="Path", title="Parameter Space",
              xlabel="Intercept (Bias)", ylabel="Slope (Weight)")



    display(plot(p1, p2, layout=(1, 2), size=(900, 400)))




