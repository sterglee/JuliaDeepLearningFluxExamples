using Flux
using Statistics
using Plots
using LinearAlgebra # For norm()

# ---------------------------------------------------------
# 1. Define Model and Data
# ---------------------------------------------------------
# Synthetic data similar to the notebook's "Gabor" or 1D examples
x_train = Float32.([0.03, 0.19, 0.34, 0.46, 0.78, 0.81, 1.08, 1.18, 1.39, 1.60, 1.65, 1.90]')
y_train = Float32.([0.67, 0.85, 1.05, 1.10, 1.40, 1.50, 1.30, 1.60, 1.70, 2.10, 2.30, 2.20]')

# Use a slightly more complex model to see the effect of regularization
model = Chain(
    Dense(1 => 10, relu),
    Dense(10 => 1)
    )

# ---------------------------------------------------------
# 2. Define Loss with L2 Regularization
# ---------------------------------------------------------
# lambda (λ) is the regularization strength
λ = 0.1f0

function loss_l2(m, x, y)
    # Standard MSE loss
    mse_loss = Flux.mse(m(x), y)

    # L2 Penalty: sum of squares of all weights
    # We use Flux.params(m) to grab all parameters, then filter for weights if desired
    # or simply penalize all parameters (weights and biases)
    l2_penalty = sum(p -> sum(abs2, p), Flux.params(m))

    return mse_loss + λ * l2_penalty
end

# ---------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------
# Method A: Manual penalty in loss (shown above)
opt_state = Flux.setup(Flux.Adam(0.01), model)

# Method B: Using Flux's built-in WeightDecay (Modern approach)
# This automatically modifies the gradients to include the L2 term
# opt_state = Flux.setup(OptimiserChain(WeightDecay(λ), Adam(0.01)), model)

epochs = 500
for epoch in 1:epochs
    # We use the loss_l2 defined above
    val, grads = Flux.withgradient(m -> loss_l2(m, x_train, y_train), model)
    Flux.update!(opt_state, model, grads[1])

    if epoch % 100 == 0
        println("Epoch $epoch: Loss = $val")
    end
end

# ---------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------
x_plot = Float32.(collect(0:0.01:2)')
y_pred = model(x_plot)



p1 = scatter(x_train', y_train', label="Data", color=:red, title="L2 Regularized Model (λ=$λ)")
plot!(x_plot', y_pred', label="Prediction", lw=3, color=:blue)

# Visualize weight distribution
all_weights = vcat([vec(p) for p in Flux.params(model)]...)
    p2 = histogram(all_weights, bins=20, title="Weight Distribution", label="Weights")
    xlabel!("Value")

    display(plot(p1, p2, layout=(1, 2), size=(900, 400)))

