using Flux
using Statistics
using Random
using Plots

# ---------------------------------------------------------
# 1. Define the Classification Network
# ---------------------------------------------------------
# Architecture: 1 input, 10 hidden units (ReLU), 1 output
# Note: In Flux, we usually output "logits" (raw values)
# and let the loss function handle the sigmoid mapping.
D_i, D_h, D_o = 1, 10, 1

model = Chain(
    Dense(D_i => D_h, relu),
    Dense(D_h => D_o)
    )

# ---------------------------------------------------------
# 2. Data Generation
# ---------------------------------------------------------
# Binary data: Input x and labels y ∈ {0, 1}
x_train = Float32.([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]')
y_train = Float32.([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]') # Simple step-like classification

loader = Flux.DataLoader((x_train, y_train), batchsize=2, shuffle=true)

# ---------------------------------------------------------
# 3. Binary Cross-Entropy Loss
# ---------------------------------------------------------
# logitbinarycrossentropy(y_hat, y) takes raw model outputs (logits)
# and computes the BCE loss using the sigmoid function internally.
loss_fn(m, x, y) = Flux.logitbinarycrossentropy(m(x), y)

# ---------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------
opt_state = Flux.setup(Flux.Adam(0.01), model)
epochs = 500

println("Training to minimize Binary Cross-Entropy...")
for epoch in 1:epochs
    Flux.train!(loss_fn, model, loader, opt_state)

    if epoch % 100 == 0
        current_loss = loss_fn(model, x_train, y_train)
        println("Epoch $epoch: BCE Loss = $current_loss")
    end
end

# ---------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------
x_plot = Float32.(collect(0:0.001:1)')
# To visualize the probability λ, we pass logits through sigmoid
probs = sigmoid.(model(x_plot))



scatter(x_train', y_train', label="Training Labels (0 or 1)", color=:red, marker=:copy)
display(plot!(x_plot', probs', label="Predicted Probability (λ)", lw=3, color=:blue))
hline!([0.5], linestyle=:dash, label="Decision Threshold", color=:black)
xlabel!("Input x")
ylabel!("Probability P(y=1|x)")
title!("Notebook 5.2: Binary Cross-Entropy in Flux")

