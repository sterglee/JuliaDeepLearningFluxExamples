using Flux
using Statistics
using Random
using Plots

# ---------------------------------------------------------
# 1. Define the Multiclass Network
# ---------------------------------------------------------
# Architecture: 1 input, 10 hidden units (ReLU), 3 outputs (classes)
# The notebook uses 3 output classes to demonstrate the Categorical distribution.
D_i, D_h, D_o = 1, 10, 3

model = Chain(
    Dense(D_i => D_h, relu),
    Dense(D_h => D_o)
    )

# ---------------------------------------------------------
# 2. Data Generation
# ---------------------------------------------------------
# Multiclass data: Input x and labels y ∈ {1, 2, 3}
# Flux expects labels as a 1D vector of integers or a one-hot matrix.
x_train = Float32.(collect(0.1:0.1:0.9)')
y_labels = [1, 1, 1, 2, 2, 2, 3, 3, 3]
y_train = Flux.onehotbatch(y_labels, 1:3) # Converts to 3x9 one-hot matrix

loader = Flux.DataLoader((x_train, y_train), batchsize=3, shuffle=true)

# ---------------------------------------------------------
# 3. Multiclass Cross-Entropy Loss
# ---------------------------------------------------------
# logitcrossentropy(y_hat, y) takes raw logits and computes
# the softmax internally to find the categorical probabilities.
loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

# ---------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------
opt_state = Flux.setup(Flux.Adam(0.01), model)
epochs = 500

println("Training to minimize Multiclass Cross-Entropy...")
for epoch in 1:epochs
    Flux.train!(loss_fn, model, loader, opt_state)

    if epoch % 100 == 0
        current_loss = loss_fn(model, x_train, y_train)
        println("Epoch $epoch: Loss = $current_loss")
    end
end

# ---------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------
x_plot = Float32.(collect(0:0.001:1)')
# Pass logits through softmax to get probabilities for each class
probs = softmax(model(x_plot))



display(plot(x_plot', probs[1, :], label="Prob Class 1", lw=2))
plot!(x_plot', probs[2, :], label="Prob Class 2", lw=2)
plot!(x_plot', probs[3, :], label="Prob Class 3", lw=2)
scatter!(x_train', [y == 1 for y in y_labels], label="Data C1", color=:blue)
    scatter!(x_train', [y == 2 for y in y_labels], label="Data C2", color=:orange)
        scatter!(x_train', [y == 3 for y in y_labels], label="Data C3", color=:green)
            xlabel!("Input x")
            ylabel!("Probability P(y=k|x)")
            title!("Notebook 5.3: Multiclass Cross-Entropy in Flux")
