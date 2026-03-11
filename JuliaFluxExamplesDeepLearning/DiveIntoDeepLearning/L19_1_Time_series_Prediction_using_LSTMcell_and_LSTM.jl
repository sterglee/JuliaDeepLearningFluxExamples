using Flux
using Statistics
using Plots
using Random

# 1. Data Generation
T = 1000
embedding = 4
time_steps = collect(0.0:T-1)
# Sine wave with noise (equivalent to the PyTorch snippet)
x = sin.(0.01 .* time_steps) .+ 0.2 .* randn(T)
plot(time_steps, x, title="Generated Sine Wave", label="Data")

# 2. Dataset Preparation
# We create features (previous 'embedding' points) and targets (next point)
features = []
labels = []
for i in 1:(T - embedding)
    push!(features, x[i : i + embedding - 1])
    push!(labels, x[i + embedding])
end

# Convert to Float32 for GPU compatibility and performance
features = hcat(features...)  # Shape: (embedding, 996)
labels = Float32.(labels)      # Shape: (996,)

# Split into Train and Test (e.g., first 600 for training)
train_split = 600
x_train = Float32.(features[:, 1:train_split])
y_train = labels[1:train_split]
x_test = Float32.(features[:, train_split+1:end])
y_test = labels[train_split+1:end]

# 3. Model Definition (Modern Flux style)
# Equivalent to: nn.LSTM(4, 100) -> nn.Linear(100, 1)
model = Chain(
    LSTM(embedding => 100), 
    Dense(100 => 1)
)

# 4. Training Configuration
loss(m, x, y) = Flux.mse(m(x), y')
opt_state = Flux.setup(Flux.Adam(0.01), model)

# 5. Training Loop
epochs = 10
batch_size = 16

println("Starting training...")
for epoch in 1:epochs
    # Simple batching implementation
    for i in 1:batch_size:size(x_train, 2)
        idx = i:min(i+batch_size-1, size(x_train, 2))
        batch_x = x_train[:, idx]
        batch_y = y_train[idx]
        
        # In RNNs, we often reset state between independent sequences or batches
        Flux.reset!(model) 
        
        grads = Flux.gradient(model) do m
            loss(m, batch_x, batch_y)
        end
        Flux.update!(opt_state, model, grads[1])
    end
    
    # Calculate Epoch Loss
    Flux.reset!(model)
    current_loss = loss(model, x_train, y_train)
    println("Epoch $epoch, Loss: $current_loss")
end

# 6. Evaluation/Prediction
Flux.reset!(model)
preds = model(x_test)
test_loss = Flux.mse(preds, y_test')
println("Test Loss: $test_loss")

# Plot Results
plot(y_test, label="Ground Truth", color=:blue)
plot!(preds', label="Predictions", color=:red, linestyle=:dash)

