using Flux
using Flux: onehotbatch, onecold, logitcrossentropy, Adam
using MLDatasets
using Statistics
using Random

# ---------------------------------------------------------
# 1. Define LeNet Architecture
# ---------------------------------------------------------
function build_lenet(num_classes)
    return Chain(
        # Input: (28, 28, 1, Batch)
        Conv((5,5), 1=>20, relu),        # -> (24, 24, 20, Batch)
        MaxPool((2,2)),                  # -> (12, 12, 20, Batch)
        Conv((5,5), 20=>50, relu),       # -> (8, 8, 50, Batch)
        MaxPool((2,2)),                  # -> (4, 4, 50, Batch)
        Flux.flatten,                    # -> (800, Batch)
        Dense(800, 500, relu),
        Dense(500, num_classes)          # Output logits (no softmax)
    )
end

# ---------------------------------------------------------
# 2. Data Preparation
# ---------------------------------------------------------
# Load MNIST (requires internet connection for first run)
train_data = MNIST(split=:train)
test_data  = MNIST(split=:test)

# CRITICAL FIX: Reshape to 4D (Width, Height, Channels, Batch)
# Original: (28, 28, 60000) -> New: (28, 28, 1, 60000)
train_x = reshape(Float32.(train_data.features), 28, 28, 1, :)
test_x  = reshape(Float32.(test_data.features), 28, 28, 1, :)

classes = 0:9
train_y_oh = onehotbatch(train_data.targets, classes)
test_y_oh  = onehotbatch(test_data.targets,  classes)

# Create DataLoader
batch_size = 128
train_loader = Flux.DataLoader((train_x, train_y_oh), batchsize=batch_size, shuffle=true)

# ---------------------------------------------------------
# 3. Model & Optimizer Setup
# ---------------------------------------------------------
model = build_lenet(length(classes))
# Setup the optimizer state (Modern Flux style)
opt_state = Flux.setup(Adam(0.001), model)

# ---------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------
epochs = 5

println("Starting training...")
for epoch in 1:epochs
    # Train over batches
    for (x, y) in train_loader
        # Calculate gradients of the loss with respect to the model
        grads = Flux.gradient(model) do m
            y_hat = m(x)
            logitcrossentropy(y_hat, y)
        end
        # Update parameters
        Flux.update!(opt_state, model, grads[1])
    end

    # Evaluate Accuracy on full test set after each epoch
    y_hat_test = model(test_x)
    acc = mean(onecold(y_hat_test, classes) .== test_data.targets)
    
    println("Epoch $epoch: Test Accuracy = $(round(acc * 100, digits=2))%")
end

# ---------------------------------------------------------
# 5. Final Evaluation
# ---------------------------------------------------------
final_output = model(test_x)
final_loss = logitcrossentropy(final_output, test_y_oh)
println("\n--- Final Results ---")
println("Test Loss: $(round(final_loss, digits=4))")
println("Final Accuracy: $(round(mean(onecold(final_output, classes) .== test_data.targets)*100, digits=2))%")

