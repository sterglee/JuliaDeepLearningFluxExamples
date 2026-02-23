using Flux, Optimisers, ProgressMeter, MLDatasets

# 1. Setup Model (CNN)
# MNIST is 28x28x1. Output is 10 classes.
model = Chain(
    # Conv((filter_size), in_channels => out_channels, activation)
    Conv((3, 3), 1 => 16, relu, pad=1),
    MaxPool((2, 2)),
    Conv((3, 3), 16 => 32, relu, pad=1),
    MaxPool((2, 2)),
    # Flatten the 4D tensor for the Dense layers
    Flux.flatten,
    Dense(7 * 7 * 32 => 128, relu),
    Dense(128 => 10)
    )

# 2. Prepare Data
train_data = MNIST(split=:train)
# Reshape to (Width, Height, Channels, Batch)
X_train = reshape(Float32.(train_data.features), 28, 28, 1, :)
y_train = Flux.onehotbatch(train_data.targets, 0:9)

loader = Flux.DataLoader((X_train, y_train), batchsize=128, shuffle=true)

# 3. Modern Optimiser Setup
opt_state = Optimisers.setup(Optimisers.Adam(3e-4), model)

# 4. Training
loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

epochs = 5
for epoch in 1:epochs
    @showprogress "Epoch $epoch: " for (x, y) in loader
        # Mutating call: updates model and opt_state in-place
        Flux.train!(loss_fn, model, [(x, y)], opt_state)
    end

    # Simple Accuracy Check
    acc = sum(Flux.onecold(model(X_train)) .== Flux.onecold(y_train)) / size(X_train, 4)
    println("Epoch $epoch complete. Train Accuracy: $(round(acc * 100, digits=2))%")
end


