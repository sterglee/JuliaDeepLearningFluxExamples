using Flux, Optimisers, ProgressMeter, MLDatasets

# 1. Model & Data Setup
model = Chain(
    Dense(784 => 128, relu),
    Dense(128 => 10)
    )

train_data = MNIST(split=:train)
X_train = reshape(Float32.(train_data.features), 28*28, :)
y_train = Flux.onehotbatch(train_data.targets, 0:9)

# DataLoader handles batching and shuffling automatically
loader = Flux.DataLoader((X_train, y_train), batchsize=256, shuffle=true)

# 2. Modern Optimiser Setup
# This creates the state tree (momentums, etc.) required for Adam
opt_rule = Optimisers.Adam(3e-4)
opt_state = Optimisers.setup(opt_rule, model)

# 3. Define the Loss Function
# In modern Flux, the first argument to the loss inside train!
# is typically the model itself.
compute_loss(m, x, y) = Flux.logitcrossentropy(m(x), y)

epochs = 30
# 4. Training Loop
for epoch in 1:epochs
    @showprogress "Epoch $epoch: " for (x, y) in loader
        # We pass the batch as a single-element collection: [(x, y)]
        # This updates 'model' and 'opt_state' in-place/internally
        Flux.train!(compute_loss, model, [(x, y)], opt_state)
    end

    # Check progress
    loss = sum(compute_loss(model, x, y) for (x, y) in loader) / length(loader)
        println("Epoch $epoch Loss: $loss")
    end

    end


