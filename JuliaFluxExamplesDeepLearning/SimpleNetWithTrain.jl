using Flux, Optimisers, ProgressMeter, MLDatasets

# 1. Setup Model
model = Chain(
    Dense(784 => 128, relu),
    Dense(128 => 10)
    )

# 2. Setup Data
train_data = MNIST(split=:train)
X_train = reshape(Float32.(train_data.features), 28*28, :)
y_train = Flux.onehotbatch(train_data.targets, 0:9)
loader = Flux.DataLoader((X_train, y_train), batchsize=256, shuffle=true)

# 3. Optimiser Setup
opt_state = Optimisers.setup(Optimisers.Adam(3e-4), model)

# 4. Training Loop
loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

epochs = 10
for epoch in 1:epochs
    @showprogress "Epoch $epoch: " for (x, y) in loader
        # FIX: Just call train! without 'opt_state, model ='
        # This updates the 'model' and 'opt_state' objects directly
        Flux.train!(loss_fn, model, [(x, y)], opt_state)
    end

    # Calculate average loss for the epoch
    total_loss = sum(loss_fn(model, x, y) for (x, y) in loader) / length(loader)
        println("Epoch $epoch: avg loss = $total_loss")
    end


