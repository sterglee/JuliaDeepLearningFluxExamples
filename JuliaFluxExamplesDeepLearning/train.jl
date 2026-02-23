using Flux, Optimisers, ProgressMeter, MLDatasets

# 1. Setup Model
model = Chain(
    Dense(784 => 128, relu),
    Dense(128 => 10)
    )

# 2. Setup Data
train_data = MNIST(split=:train)
X_train = reshape(Float32.(train_data.features), 28*28, :)
# Note: MNIST targets are 0-9, so we use 0:9 directly for onehot
y_train = Flux.onehotbatch(train_data.targets, 0:9)

loader = Flux.DataLoader((X_train, y_train), batchsize=256, shuffle=true)

# 3. Modern Optimiser Setup
# Optimisers.setup initializes the state (momenta, etc.) for this specific model
opt_state = Optimisers.setup(Optimisers.Adam(3e-4), model)

epochs = 30

# 4. Training Loop
for epoch in 1:epochs
    total_loss = 0.0f0

    @showprogress "Epoch $epoch: " for (x, y) in loader
        # Gradient of the model itself, not Flux.params
        loss, grads = Flux.withgradient(model) do m
            y_hat = m(x)
            Flux.logitcrossentropy(y_hat, y)
        end

        # Update both the model parameters and the optimiser state
        opt_state, model = Optimisers.update(opt_state, model, grads[1])

        total_loss += loss * size(x, 2)
    end

    println("Avg loss: $(total_loss / size(X_train, 2))")
end

