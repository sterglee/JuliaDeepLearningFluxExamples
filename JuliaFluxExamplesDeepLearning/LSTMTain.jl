using Flux, Optimisers, ProgressMeter

# 1. Setup Model
# We use a custom function or a Chain that specifically picks the last time step
model = Chain(
    LSTM(1 => 32),
    # Take the last time step: (32, 20, 32) -> (32, 32)
    x -> x[:, end, :],
    Dense(32 => 1)
    )

# 2. Data (Features=1, SeqLen=20, Batch=32)
X = randn(Float32, 1, 20, 32)
Y = randn(Float32, 1, 32) # Target is now (1, 32), matching the model output

loader = [(X, Y)]

# 3. Optimiser
opt_state = Optimisers.setup(Optimisers.Adam(1e-3), model)

# 4. Training
function loss_fn(m, x, y)
    Flux.reset!(m)
    y_hat = m(x)
    return Flux.mse(y_hat, y)
end

# Training Loop
for epoch in 1:10
    for (batch_x, batch_y) in loader
        Flux.train!(loss_fn, model, [(batch_x, batch_y)], opt_state)
    end
    println("Epoch $epoch Loss: $(loss_fn(model, X, Y))")
end

