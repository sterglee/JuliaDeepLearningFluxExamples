using Flux
using Statistics
using Random

# ---------------------------------------------------------
# 1. Define the Network Architecture
# ---------------------------------------------------------
# D_i=4 inputs, D_1=5, D_2=2, D_3=4 hidden units, D_o=1 output
D_i, D_1, D_2, D_3, D_o = 4, 5, 2, 4, 1

# In Flux, Chain is used to stack layers.
# Dense(in => out, activation) handles the Omega (weight) and beta (bias) matrices.
model = Chain(
    Dense(D_i => D_1, relu),
    Dense(D_1 => D_2, relu),
    Dense(D_2 => D_3, relu),
    Dense(D_3 => D_o)
    )

# ---------------------------------------------------------
# 2. Generate Dummy Data
# ---------------------------------------------------------
n_data = 100
# Julia convention is (features, observations)
x_train = randn(Float32, D_i, n_data)
# Let's create a simple target (e.g., sum of inputs) for training demo
y_train = sum(x_train, dims=1)

# Flux uses DataLoaders to manage batches
loader = Flux.DataLoader((x_train, y_train), batchsize=16, shuffle=true)

# ---------------------------------------------------------
# 3. Training Setup
# ---------------------------------------------------------
# Select an Optimizer (Adam is a standard modern choice)
optim = Flux.setup(Flux.Adam(0.01), model)

# Define the Loss Function (Mean Squared Error)
loss_fn(m, x, y) = Flux.mse(m(x), y)

# ---------------------------------------------------------
# 4. The Training Loop (using train!)
# ---------------------------------------------------------
epochs = 50
println("Starting training...")

for epoch in 1:epochs
    # Flux.train! takes: loss function, model, data loader, and optim state
    Flux.train!(loss_fn, model, loader, optim)

    if epoch % 10 == 0
        current_loss = loss_fn(model, x_train, y_train)
        println("Epoch $epoch: Loss = $current_loss")
    end
end

# ---------------------------------------------------------
# 5. Inference
# ---------------------------------------------------------
# Generate a random test point
x_test = randn(Float32, D_i, 1)
y_pred = model(x_test)

println("\nTest Input:")
display(x_test)
println("Model Output:")
display(y_pred)

