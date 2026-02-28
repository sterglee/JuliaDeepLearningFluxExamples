using Flux
using Random
using Statistics
using Plots

# --- 1. Data Generation ---
# Replicates the synthetic data from Notebook 20.2
function generate_data(n_samples=50, n_features=2)
    Random.seed!(42)
    X = randn(Float32, n_features, n_samples)
    # Convert elements to Float32, then transpose to get a (1, N) matrix
    Y = Float32.(sin.(X[1, :] .* X[2, :]))'
    return X, Y
end

# --- 2. Model Factory ---
# Creates models with varying depths to replicate the experiment in Figure 20.2
function build_model(n_features, n_layers, width=16)
    layers = []
    # Input layer
    push!(layers, Dense(n_features, width, relu))

    # Hidden layers
    for _ in 1:(n_layers - 1)
        push!(layers, Dense(width, width, relu))
    end

    # Output layer
    push!(layers, Dense(width, 1))

    return Chain(layers...)
end

# --- 3. Modern Training Function (Full Batch) ---
function train_full_batch!(model, X, Y; n_epochs=5000, lr=0.01f0)
    # Modern Flux optimizer setup
    opt_state = Flux.setup(Adam(lr), model)
    loss_history = Float32[]

    for epoch in 1:n_epochs
        # Full Batch Gradient Calculation
        val, grads = Flux.withgradient(model) do m
            y_hat = m(X)
            # Mean Squared Error for regression
            Flux.mse(y_hat, Y)
        end

        # Apply updates to the model
        Flux.update!(opt_state, model, grads[1])

        push!(loss_history, val)

        if epoch % 1000 == 0
            println("Epoch $epoch: Loss = $val")
        end
    end
    return loss_history
end

# --- 4. Running the Depth Experiment ---
X, Y = generate_data()

depths = [1, 2, 3, 4]
results = Dict()

println("Starting depth experiment...")
for d in depths
    println("Training model with depth: $d")
    model = build_model(2, d)
    results[d] = train_full_batch!(model, X, Y)
end

# --- 5. Visualization ---


p = plot(title="Full Batch Gradient Descent: Effect of Depth",
         xlabel="Epoch", ylabel="MSE Loss", yaxis=:log)

for d in depths
    plot!(p, results[d], label="Depth $d")
end
display(p)

