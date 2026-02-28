using Flux
using CUDA
using Random
using Statistics
using Plots

# --- 1. Environment & GPU Check ---
# Check if a GPU is available, otherwise fallback to CPU
const DEVICE = CUDA.functional() ? gpu : cpu
println("Using device: ", CUDA.functional() ? "GPU (CUDA)" : "CPU")

# --- 2. Data Generation ---
function generate_data(n_samples=50, n_features=2)
    Random.seed!(42)
    X = randn(Float32, n_features, n_samples)
    # Target: non-linear relationship
    Y = Float32.(sin.(X[1, :] .* X[2, :]))'

    # Move data to the selected device (GPU/CPU)
    return X |> DEVICE, Y |> DEVICE
end

# --- 3. Model Factory ---
function build_model(n_features, n_layers, width=16)
    layers = []
    push!(layers, Dense(n_features, width, relu))
    for _ in 1:(n_layers - 1)
        push!(layers, Dense(width, width, relu))
    end
    push!(layers, Dense(width, 1))

    # Construct the Chain and move all parameters to the GPU/CPU
    return Chain(layers...) |> DEVICE
end

# --- 4. Modern Training Function (Full Batch) ---
function train_full_batch!(model, X, Y; n_epochs=5000, lr=0.01f0)
    # Modern Flux optimizer setup
    opt_state = Flux.setup(Adam(lr), model)
    loss_history = Float32[]

    for epoch in 1:n_epochs
        # Full Batch Gradient Calculation on GPU
        loss_val, grads = Flux.withgradient(model) do m
            y_hat = m(X)
            Flux.mse(y_hat, Y)
        end

        # Apply updates
        Flux.update!(opt_state, model, grads[1])

        # Move loss to CPU for logging
        push!(loss_history, cpu(loss_val))

        if epoch % 1000 == 0
            println("Epoch $epoch: Loss = $loss_val")
        end
    end
    return loss_history
end

# --- 5. Running the Experiment ---
X, Y = generate_data()
depths = [1, 2, 3, 4]
results = Dict()

for d in depths
    println("\nTraining model with depth: $d on GPU...")
    model = build_model(2, d)
    results[d] = train_full_batch!(model, X, Y)
end

# --- 6. Visualization ---

p = plot(title="Full Batch Gradient Descent (GPU)",
         xlabel="Epoch", ylabel="MSE Loss", yaxis=:log)

for d in depths
    plot!(p, results[d], label="Depth $d")
end
display(p)

