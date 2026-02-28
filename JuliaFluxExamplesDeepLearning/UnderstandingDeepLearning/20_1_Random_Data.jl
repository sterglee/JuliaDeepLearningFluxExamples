using Flux
using Random
using Statistics
using Plots

# --- 1. Generate Synthetic Data ---
# Replicates the "Random Data" experiment from Notebook 20.1
function generate_data(n_samples=100, n_features=20)
    # Generate random features
    X = randn(Float32, n_features, n_samples)

    # True labels: simple linear separator with noise
    true_w = randn(Float32, 1, n_features)
    Y_true = (true_w * X .> 0) |> vec .|> Float32 # Binary labels

    # Random labels: completely uncorrelated with X
    Y_random = rand([0.0f0, 1.0f0], n_samples)

    return X, Y_true, Y_random
end

# --- 2. Build the Model ---
# A simple MLP that is "over-parameterized" relative to the small sample size
function build_model(n_features)
    return Chain(
        Dense(n_features, 512, relu),
        Dense(512, 512, relu),
        Dense(512, 1, sigmoid)
        )
end

# --- 3. Modern Training Function ---
function train_model!(model, X, Y; n_epochs=200, lr=0.01f0)
    # Setup modern optimizer state
    opt_state = Flux.setup(Adam(lr), model)
    loss_history = Float32[]

    for epoch in 1:n_epochs
        # Calculate gradients and loss
        loss_val, grads = Flux.withgradient(model) do m
            y_hat = m(X)
            # Reshape Y from (100,) to (1, 100)
            Flux.logitbinarycrossentropy(y_hat, reshape(Y, 1, :))
        end

        # Apply updates to the model
        Flux.update!(opt_state, model, grads[1])

        push!(loss_history, loss_val)

        if epoch % 50 == 0
            println("Epoch $epoch: Loss = $loss_val")
        end
    end
    return loss_history
end

# --- 4. Running the Experiment ---
n_features = 20
n_samples = 100
X, Y_true, Y_random = generate_data(n_samples, n_features)

println("Training on TRUE labels...")
model_true = build_model(n_features)
history_true = train_model!(model_true, X, Y_true)

println("\nTraining on RANDOM labels (Memorization test)...")
model_random = build_model(n_features)
history_random = train_model!(model_random, X, Y_random)

# --- 5. Visualization ---


plot(history_true, label="True Labels", lw=2, title="Generalization vs. Memorization",
     xlabel="Epoch", ylabel="Loss", color=:red)
plot!(history_random, label="Random Labels", lw=2, color=:blue)

