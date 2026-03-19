using CSV
using DataFrames
using Flux
using Flux: train!
using Statistics
using Plots
using Random
using Distributions

# --- 1. Generate Synthetic Data ---
# Mimicking the shape: rows = samples, columns = genes
n_samples = 1000
n_features = 2000
println("Generating synthetic data ($n_samples samples, $n_features genes)...")

Random.seed!(42)
# Generate random data between 0 and 1 (simulating scaled RNAseq)
synthetic_data = rand(Uniform(0, 1), n_samples, n_features)

# Create DataFrame
pancan_rnaseq_df = DataFrame(synthetic_data, :auto)
# Add a sample ID column (simulating the index_col=0 in pandas)
insertcols!(pancan_rnaseq_df, 1, :sample_id => ["Sample_" * string(i) for i in 1:n_samples])

# --- 2. Preprocess for Flux ---
# Remove the ID column for numerical training
features = Matrix(pancan_rnaseq_df[:, 2:end])

# Transpose to get features x samples for Flux (input features are rows)
X = Float32.(transpose(features))

# --- 3. Train Test Split ---
Random.seed!(42) # For reproducibility
indices = randperm(n_samples)
train_idx = indices[1:Int(floor(0.9 * n_samples))]
test_idx = indices[Int(floor(0.9 * n_samples))+1:end]

X_train = X[:, train_idx]
X_test = X[:, test_idx]

println("Training shape: ", size(X_train))

# --- 4. Define Autoencoder Model ---
encoding_dim = 100
n_input_features = size(X, 1)

# Flux uses layers explicitly. Dense(input, output, activation)
model = Chain(
    Dense(n_input_features, encoding_dim, relu),
    Dense(encoding_dim, n_input_features, sigmoid)
)

# Loss function: Mean Squared Error
loss(x) = Flux.mse(model(x), x)

# Optimizer
opt = ADAM()
parameters = Flux.params(model)

# --- 5. Training ---# --- 5. Training (Modern Explicit Style) ---
epochs = 10
batch_size = 50

println("Starting training...")
train_loader = Flux.DataLoader(X_train, batchsize=batch_size, shuffle=true)

# Tracking loss
loss_history = []

# Define the optimizer explicitly
opt = Adam()
# State for the optimizer
opt_state = Flux.setup(opt, model)

for epoch in 1:epochs
    for batch in train_loader
        # Calculate gradients explicitly
        loss_val, grads = Flux.withgradient(model) do m
            Flux.mse(m(batch), batch)
        end

        # Update model parameters
        Flux.update!(opt_state, model, grads[1])
    end

    current_loss = loss(X_train)
    push!(loss_history, current_loss)
    println("Epoch $epoch, Loss: $current_loss")
end

# --- 6. Visualize Training ---
plot(loss_history, label="Training Loss", xlabel="Epochs", ylabel="MSE", title="Reconstruction Loss")

# --- 7. Reconstruction & Fidelity ---
println("Calculating reconstruction...")

# Reconstruct all data
reconstruction = model(X)

# Calculate difference (Fidelity)
fidelity = reconstruction - X

# Gene mean and abs sum (similar to pandas axis operations)
gene_mean = mean(fidelity, dims=2)
gene_abssum = sum(abs.(fidelity), dims=2) / n_samples

# Create summary dataframe
gene_summary = DataFrame(
    gene_mean = vec(gene_mean),
    gene_abs_sum = vec(gene_abssum)
)

println("Top reconstruction differences:")
println(first(sort(gene_summary, :gene_abs_sum, rev=true), 5))

# --- 8. Joint Plot ---
scatter(gene_summary.gene_mean, gene_summary.gene_abs_sum,
        xlabel="Gene Mean", ylabel="Gene Abs(Sum)",
        title="Reconstruction Fidelity", label="Genes")
