using Flux
using Statistics
using Plots
using Random
using Distributions
using Transformers
using Transformers.Layers

# --- 1. Setup Synthetic Data ---
n_samples = 100
n_features = 512
Random.seed!(42)

# Generate synthetic RNAseq-like data [0, 1]
synthetic_data = rand(Uniform(0, 1), n_samples, n_features)

# Transformer Shape: (Feature_Dim, Seq_Len, Batch)
# We treat each gene as 1 feature in a sequence of length 512
X = Float32.(reshape(transpose(synthetic_data), (1, n_features, n_samples)))

# --- 2. Architecture Constants ---
const H_DIM = 32      # Hidden embedding dimension
const N_HEADS = 4     # Multi-head attention heads
const H_HEAD = 8      # Dimension per head (Must be H_DIM / N_HEADS)
const H_FF = 64       # Feed-forward intermediate size

# --- 3. Define the Transformer Model ---
struct GeneTransformer
    embed          # Projects 1 feature to H_DIM
    pos_enc        # Learned Positional Encoding
    transformer    # TransformerBlock
    project_back   # Projects H_DIM back to 1 feature
end

Flux.@functor GeneTransformer

function GeneTransformer(seq_len)
    # CORRECT ARGUMENT ORDER: (n_heads, head_size, hidden_size, intermediate_size)
    t_block = TransformerBlock(N_HEADS, H_HEAD, H_DIM, H_FF)

    return GeneTransformer(
        Dense(1, H_DIM),
        Flux.randn32(H_DIM, seq_len, 1), # (32, 512, 1) broadcasts over batch
        t_block,
        Dense(H_DIM, 1)
        )
end

function (m::GeneTransformer)(x)
    # 1. Embedding + Position (Flux Dense handles 3D input automatically)
    h = m.embed(x) .+ m.pos_enc     # (32, 512, Batch)

    # 2. Self-Attention Block
    # Requires (hidden_state, attention_mask); pass nothing for full attention
    h = m.transformer(h, nothing)   # (32, 512, Batch)

    # 3. Output Projection
    return m.project_back(h)        # (1, 512, Batch)
end

# --- 4. Training Pipeline ---
model = GeneTransformer(n_features)
opt_state = Flux.setup(Adam(0.001), model)
train_loader = Flux.DataLoader(X, batchsize=16, shuffle=true)

println("Training Transformer Autoencoder...")
history = []
for epoch in 1:10
    total_loss = 0f0
    for batch in train_loader
        loss_val, grads = Flux.withgradient(model) do m
            pred = m(batch)
            Flux.mse(pred, batch)
        end
        Flux.update!(opt_state, model, grads[1])
        total_loss += loss_val
    end
    push!(history, total_loss / length(train_loader))
    println("Epoch $epoch | Loss: $(history[end])")
end

# --- 5. Reconstruction & Fidelity ---
reconstruction = model(X)

# Calculate fidelity (Transposing back to Sample x Gene for analysis)
X_orig_flat = transpose(dropdims(X, dims=1))
X_rec_flat  = transpose(dropdims(reconstruction, dims=1))

diff = X_rec_flat .- X_orig_flat
gene_mean = vec(mean(diff, dims=1))
gene_abs_sum = vec(sum(abs.(diff), dims=1) ./ n_samples)

# --- 6. Visualization ---
p1 = plot(history, title="Training Loss", xlabel="Epoch", ylabel="MSE")
p2 = scatter(gene_mean, gene_abs_sum,
             xlabel="Gene Mean Diff", ylabel="Gene Abs(Sum) Diff",
             title="Fidelity JointPlot", alpha=0.5)

plot(p1, p2, layout=(2,1), size=(800, 700))

