using Flux
using Plots
using Statistics

# --- 0. Parameters ---
const NOISE_EMBEDDING_SIZE = 32
const MIN_FREQ = 1.0f0
const MAX_FREQ = 1000.0f0

# --- 1. Sinusoidal Embedding Function ---
# Converted from Python logic
function sinusoidal_embedding(x::AbstractArray{Float32})
    # Calculate frequencies in Float32
    # np.linspace(log(1), log(1000), 16) equivalent
    freq_range = range(log(MIN_FREQ), stop=log(MAX_FREQ), length=Int(NOISE_EMBEDDING_SIZE / 2))
    frequencies = exp.(Float32.(collect(freq_range)))

    angular_speeds = 2.0f0 * Float32(π) .* frequencies

    # Broadcast across the input dimensions
    # We use reshape to ensure angular_speeds aligns with the features
    # Julia Flux shape: (W, H, C, Batch)
    # x is expected to be the noise variance (1, 1, 1, Batch)
    angles = x .* reshape(angular_speeds, 1, 1, Int(NOISE_EMBEDDING_SIZE / 2), 1)

    return cat(sin.(angles), cos.(angles), dims=3)
end

# --- 2. Generation and Timing ---
function run_embedding_demo()
    println("Generating embeddings...")

    # Create a range of noise variances from 0 to 1
    y_range = Float32.(collect(0.0:0.01:0.99))

    # Time the embedding generation
    @time begin
        # Shape into (1, 1, 1, 100) to match Flux expectations
        input_data = reshape(y_range, 1, 1, 1, :)
        embeddings = sinusoidal_embedding(input_data)
    end

    # Reshape for visualization: (embedding_dim, noise_variance_steps)
    # embeddings is (1, 1, 32, 100) -> we want (32, 100)
    embedding_matrix = dropdims(embeddings, dims=(1, 2))

    # --- 3. Visualization ---
    # Replicating the pcolor plot from the notebook
    heatmap(y_range, 1:NOISE_EMBEDDING_SIZE, embedding_matrix,
            cmap=:coolwarm,
            xlabel="Noise Variance",
            ylabel="Embedding Dimension",
            title="Sinusoidal Noise Embedding (Julia)",
            xticks=0.0:0.1:1.0)
end

# Execute
run_embedding_demo()



