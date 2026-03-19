using CSV
using DataFrames
using Flux
using Statistics
using Plots
using Random
using Distributions

# --- 1. Model Definition ---
struct VAE
    encoder_base
    mu_head
    logvar_head
    decoder
end

Flux.@functor VAE

function VAE(input_dim, hidden_dim, latent_dim)
    return VAE(
        Dense(input_dim, hidden_dim, relu),  # Shared Encoder
        Dense(hidden_dim, latent_dim),       # Mu
        Dense(hidden_dim, latent_dim),       # Log-variance
        Chain(Dense(latent_dim, hidden_dim, relu),
              Dense(hidden_dim, input_dim, sigmoid))
        )
end

function sample_z(mu, logvar)
    epsilon = randn(Float32, size(mu))
    return mu + exp.(logvar * 0.5f0) .* epsilon
end

function (m::VAE)(x)
    h = m.encoder_base(x)
    mu, logvar = m.mu_head(h), m.logvar_head(h)
    z = sample_z(mu, logvar)
    return m.decoder(z), mu, logvar
end

# --- 2. Loss Function (ELBO) ---
function vae_loss(model, x, beta=0.1f0)
    x_hat, mu, logvar = model(x)
    # Reconstruction loss
    recon_loss = Flux.mse(x_hat, x, agg=sum)
    # KL Divergence
    kl_loss = -0.5f0 * sum(1f0 .+ logvar .- mu.^2 .- exp.(logvar))
    return (recon_loss + beta * kl_loss) / size(x, 2)
end

# --- 3. Main Execution Block ---
function main()
    # 3a. Generate/Load Data
    println("Loading data...")
    n_samples, n_features = 1000, 2000
    Random.seed!(42)
    # Simulating your 'pancan_scaled_zeroone_rnaseq.tsv'
    data_mat = Float32.(rand(n_samples, n_features))

    # Flux: (Features, Samples)
    X = transpose(data_mat) |> collect

    # Train/Test Split
    train_idx = 1:Int(0.9 * n_samples)
    X_train = X[:, train_idx]
    X_test  = X[:, (Int(0.9 * n_samples) + 1):end]

    # 3b. Initialize Model
    latent_dim = 100
    hidden_dim = 256
    model = VAE(n_features, hidden_dim, latent_dim)
    opt_state = Flux.setup(Adam(0.001), model)
    train_loader = Flux.DataLoader(X_train, batchsize=50, shuffle=true)

    # 3c. Training Loop
    epochs = 10
    println("Starting VAE Training ($epochs epochs)...")
    for epoch in 1:epochs
        total_loss = 0f0
        for batch in train_loader
            l, grads = Flux.withgradient(model) do m
                vae_loss(m, batch)
            end
            Flux.update!(opt_state, model, grads[1])
            total_loss += l
        end
        println("Epoch $epoch | Avg Loss: $(total_loss / length(train_loader))")
    end

    # 3d. Reconstruction & Fidelity
    println("Calculating reconstruction fidelity...")
    X_hat, mu_latent, _ = model(X)

    # Fidelity calculation (diff between original and reconstructed)
    fidelity = X_hat .- X
    gene_mean = vec(mean(fidelity, dims=2))
    gene_abs_sum = vec(sum(abs.(fidelity), dims=2) ./ n_samples)

    # 3e. Visualization
    p1 = scatter(gene_mean, gene_abs_sum,
                 xlabel="Gene Mean Diff", ylabel="Gene Abs(Sum) Diff",
                 title="VAE Fidelity", alpha=0.4, label="Genes")

    # Latent space visualization (first two dimensions of mu)
    p2 = scatter(mu_latent[1, :], mu_latent[2, :],
                 xlabel="Z1", ylabel="Z2",
                 title="Latent Space (μ)", label="Samples", color=:viridis)

    plot(p1, p2, layout=(1,2), size=(900, 400))
end

# Run the full pipeline
main()

