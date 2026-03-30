using HTTP, Flux, Statistics, Random, LinearAlgebra, GLMakie, Printf
using Flux: onehotbatch, DataLoader, onecold, sigmoid, logitcrossentropy

# --- CONSTANTS ---
const ALPHABET = ['a', 'g', 'c', 't']
const SEQ_LEN = 57
const LATENT_DIM = 10

# ------------------------------------------------------------
# 1. DATA LOADING
# ------------------------------------------------------------
function download_and_preprocess()
    println("Fetching dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data"
    response = HTTP.get(url)
    raw_lines = split(strip(String(response.body)), "\n")

    X_promoters, X_all, Y_all = [], [], []

    for line in raw_lines
        parts = split(line, ",")
        if length(parts) < 3 continue end

        label = strip(parts[1]) == "+" ? 1 : 2
        seq = lowercase(replace(strip(parts[3]), r"\s+" => ""))
        if length(seq) != 57 continue end

        encoded = Float32.(onehotbatch(collect(seq), ALPHABET))
        encoded_t = collect(encoded') # (57, 4)

        push!(X_all, encoded_t)
        push!(Y_all, onehotbatch(label, 1:2))

        if label == 1
            push!(X_promoters, encoded_t)
        end
    end
    return cat(X_promoters..., dims=3), cat(X_all..., dims=3), cat(Y_all..., dims=2)
end

# ------------------------------------------------------------
# 2. MODELS (Oracle, Generator, Discriminator)
# ------------------------------------------------------------
function build_oracle()
    return Chain(
        Conv((7,), 4 => 16, relu, pad=SamePad()),
        Flux.flatten,
        Dense(57 * 16, 2)
        )
end

function build_generator()
    return Chain(
        Dense(LATENT_DIM, 128, relu),
        Dense(128, 57 * 4),
        x -> reshape(x, 57, 4, :),
        x -> softmax(x, dims=2)
        )
end

function build_discriminator()
    return Chain(
        Conv((7,), 4 => 16, leakyrelu, pad=SamePad()),
        Flux.flatten,
        Dense(57 * 16, 1),
        sigmoid
        )
end

# ------------------------------------------------------------
# 3. PERFORMANCE & PLOTTING
# ------------------------------------------------------------
function plot_gan_results(history_prec, history_loss, gen_sample)
    fig = Figure(size = (1200, 800))

    # Plot 1: Oracle Precision
    ax1 = Axis(fig[1, 1], title="Oracle Precision (%)", xlabel="Evaluation Point", ylabel="% Real")
    lines!(ax1, history_prec, color=:blue, linewidth=2)

    # Plot 2: Discriminator Loss
    ax2 = Axis(fig[1, 2], title="Discriminator Loss", xlabel="Evaluation Point", ylabel="Loss")
    lines!(ax2, history_loss, color=:red, linewidth=2)

    # Plot 3: Probability Heatmap of a Generated Sample
    ax3 = Axis(fig[2, 1:2], title="Synthetic Promoter Probability Map (Generator Confidence)",
               xticks=(1:57, string.(1:57)), yticks=(1:4, ["A","G","C","T"]))
    hm = heatmap!(ax3, 1:57, 1:4, gen_sample[:, :, 1]', colormap=:magma)
    Colorbar(fig[2, 3], hm)

    save("gan_performance.png", fig)
    println("\n[System] Visualizations saved to: gan_performance.png")
    display(fig)
end

# ------------------------------------------------------------
# 4. MAIN EXECUTION
# ------------------------------------------------------------
function main()
    Random.seed!(42)
    X_promoters, X_all, Y_all = download_and_preprocess()

    # 1. Train Oracle Judge
    oracle = build_oracle()
    opt_o = Flux.setup(Adam(0.001), oracle)
    println("Training Oracle Judge...")
    for epoch in 1:50
        for (x, y) in DataLoader((X_all, Y_all), batchsize=16, shuffle=true)
            grads = Flux.gradient(m -> logitcrossentropy(m(x), y), oracle)
            Flux.update!(opt_o, oracle, grads[1])
        end
    end

    # 2. Initialize GAN
    gen = build_generator()
    disc = build_discriminator()
    opt_g = Flux.setup(Adam(0.0002), gen)
    opt_d = Flux.setup(Adam(0.0002), disc)
    loader = DataLoader(X_promoters, batchsize=16, shuffle=true)

    history_prec = Float32[]
    history_loss = Float32[]

    println("\n--- Training GAN ---")
    for epoch in 1:1000
        running_loss = 0.0f0
        for x_real in loader
            batch_sz = size(x_real, 3)
            z = randn(Float32, LATENT_DIM, batch_sz)

            # Update Discriminator
            grads_d = Flux.gradient(disc) do d
                l_real = mean(-log.(d(x_real) .+ 1f-8))
                l_fake = mean(-log.(1f-0 .- d(gen(z)) .+ 1f-8))
                l_real + l_fake
            end
            Flux.update!(opt_d, disc, grads_d[1])

            # Update Generator
            grads_g = Flux.gradient(gen) do g
                mean(-log.(disc(g(z)) .+ 1f-8))
            end
            Flux.update!(opt_g, gen, grads_g[1])

            # Logging (Safe from AD engine)
            d_out_real = disc(x_real)
            d_out_fake = disc(gen(z))
            running_loss += mean(-log.(d_out_real .+ 1f-8) .- log.(1f-0 .- d_out_fake .+ 1f-8))
        end

        # Evaluate Performance
        if epoch % 100 == 0
            z_test = randn(Float32, LATENT_DIM, 100)
            fakes = gen(z_test)
            oracle_preds = onecold(oracle(fakes))
            precision = mean(oracle_preds .== 1)

            avg_loss = running_loss / length(loader)
            push!(history_prec, precision * 100)
            push!(history_loss, avg_loss)

            @printf("Epoch %d | Precision: %.2f%% | D-Loss: %.4f\n", epoch, precision*100, avg_loss)
        end
    end

    # 3. Final Generation and Plotting
    println("\n--- Training Complete ---")
    z_final = randn(Float32, LATENT_DIM, 1)
    final_sample = gen(z_final)

    indices = [argmax(final_sample[i, :, 1]) for i in 1:57]
        println("Generated Sequence: ", join(ALPHABET[indices]))

        plot_gan_results(history_prec, history_loss, final_sample)
    end

    @time main()

