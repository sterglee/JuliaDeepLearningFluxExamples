#Epoch 1000 | GAN Precision: 100.0%

#--- Final Generation ---
#Synthetic Promoter: atgcaaattaattcttgacgtttcatcaaatttaaacacacatcacctcaaatgaat
# 18.614994 seconds (26.09 M allocations: 8.483 GiB, 6.73% gc time, 22.24% compilation time)

using HTTP, Flux, Statistics, Random, LinearAlgebra
using Flux: onehotbatch, DataLoader, onecold, sigmoid, logitcrossentropy

# ------------------------------------------------------------
# 1. DATA LOADING
# ------------------------------------------------------------
function download_and_preprocess()
    println("Fetching dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data"
    response = HTTP.get(url)
    raw_lines = split(strip(String(response.body)), "\n")

    alphabet = ['a', 'g', 'c', 't']
    X_promoters = []
    X_all = []
    Y_all = []

    for line in raw_lines
        parts = split(line, ",")
        if length(parts) < 3 continue end
        
        label = strip(parts[1]) == "+" ? 1 : 2
        seq = lowercase(replace(strip(parts[3]), r"\s+" => ""))
        if length(seq) != 57 continue end

        encoded = Float32.(onehotbatch(collect(seq), alphabet))
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
# 2. ORACLE CLASSIFIER (To Measure Precision)
# ------------------------------------------------------------
# This model acts as the "judge" for the GAN's quality
function build_oracle()
    return Chain(
        Conv((7,), 4 => 16, relu, pad=SamePad()),
        Flux.flatten,
        Dense(57 * 16, 2)
    )
end

function train_oracle(X, Y)
    model = build_oracle()
    opt = Flux.setup(Adam(0.001), model)
    loader = DataLoader((X, Y), batchsize=16, shuffle=true)
    
    println("Training Oracle Classifier...")
    for epoch in 1:50
        for (x, y) in loader
            grads = Flux.gradient(m -> logitcrossentropy(m(x), y), model)
            Flux.update!(opt, model, grads[1])
        end
    end
    return model
end

# ------------------------------------------------------------
# 3. GAN MODELS
# ------------------------------------------------------------
function build_generator(latent_dim)
    return Chain(
        Dense(latent_dim, 128, relu),
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
# 4. MAIN & PRECISION MEASUREMENT
# ------------------------------------------------------------
function main()
    Random.seed!(42)
    latent_dim = 10
    X_promoters, X_all, Y_all = download_and_preprocess()
    
    # Train the Oracle first so we can judge the GAN
    oracle = train_oracle(X_all, Y_all)
    
    gen = build_generator(latent_dim)
    disc = build_discriminator()
    opt_gen = Flux.setup(Adam(0.0002), gen)
    opt_disc = Flux.setup(Adam(0.0002), disc)
    
    loader = DataLoader(X_promoters, batchsize=16, shuffle=true)

    println("\n--- Training GAN ---")
    for epoch in 1:1000
        for x_real in loader
            batch_size = size(x_real, 3)
            z = randn(Float32, latent_dim, batch_size)

            # Train Discriminator
            grads_d = Flux.gradient(disc) do d
                real_loss = mean(-log.(d(x_real) .+ 1f-8))
                fake_loss = mean(-log.(1f-0 .- d(gen(z)) .+ 1f-8))
                real_loss + fake_loss
            end
            Flux.update!(opt_disc, disc, grads_d[1])

            # Train Generator
            grads_g = Flux.gradient(gen) do g
                mean(-log.(disc(g(z)) .+ 1f-8))
            end
            Flux.update!(opt_gen, gen, grads_g[1])
        end

        if epoch % 100 == 0
            # Calculate Precision: What % of fakes does Oracle think are real?
            z_test = randn(Float32, latent_dim, 100)
            fakes = gen(z_test)
            oracle_preds = onecold(oracle(fakes))
            # In our data, label 1 is Promoter (+)
            precision = mean(oracle_preds .== 1)
            println("Epoch $epoch | GAN Precision: $(round(precision*100, digits=2))%")
        end
    end

    println("\n--- Final Generation ---")
    z_final = randn(Float32, latent_dim, 1)
    fake_sample = gen(z_final)
    alphabet = ['a','g','c','t']
    indices = [argmax(fake_sample[i, :, 1]) for i in 1:57]
    println("Synthetic Promoter: ", join(alphabet[indices]))
end

@time main()