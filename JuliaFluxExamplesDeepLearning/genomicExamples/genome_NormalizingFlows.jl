#Epoch 1000 | Generative Precision: 94.0%

#Final Synthetic Sequence:
#ctctaattctatagttgactggtaagtaggacatcgagtaatgagccccctttcaga
# 19.994403 seconds (37.24 M allocations: 22.305 GiB, 14.11% gc time, 16.26% compilation time: 6% of which was recompilation)
 
using HTTP, Flux, Statistics, Random, LinearAlgebra
using Flux: onehotbatch, DataLoader, onecold, logitcrossentropy

# ------------------------------------------------------------
# 1. DATA LOADING
# ------------------------------------------------------------
function download_and_preprocess()
    println("Fetching dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data"
    response = HTTP.get(url)
    raw_lines = split(strip(String(response.body)), "\n")

    alphabet = ['a', 'g', 'c', 't']
    X_promoters, X_all, Y_all = [], [], []

    for line in raw_lines
        parts = split(line, ",")
        if length(parts) < 3 continue end
        
        label = strip(parts[1]) == "+" ? 1 : 2
        seq = lowercase(replace(strip(parts[3]), r"\s+" => ""))
        if length(seq) != 57 continue end

        encoded = Float32.(onehotbatch(collect(seq), alphabet))
        flat_seq = vec(encoded)
        
        push!(X_all, flat_seq)
        push!(Y_all, onehotbatch(label, 1:2))
        if label == 1 push!(X_promoters, flat_seq) end
    end
    return hcat(X_promoters...), hcat(X_all...), cat(Y_all..., dims=2)
end

# ------------------------------------------------------------
# 2. ORACLE CLASSIFIER (The "Judge")
# ------------------------------------------------------------
function train_oracle(X, Y)
    # Simple MLP to classify sequences
    model = Chain(Dense(228, 64, relu), Dense(64, 2))
    opt = Flux.setup(Adam(0.001), model)
    loader = DataLoader((X, Y), batchsize=16, shuffle=true)
    
    println("Training Oracle to judge generation quality...")
    for epoch in 1:50
        for (x, y) in loader
            grads = Flux.gradient(m -> logitcrossentropy(m(x), y), model)
            Flux.update!(opt, model, grads[1])
        end
    end
    return model
end

# ------------------------------------------------------------
# 3. NORMALIZING FLOW COMPONENTS
# ------------------------------------------------------------
struct CouplingLayer
    s_net; t_net; mask
end
Flux.@functor CouplingLayer

function (m::CouplingLayer)(x)
    x1 = x .* m.mask
    s = m.s_net(x1) .* (1f0 .- m.mask)
    t = m.t_net(x1) .* (1f0 .- m.mask)
    y = x1 .+ (x .* (1f0 .- m.mask) .* exp.(s) .+ t)
    return y, sum(s, dims=1)
end

function invert(m::CouplingLayer, y)
    y1 = y .* m.mask
    s = m.s_net(y1) .* (1f0 .- m.mask)
    t = m.t_net(y1) .* (1f0 .- m.mask)
    return y1 .+ (y .* (1f0 .- m.mask) .- t) .* exp.(-s)
end

# ------------------------------------------------------------
# 4. MAIN & ACCURACY EVALUATION
# ------------------------------------------------------------
function main()
    Random.seed!(42)
    X_prom, X_all, Y_all = download_and_preprocess()
    input_dim = size(X_prom, 1)

    # 1. Train Oracle
    oracle = train_oracle(X_all, Y_all)

    # 2. Setup Flow
    layers = []
    for i in 1:6
        m_vec = zeros(Float32, input_dim, 1)
        m_vec[i%2+1:2:end] .= 1.0
        push!(layers, CouplingLayer(
            Chain(Dense(input_dim, 128, leakyrelu), Dense(128, input_dim)),
            Chain(Dense(input_dim, 128, leakyrelu), Dense(128, input_dim)),
            m_vec
        ))
    end
    model = Chain(layers...)
    opt_state = Flux.setup(Adam(1e-4), model)
    loader = DataLoader(X_prom, batchsize=16, shuffle=true)

    println("\n--- Training Normalizing Flow ---")
    for epoch in 1:1000
        for x in loader
            grads = Flux.gradient(model) do m
                z, logdet_total = x, 0f0
                for layer in m.layers
                    z, ld = layer(z)
                    logdet_total = logdet_total .+ ld
                end
                log_pz = -0.5f0 .* (sum(z.^2, dims=1) .+ input_dim * log(2f0*π))
                mean(-(log_pz .+ logdet_total))
            end
            Flux.update!(opt_state, model, grads[1])
        end

        if epoch % 100 == 0
            # Calculate Generative Precision
            z_test = randn(Float32, input_dim, 100)
            x_gen = z_test
            for layer in reverse(model.layers) x_gen = invert(layer, x_gen) end
            
            # Use Oracle to check if generated sequences look like promoters
            preds = onecold(oracle(x_gen))
            precision = mean(preds .== 1)
            println("Epoch $epoch | Generative Precision: $(round(precision*100, digits=2))%")
        end
    end

    # Final Sample
    println("\nFinal Synthetic Sequence:")
    z_final = randn(Float32, input_dim, 1)
    x_final = z_final
    for layer in reverse(model.layers) x_final = invert(layer, x_final) end
    indices = [argmax(reshape(x_final, 4, 57)[:, i]) for i in 1:57]
    println(join(['a','g','c','t'][indices]))
end

@time main()
