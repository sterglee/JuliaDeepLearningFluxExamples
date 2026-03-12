# time 47.73 secs
# accuracy 47.62

using HTTP, Flux, Statistics, MLUtils
using Flux: onehotbatch, DataLoader
using Transformers
using Transformers.Layers

# 1. DATA LOADING
function download_and_preprocess()
    println("Fetching dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data"
    response = HTTP.get(url)
    raw_lines = split(strip(String(response.body)), "\n")

    alphabet = ['a', 'g', 'c', 't']
    X, Y = [], []

    for line in raw_lines
        parts = split(line, ",")
        if length(parts) < 3 continue end
        label = strip(parts[1]) == "+" ? 1 : 2
        seq = replace(strip(parts[3]), r"\s+" => "")
        
        encoded = Float32.(onehotbatch(collect(lowercase(seq)), alphabet))
        push!(X, encoded) 
        push!(Y, onehotbatch(label, 1:2))
    end

    return cat(X..., dims=3), cat(Y..., dims=2)
end

# 2. CUSTOM POSITIONAL ENCODING (Standard Sine/Cosine)
function get_sinusoidal_embeddings(dim, len)
    pos = reshape(0:len-1, 1, :)
    div_term = exp.((0:2:dim-1) .* -(log(10000.0) / dim))
    pe = zeros(Float32, dim, len)
    pe[1:2:end, :] .= sin.(pos .* div_term)
    pe[2:2:end, :] .= cos.(pos .* div_term)
    return pe
end

# 3. TRANSFORMER MODEL DEFINITION
function build_transformer_model()
    hidden_dim = 64
    num_heads = 4
    head_dim = 4
    seq_len = 57

    # Create the Transformer block
    # Note: Transformers.jl uses (nhead, d_model, d_head, d_inner)
    block = TransformerBlock(num_heads, hidden_dim, head_dim, hidden_dim * 2)
    
    # Pre-calculate positional embeddings
    pe = get_sinusoidal_embeddings(hidden_dim, seq_len)

    return Chain(
        # 1. Project 4 bases to hidden_dim
        Dense(4, hidden_dim),
        
        # 2. Add Positional Encoding (Broadcasting over Batch)
        x -> x .+ pe,
        
        # 3. Pass through Transformer Block (Handles the NamedTuple)
        x -> block(x, nothing),
        
        # 4. Extract the hidden state from the NamedTuple
        x -> x.hidden_state,
        
        # 5. Global Mean Pool (across sequence length)
        x -> mean(x, dims=2),
        
        # 6. Output Head
        Flux.flatten,
        Dense(hidden_dim, 2),
        softmax
    )
end

# 4. MAIN EXECUTION
function main()
    X_raw, Y_raw = download_and_preprocess()
    (x_train, y_train), (x_test, y_test) = splitobs((X_raw, Y_raw), at=0.8)
    train_loader = DataLoader((x_train, y_train), batchsize=16, shuffle=true)
    
    model = build_transformer_model()
    loss(m, x, y) = Flux.crossentropy(m(x), y)
    opt_state = Flux.setup(Adam(0.0005), model)

    println("\n--- Training Transformer ---")
    for epoch in 1:1000
        for (x, y) in train_loader
            grads = Flux.gradient(m -> loss(m, x, y), model)
            Flux.update!(opt_state, model, grads[1])
        end
        
        if epoch % 20 == 0
            acc = mean(Flux.onecold(model(x_test)) .== Flux.onecold(y_test))
            println("Epoch $epoch | Test Accuracy: $(round(acc*100, digits=2))%")
        end
    end
end

@time main()
