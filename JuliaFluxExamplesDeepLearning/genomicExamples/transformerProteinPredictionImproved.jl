using Flux
using Flux: DataLoader, logitcrossentropy, onecold, trainmode!, testmode!
using Flux: Dense, Chain, LayerNorm, MultiHeadAttention, Dropout
using Functors
using HTTP, Statistics, MLUtils, Random

# ------------------------------------------------------------
# 1. DATA ACQUISITION
# ------------------------------------------------------------

function get_protein_data()
    url = "https://raw.githubusercontent.com/cmbi/protein-sequence-analysis/master/data/cyc.fasta"
    try
        return String(HTTP.get(url).body)
    catch
        @warn "Download failed, generating synthetic proteins"
        aa = collect("ARNDCEQGHILKMFPSTWYV")
        return join([">seq$i\n" * join(rand(aa, 100)) for i in 1:1000], "\n")
    end
end

function parse_kmers(raw_data, window)
    sequences = String[]
    current = ""
    for line in split(raw_data, "\n")
        line = strip(line)
        if startswith(line, ">")
            !isempty(current) && push!(sequences, current)
            current = ""
        elseif !isempty(line)
            current *= line
        end
    end
    !isempty(current) && push!(sequences, current)

    chars = sort(unique(join(sequences)))
    c2i = Dict(c => i for (i, c) in enumerate(chars))
    
    X_indices = Vector{Vector{Int}}()
    Y_indices = Int[]

    for seq in sequences
        L = length(seq)
        L <= window && continue
        for i in 1:(L - window)
            push!(X_indices, [c2i[c] for c in seq[i:i+window-1]])
            push!(Y_indices, c2i[seq[i+window]])
        end
    end
    return X_indices, Y_indices, length(chars)
end

# ------------------------------------------------------------
# 2. POSITIONAL ENCODING
# ------------------------------------------------------------

struct PositionalEncoding
    weight::AbstractArray
end

Functors.@functor PositionalEncoding

# Ensure the PE broadcasts correctly over the batch dimension
(m::PositionalEncoding)(x) = x .+ m.weight[:, 1:size(x, 2), :]

function positional_weights(dim, max_len)
    pe = zeros(Float32, dim, max_len, 1)
    for pos in 1:max_len
        for i in 1:2:(dim - 1)
            pe[i, pos, 1] = sin(pos / 10000^((i - 1) / dim))
            pe[i + 1, pos, 1] = cos(pos / 10000^((i - 1) / dim))
        end
    end
    return pe
end

# ------------------------------------------------------------
# 3. IMPROVED TRANSFORMER BLOCK (Pre-Norm)
# ------------------------------------------------------------

struct TransformerBlock
    norm1
    attn
    norm2
    mlp
    drop
end

Functors.@functor TransformerBlock

function TransformerBlock(dim, heads, mlp_dim; dropout=0.1f0)
    return TransformerBlock(
        LayerNorm(dim),
        MultiHeadAttention(dim; nheads=heads),
        LayerNorm(dim),
        Chain(
            Dense(dim, mlp_dim, relu),
            Dropout(dropout),
            Dense(mlp_dim, dim)
        ),
        Dropout(dropout)
    )
end

function (m::TransformerBlock)(x)
    # Sub-layer 1: Attention with Residual Connection (Pre-Norm)
    h = m.norm1(x)
    a, _ = m.attn(h, h, h)
    x = x .+ m.drop(a)

    # Sub-layer 2: MLP with Residual Connection (Pre-Norm)
    x = x .+ m.drop(m.mlp(m.norm2(x)))
    return x
end

# ------------------------------------------------------------
# 4. MAIN TRAINING LOGIC
# ------------------------------------------------------------

function main()
    # Hyperparameters
    window = 25    # Increased window for better context
    embed  = 128   # Increased embedding size
    batch  = 128
    epochs = 50
    dropout_rate = 0.1f0

    raw = get_protein_data()
    Xidx, Yidx, vocab = parse_kmers(raw, window)
    labels = 1:vocab

    println("Dataset windows: ", length(Xidx))
    println("Vocabulary size: ", vocab)

    # Convert to One-Hot Tensors (Preprocessing once for speed)
    X_tensor = Float32.(MLUtils.stack([Flux.onehotbatch(x, labels) for x in Xidx]))
    Y_tensor = Float32.(Flux.onehotbatch(Yidx, labels))

    train_data, test_data = splitobs((X_tensor, Y_tensor), at=0.85)

    train_loader = DataLoader(train_data, batchsize=batch, shuffle=true)
    test_loader  = DataLoader(test_data, batchsize=batch)

    # Improved Model Architecture
    model = Chain(
        Dense(vocab, embed),
        PositionalEncoding(positional_weights(embed, window)),
        Dropout(dropout_rate),
        TransformerBlock(embed, 8, 256; dropout=dropout_rate),
        TransformerBlock(embed, 8, 256; dropout=dropout_rate),
        x -> x[:, end, :], # "Global" view from the last token
        LayerNorm(embed),
        Dense(embed, vocab)
    )

    # AdamW (Weight Decay) helps with Transformer generalization
    opt = Flux.setup(Flux.Optimisers.AdamW(1f-3, (0.9, 0.999), 1f-4), model)

    println("Starting training...")

    for epoch in 1:epochs
        trainmode!(model)
        total_loss = 0.0

        for (x, y) in train_loader
            l, grads = Flux.withgradient(model) do m
                y_hat = m(x)
                logitcrossentropy(y_hat, y)
            end
            Flux.update!(opt, model, grads[1])
            total_loss += l
        end

        # Evaluation
        testmode!(model)
        acc_sum = 0.0
        total_samples = 0
        for (x, y) in test_loader
            y_hat = model(x)
            acc_sum += sum(onecold(y_hat) .== onecold(y))
            total_samples += size(y, 2)
        end

        avg_loss = total_loss / length(train_loader)
        avg_acc  = (acc_sum / total_samples) * 100
        println("Epoch $epoch | Loss: $(round(avg_loss, digits=4)) | Test Acc: $(round(avg_acc, digits=2))%")
    end
end

main()

