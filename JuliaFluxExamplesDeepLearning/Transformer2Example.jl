# =============================================
# Flux.jl Character-Level Transformer (CPU)
# =============================================
using Flux
using MLUtils: DataLoader  # For batching
using Statistics
using Random
using BSON  # For checkpointing

# -----------------------------
# 1. Dataset (toy text example)
# -----------------------------
text = """
hello world! this is a mini transformer example in julia.
you can extend it to larger datasets easily.
"""

chars = collect(unique(text))
vocab_size = length(chars)
char2idx = Dict(c => i for (i,c) in enumerate(chars))
idx2char = Dict(i => c for (i,c) in enumerate(chars))

data = [char2idx[c] for c in text]
seq_len = 20  # sequence length for training

# Create sequences
X = [data[i:i+seq_len-1] for i in 1:(length(data)-seq_len)]
Y = [data[i+1:i+seq_len] for i in 1:(length(data)-seq_len)]

# -----------------------------
# 2. Helper functions
# -----------------------------
# One-hot encode a sequence
onehot_seq(seq) = Flux.onehotbatch(seq, 1:vocab_size)

# Positional Encoding
function positional_encoding(d_model, seq_len)
    PE = zeros(Float32, d_model, seq_len)
    for pos in 0:(seq_len-1)
        for i in 0:(d_model-1)
            if iseven(i)
                PE[i+1, pos+1] = sin(pos / (10000^(i/d_model)))
            else
                PE[i+1, pos+1] = cos(pos / (10000^((i-1)/d_model)))
            end
        end
    end
    return PE
end

# Scaled Dot-Product Attention
function scaled_dot_attention(Q, K, V)
    dk = size(K, 1)
    scores = (Q' * K) ./ sqrt(dk)
    attn = Flux.softmax(scores, dims=2)
    return V * attn'
end

# Multi-Head Attention
function multi_head_attention(x, num_heads, d_model)
    head_dim = div(d_model, num_heads)
    Q = Dense(d_model, d_model)(x)
    K = Dense(d_model, d_model)(x)
    V = Dense(d_model, d_model)(x)
    heads = []
    for i in 1:num_heads
        idx = ((i-1)*head_dim+1):(i*head_dim)
        push!(heads, scaled_dot_attention(Q[idx, :], K[idx, :], V[idx, :]))
    end
    return Dense(d_model, d_model)(cat(heads..., dims=1))
end

# Transformer Encoder Block
function transformer_encoder(d_model, num_heads, ff_dim)
    mha = x -> multi_head_attention(x, num_heads, d_model)
    ffn = Chain(
        Dense(d_model, ff_dim), relu,
        Dense(ff_dim, d_model)
    )
    return Chain(
        x -> x .+ mha(x),   # residual + attention
        x -> x .+ ffn(x)    # residual + feedforward
    )
end

# -----------------------------
# 3. Model Parameters
# -----------------------------
d_model = 64
num_heads = 4
ff_dim = 256
num_layers = 2

# Embedding + Encoder Stack + Decoder
embedding = Dense(vocab_size, d_model)
encoder = Chain([transformer_encoder(d_model, num_heads, ff_dim) for _ in 1:num_layers]...)
decoder = Dense(d_model, vocab_size)

# -----------------------------
# 4. Training Setup
# -----------------------------
batch_size = 16
train_loader = DataLoader((X, Y), batchsize=batch_size, shuffle=true)
opt = Flux.ADAM(1e-3)

# Loss function
function loss_fn(x_seq, y_seq)
    x = onehot_seq(x_seq)
    y = onehot_seq(y_seq)
    x_emb = embedding(x) .+ positional_encoding(d_model, size(x, 2))
    y_hat = encoder(x_emb)
    y_hat = decoder(y_hat)
    return sum(crossentropy.(y_hat, y))
end

# -----------------------------
# 5. Training Loop
# -----------------------------
epochs = 20

for epoch in 1:epochs
    total_loss = 0.0
    for (x_batch, y_batch) in train_loader
        gs = gradient(Flux.params(embedding, encoder, decoder)) do
            batch_loss = 0.0
            for (x_seq, y_seq) in zip(x_batch, y_batch)
                batch_loss += loss_fn(x_seq, y_seq)
            end
            batch_loss / length(x_batch)
        end
        Flux.Optimise.update!(opt, Flux.params(embedding, encoder, decoder), gs)
        total_loss += sum([loss_fn(x_seq, y_seq) for (x_seq, y_seq) in zip(x_batch, y_batch)])
    end
    println("Epoch $epoch, Loss: $(round(total_loss, digits=3))")
    # Save checkpoint
    BSON.@save "transformer_epoch_$epoch.bson" embedding encoder decoder opt
end

# -----------------------------
# 6. Autoregressive Generation
# -----------------------------
function generate_sequence(start_char, length)
    sequence = [start_char]
    for i in 2:length
        x_idx = [char2idx[c] for c in sequence]
        x = onehot_seq(x_idx)
        x_emb = embedding(x) .+ positional_encoding(d_model, size(x, 2))
        y_hat = encoder(x_emb)
        probs = Flux.softmax(decoder(y_hat[:, end]))  # ✅ fixed
        next_idx = Flux.onecold(probs)
        push!(sequence, idx2char[next_idx])
    end
    return join(sequence)
end

# Example generation
println("Generated text: ", generate_sequence('h', 50))

