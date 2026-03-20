using Flux
using Flux: Dense, Embedding, LayerNorm, Dropout, relu, logitcrossentropy, onehotbatch
using NNlib: softmax, batched_mul
using Statistics
using Random
using LinearAlgebra
using CUDA
using Printf
using Optimisers

# -----------------------------
# 1. DEVICE SETUP
# -----------------------------
const device = CUDA.functional() ? gpu : cpu
println("Using device: ", device == gpu ? "GPU" : "CPU")

# -----------------------------
# 2. POSITIONAL ENCODING
# -----------------------------
struct PositionalEncoding
    pe
    dropout
end

Flux.@functor PositionalEncoding

function PositionalEncoding(d_model::Int, dropout=0.1, max_len=5000)
    pe = zeros(Float32, d_model, max_len)
    pos = reshape(0:max_len-1, 1, :)
    div_term = exp.((0:2:d_model-1) .* (-log(10000.0) / d_model))

    pe[1:2:end, :] .= sin.(div_term .* pos)
    pe[2:2:end, :] .= cos.(div_term .* pos)

    return PositionalEncoding(device(reshape(pe, d_model, max_len, 1)), Dropout(dropout))
end

function (p::PositionalEncoding)(x)
    return p.dropout(x .+ p.pe[:, 1:size(x, 2), :])
end

# -----------------------------
# 3. MULTI-HEAD ATTENTION
# -----------------------------
struct SimpleMultiHeadAttention
    Wq; Wk; Wv; Wo
    nheads::Int
    head_dim::Int
end

Flux.@functor SimpleMultiHeadAttention

function SimpleMultiHeadAttention(d_model::Int, nheads::Int)
    head_dim = div(d_model, nheads)
    return SimpleMultiHeadAttention(
        Dense(d_model, d_model),
        Dense(d_model, d_model),
        Dense(d_model, d_model),
        Dense(d_model, d_model),
        nheads,
        head_dim
        )
end

function (m::SimpleMultiHeadAttention)(x, mask)
    dim, seq, batch = size(x)
    h, d = m.nheads, m.head_dim

    # Projections
    Q = reshape(m.Wq(x), d, h, seq, batch)
    K = reshape(m.Wk(x), d, h, seq, batch)
    V = reshape(m.Wv(x), d, h, seq, batch)

    # Transpose for batched_mul: (d, seq, h*batch)
    Qr = reshape(permutedims(Q, (1, 3, 2, 4)), d, seq, h * batch)
    Kr = reshape(permutedims(K, (1, 3, 2, 4)), d, seq, h * batch)
    Vr = reshape(permutedims(V, (1, 3, 2, 4)), d, seq, h * batch)

    # Attention scores
    scores = batched_mul(permutedims(Qr, (2, 1, 3)), Kr) ./ Float32(sqrt(d))

    if mask !== nothing
        scores = scores .+ device(reshape(mask, seq, seq, 1))
    end

    attn = softmax(scores, dims=1)
    out = batched_mul(Vr, attn)

    # Recombine heads
    out = reshape(out, d, seq, h, batch)
    out = permutedims(out, (1, 3, 2, 4))
    return m.Wo(reshape(out, dim, seq, batch))
end

# -----------------------------
# 4. TRANSFORMER ARCHITECTURE
# -----------------------------


struct TransformerBlock
    mha; ln1; ff; ln2; dropout
end
Flux.@functor TransformerBlock

function TransformerBlock(dim, heads, hidden, dropout)
    return TransformerBlock(
        SimpleMultiHeadAttention(dim, heads),
        LayerNorm(dim),
        Chain(Dense(dim, hidden, relu), Dense(hidden, dim)),
        LayerNorm(dim),
        Dropout(dropout)
        )
end

function (m::TransformerBlock)(x, mask)
    x = m.ln1(x .+ m.dropout(m.mha(x, mask)))
    x = m.ln2(x .+ m.dropout(m.ff(x)))
    return x
end

struct Transformer
    embedding; pos_enc; layers; decoder; dim::Int
end
Flux.@functor Transformer

function Transformer(vocab_size, dim, heads, hidden, n_layers, dropout)
    blocks = [TransformerBlock(dim, heads, hidden, dropout) for _ in 1:n_layers]
        return Transformer(
            Embedding(vocab_size => dim),
            PositionalEncoding(dim, dropout),
            Chain(blocks...),
            Dense(dim, vocab_size),
            dim
            )
    end

    function (m::Transformer)(x, mask=nothing)
        x = m.embedding(x) .* Float32(sqrt(m.dim))
        x = m.pos_enc(x)
        for layer in m.layers
            x = layer(x, mask)
        end
        return m.decoder(x)
    end

    # -----------------------------
    # 5. TRAINING LOOP (FIXED LOSS)
    # -----------------------------
    function train()
        VOCAB, DIM, HEADS, HIDDEN, LAYERS, SEQ, BATCH = 500, 128, 4, 256, 2, 32, 16

        # Synthetic Data
        x_raw = rand(1:VOCAB, SEQ, 1000)
        y_raw = circshift(x_raw, (-1, 0))

        model = Transformer(VOCAB, DIM, HEADS, HIDDEN, LAYERS, 0.1) |> device
        opt_state = Flux.setup(Optimisers.Adam(0.001), model)
        mask = Float32[j > i ? -1f9 : 0f0 for i in 1:SEQ, j in 1:SEQ] |> device

            println("Training...")

            for epoch in 1:3
                total_loss, steps = 0.0, 0
                for i in 1:BATCH:(size(x_raw, 2) - BATCH)
                    xb = x_raw[:, i:i+BATCH-1] |> device
                    yb = y_raw[:, i:i+BATCH-1] |> device

                    loss, grads = Flux.withgradient(model) do m
                        logits = m(xb, mask) # (Vocab, Seq, Batch)

                        # --- THE FIX ---
                        y_hat = reshape(logits, VOCAB, :)
                        # Convert integer labels to a one-hot matrix on the correct device
                        y_true = onehotbatch(vec(yb), 1:VOCAB) |> device

                        logitcrossentropy(y_hat, y_true)
                    end

                    Flux.update!(opt_state, model, grads[1])
                    total_loss += loss
                    steps += 1
                end
                @printf("Epoch %d | Loss: %.4f\n", epoch, total_loss/steps)
            end
        end

        train()

