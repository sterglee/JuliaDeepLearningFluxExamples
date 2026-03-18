module JTensorFlow

using Flux
using Flux: Chain, Dense, Conv, ConvTranspose, Dropout, flatten,
    LSTM, GRU, SamePad

# =========================
# SAFE MultiHeadAttention
# =========================
struct SimpleMHA
    Wq::Dense
    Wk::Dense
    Wv::Dense
    Wo::Dense
    n_heads::Int
    d_model::Int
    d_head::Int
end

function SimpleMHA(n_heads::Int, d_model::Int)
    @assert d_model % n_heads == 0
    d_head = div(d_model, n_heads)

    SimpleMHA(
        Dense(d_model => d_model),
        Dense(d_model => d_model),
        Dense(d_model => d_model),
        Dense(d_model => d_model),
        n_heads,
        d_model,
        d_head
        )
end

# (d_model, seq, batch) → (d_head, n_heads, seq, batch)
split_heads(x, h, d) = reshape(x, d, h, size(x,2), size(x,3))

# back to (d_model, seq, batch)
combine_heads(x) = reshape(x, :, size(x,3), size(x,4))

function (m::SimpleMHA)(q, k, v)
    Q = split_heads(m.Wq(q), m.n_heads, m.d_head)
    K = split_heads(m.Wk(k), m.n_heads, m.d_head)
    V = split_heads(m.Wv(v), m.n_heads, m.d_head)

    heads = Vector{Array{Float32,3}}()

    for i in 1:m.n_heads
        # (d_head, seq, batch)
        qh = Q[:, i, :, :]
        kh = K[:, i, :, :]
        vh = V[:, i, :, :]

        # (seq, d_head, batch)
        qh_t = permutedims(qh, (2,1,3))

        # attention scores: (seq, seq, batch)
        scores = batched_mul(qh_t, kh) ./ sqrt(m.d_head)

        attn = Flux.softmax(scores; dims=2)

        # (d_head, seq, batch)
        out = batched_mul(vh, attn)

        push!(heads, out)
    end

    # concatenate heads
    H = cat(heads..., dims=1)

    return m.Wo(H)
end

# =========================
# Layer Factory
# =========================
struct Layers end
const layers = Layers()

function Base.getproperty(::Layers, s::Symbol)

    if s === :Dense
        return (in_out; activation=identity) ->
            Chain(Dense(in_out), activation)

        elseif s === :Flatten
        return () -> flatten

        elseif s === :Dropout
        return (rate) -> Dropout(rate)

        elseif s === :Conv2D
        return (filters, kernel; strides=1, padding="same", activation=identity) -> begin
            pad = padding == "same" ? SamePad() : 0
            return (in_ch) ->
                Chain(
                    Conv(kernel, in_ch => filters; stride=strides, pad=pad),
                    activation
                    )
        end

        elseif s === :Conv2DTranspose
        return (filters, kernel; strides=1, padding="same", activation=identity) -> begin
            pad = padding == "same" ? SamePad() : 0
            return (in_ch) ->
                Chain(
                    ConvTranspose(kernel, in_ch => filters; stride=strides, pad=pad),
                    activation
                    )
        end

        elseif s === :DepthwiseConv2D
        return (kernel; strides=1, padding="same", activation=identity) -> begin
            pad = padding == "same" ? SamePad() : 0
            return (in_ch) ->
                Chain(
                    Conv(kernel, in_ch => in_ch;
                         stride=strides, pad=pad, groups=in_ch),
                    activation
                    )
        end

        elseif s === :LSTM
        return (in_out) -> LSTM(in_out)

        elseif s === :GRU
        return (in_out) -> GRU(in_out)

        elseif s === :MultiHeadAttention
        return (n_heads, d_model) -> SimpleMHA(n_heads, d_model)

    else
        error("Unknown layer: $s")
    end
end

# =========================
# Model Wrapper
# =========================
mutable struct Model
    chain::Chain
    Model(args...) = new(Chain(args...))
end

(m::Model)(x) = m.chain(x)

export layers, Model

end

# =========================
# MAIN
# =========================
using Flux
using Flux: SkipConnection, batched_mul
using .JTensorFlow

function main()
    println("=== JTensorFlow Interface Examples ===")

    # Dense
    mlp = JTensorFlow.layers.Dense(784 => 10, activation=relu)
    println("1. Dense Out: ", size(mlp(rand(Float32, 784, 1))))

    # Conv
    conv = JTensorFlow.layers.Conv2D(16, (3,3))(1)
    deconv = JTensorFlow.layers.Conv2DTranspose(1, (3,3), strides=2)(16)
    img = rand(Float32, 28, 28, 1, 1)
    println("2. Conv -> Deconv Out: ", size(deconv(conv(img))))

    # LSTM
    lstm = JTensorFlow.layers.LSTM(10 => 32)
    Flux.reset!(lstm)
    println("3. LSTM Step Out: ", size(lstm(rand(Float32, 10, 1))))

    # Transformer (FIXED)
    mha = JTensorFlow.layers.MultiHeadAttention(4, 64)
    x_seq = rand(Float32, 64, 10, 1)
    println("4. Transformer Out: ", size(mha(x_seq, x_seq, x_seq)))

    # Residual
    residual_block = SkipConnection(
        Chain(JTensorFlow.layers.Conv2D(16, (3,3))(16)),
        +
            )
    println("5. Residual Block Out: ",
            size(residual_block(rand(Float32, 28, 28, 16, 1))))
end

main()

