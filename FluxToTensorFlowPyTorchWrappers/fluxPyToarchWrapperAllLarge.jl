# =========================
# JPyTorch.jl
# =========================

using Flux
using Flux: Chain, Dense, Conv, ConvTranspose, Dropout, flatten,
    LSTM, GRU, SamePad, batched_mul, relu, softmax, reset!
    using Functors: @functor

    # 1. Define the internal Layer Structs in Main scope
    abstract type Module end

    struct LinearLayer <: Module
        layer::Chain
    end
    @functor LinearLayer
    LinearLayer(in::Int, out::Int; activation=identity) = LinearLayer(Chain(Dense(in => out, activation)))
    (m::LinearLayer)(x) = m.layer(x)

    struct Conv2dLayer <: Module
        layer::Chain
    end
    @functor Conv2dLayer
    function Conv2dLayer(in_ch, out_ch, kernel; stride=1, padding="same", activation=identity)
        pad = padding == "same" ? SamePad() : 0
        Conv2dLayer(Chain(Conv(kernel, in_ch => out_ch; stride=stride, pad=pad), activation))
    end
    (m::Conv2dLayer)(x) = m.layer(x)

    struct DropoutLayer <: Module
        layer::Dropout
    end
    @functor DropoutLayer
    DropoutLayer(p) = DropoutLayer(Dropout(p))
    (m::DropoutLayer)(x) = m.layer(x)

    struct FlattenLayer <: Module end
    (m::FlattenLayer)(x) = flatten(x)

    struct LSTMLayer <: Module
        layer::LSTM
    end
    @functor LSTMLayer
    LSTMLayer(in_out) = LSTMLayer(LSTM(in_out))
    (m::LSTMLayer)(x) = m.layer(x)

    struct SequentialLayer <: Module
        layers::Chain
    end
    @functor SequentialLayer
    SequentialLayer(args...) = SequentialLayer(Chain(args...))
    (m::SequentialLayer)(x) = m.layers(x)

    # 2. Multihead Attention
    struct MultiheadAttentionLayer <: Module
        Wq::Dense; Wk::Dense; Wv::Dense; Wo::Dense
        n_heads::Int; d_head::Int
    end
    @functor MultiheadAttentionLayer

    function MultiheadAttentionLayer(embed_dim::Int, num_heads::Int)
        d_head = div(embed_dim, num_heads)
        MultiheadAttentionLayer(
            Dense(embed_dim => embed_dim), Dense(embed_dim => embed_dim),
            Dense(embed_dim => embed_dim), Dense(embed_dim => embed_dim),
            num_heads, d_head
            )
    end

    function (m::MultiheadAttentionLayer)(q, k, v)
        E, S, B = size(q)
        H, D = m.n_heads, m.d_head
        Q = reshape(m.Wq(q), D, H, S, B)
        K = reshape(m.Wk(k), D, H, S, B)
        V = reshape(m.Wv(v), D, H, S, B)
        heads = []
        for i in 1:H
            qh, kh, vh = Q[:, i, :, :], K[:, i, :, :], V[:, i, :, :]
            scores = batched_mul(permutedims(qh, (2, 1, 3)), kh) ./ sqrt(Float32(D))
            attn = softmax(scores, dims=2)
            push!(heads, batched_mul(vh, attn))
        end
        return m.Wo(cat(heads..., dims=1))
    end

    # 3. The nn Namespace (The Fix)
    module nn
    # Bring the layers into this module's scope
    import ..LinearLayer, ..Conv2dLayer, ..DropoutLayer, ..FlattenLayer,
        ..LSTMLayer, ..MultiheadAttentionLayer, ..SequentialLayer
    import Flux

    # Alias them to PyTorch names
    const Linear = LinearLayer
    const Conv2d = Conv2dLayer
    const Dropout = DropoutLayer
    const Flatten = FlattenLayer
    const LSTM = LSTMLayer
    const MultiheadAttention = MultiheadAttentionLayer
    const Sequential = SequentialLayer

    # Common Activations
    const ReLU = Flux.relu
    const Sigmoid = Flux.sigmoid
    end

    # 4. Main Demo
    function main()
        println("=== JPyTorch Complete & Corrected ===")

        # Test Linear
        fc = nn.Linear(784, 10)
        println("1. Linear Out: ", size(fc(rand(Float32, 784, 1))))

        # Test Sequential
        model = nn.Sequential(
            nn.Conv2d(1, 16, (3,3)),
            nn.Flatten(),
            nn.Linear(16*28*28, 10)
            )
        x = rand(Float32, 28, 28, 1, 1)
        println("2. Sequential Out: ", size(model(x)))

        # Verify Parameter Tracking
        ps = Flux.params(model)
        println("3. Total Parameter Arrays: ", length(ps))
    end

    main()

