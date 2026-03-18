using Flux
using CUDA
using Statistics
using Flux: Chain, Dense, Conv, SamePad, flatten, relu, softmax, batched_mul
using Functors: @functor

# =========================
# 1. The Keras-Base "Layer"
# =========================
abstract type Layer end

# Mimicking Keras's functional style
(m::Layer)(x) = call(m, x)

# =========================
# 2. Wrapped Keras Routines
# =========================

# --- nn.Linear (Keras Dense) ---
struct Linear <: Layer
    weight
    bias
    activation
end
@functor Linear

function Linear(in_features::Int, out_features::Int; activation=identity)
    # Keras-style weight initialization
    w = Flux.glorot_uniform(out_features, in_features)
    b = zeros(Float32, out_features)
    return Linear(w, b, activation)
end

function call(m::Linear, x)
    return m.activation.(m.weight * x .+ m.bias)
end

# --- TransformerBlock (Keras Implementation) ---
struct TransformerBlock <: Layer
    att::Flux.MultiHeadAttention
    layernorm1::Flux.LayerNorm
    layernorm2::Flux.LayerNorm
    mlp::Chain
end
@functor TransformerBlock

function TransformerBlock(embed_dim, num_heads, ff_dim)
    return TransformerBlock(
        Flux.MultiHeadAttention(embed_dim, nheads=num_heads),
        Flux.LayerNorm(embed_dim),
        Flux.LayerNorm(embed_dim),
        Chain(Dense(embed_dim => ff_dim, relu), Dense(ff_dim => embed_dim))
        )
end

function call(m::TransformerBlock, x)
    # MultiHeadAttention in Flux returns (output, weights)
    attn_output, _ = m.att(x, x, x)
    out1 = m.layernorm1(x + attn_output)
    ffn_output = m.mlp(out1)
    return m.layernorm2(out1 + ffn_output)
end

# --- VAE (Keras-Style Model) ---
struct VAE <: Layer
    encoder::Chain
    decoder::Chain
    latent_dim::Int
end
@functor VAE

function (m::VAE)(x)
    # Encoder logic
    h = m.encoder(x)
    # Split for mu and logvar
    mu = h[1:m.latent_dim, :]
    logvar = h[m.latent_dim+1:end, :]

    # Reparameterization
    std = exp.(0.5f0 .* logvar)
    eps = CUDA.randn(Float32, size(std))
    z = mu + eps .* std

    return (reconstruction=m.decoder(z), mu=mu, logvar=logvar)
end

# =========================
# 3. Keras "Sequential" Wrapper
# =========================
struct Sequential <: Layer
    layers::Vector{Any}
end
@functor Sequential

function (m::Sequential)(x)
    for layer in m.layers
        x = layer(x)
    end
    return x
end

# =========================
# 4. The nn Namespace
# =========================
module nn
import ..Linear, ..TransformerBlock, ..VAE, ..Sequential
import Flux

# Mapping Keras names to our Julia wrappers
const Dense = Linear
const Transformer = TransformerBlock
const VariationalAutoEncoder = VAE
const Model = Sequential

# Functions
const ReLU = Flux.relu
const Sigmoid = Flux.sigmoid
const Flatten = Flux.flatten
end

# =========================
# 5. Device & Main Execution
# =========================
to_device(m) = CUDA.functional() ? fmap(gpu, m) : m

function main()
    println("=== JPyTorch: Pure Julia Keras Wrappers ===")

    # 1. Keras Dense Layer usage
    dense_layer = nn.Dense(784, 128, activation=nn.ReLU) |> to_device
    x = CUDA.randn(Float32, 784, 5) # 5 samples
    println("Dense Output: ", size(dense_layer(x)))

    # 2. Keras TransformerBlock usage
    tform = nn.Transformer(64, 4, 128) |> to_device
    x_seq = CUDA.randn(Float32, 64, 10, 1) # (Embed, Seq, Batch)
    println("Transformer Output: ", size(tform(x_seq)))

    # 3. Keras VAE usage
    vae = nn.VAE(
        Chain(Dense(784 => 128, relu), Dense(128 => 40)), # latent_dim * 2
        Chain(Dense(20 => 128, relu), Dense(128 => 784, nn.Sigmoid)),
        20
        ) |> to_device
    res = vae(x)
    println("VAE Reconstruction: ", size(res.reconstruction))

    println("\n✅ Successfully wrapped Keras routines in pure Julia.")
end

main()

