# ==============================================================================
# JPyTorch.jl - Final Stable Implementation
# ==============================================================================

using Flux
using CUDA
using Statistics
using Flux: Chain, Dense, Conv, ConvTranspose, MaxPool, BatchNorm, LayerNorm,
    flatten, relu, softplus, tanh, logsoftmax, batched_mul, SamePad
using Functors: @functor, fmap

# Ensure everything is Float32 for GPU compatibility
f32(m) = fmap(x -> x isa AbstractArray{Float64} ? Float32.(x) : x, m)

# =========================
# 1. Base Layer Definitions
# =========================
abstract type Module end

struct LinearLayer <: Module; layer::Dense; end
@functor LinearLayer
LinearLayer(in::Int, out::Int; act=identity) = LinearLayer(Dense(in => out, act))
(m::LinearLayer)(x) = m.layer(x)

# =========================
# 2. Advanced Model Components
# =========================

# --- Transformer (Fixed MHA Tuple Return) ---
struct TransformerBlock <: Module
    attn::Flux.MultiHeadAttention
    norm1::LayerNorm
    norm2::LayerNorm
    mlp::Chain
end
@functor TransformerBlock

function (m::TransformerBlock)(x)
    h = m.norm1(x)
    # Destructure the (output, weights) tuple from Flux MHA
    attn_out, _ = m.attn(h, h, h)
    x = x + attn_out
    x = x + m.mlp(m.norm2(x))
    return x
end

# --- VAE (Fixed NamedTuple for Math Safety) ---
struct VAEModule <: Module
    encoder::Chain; mu_l::Dense; logvar_l::Dense; decoder::Chain
end
@functor VAEModule

function (m::VAEModule)(x)
    h = m.encoder(x)
    mu, logvar = m.mu_l(h), m.logvar_l(h)
    std = exp.(0.5f0 .* logvar)
    z = mu + CUDA.randn(Float32, size(std)) .* std
    return (reconstruction=m.decoder(z), mu=mu, logvar=logvar)
end

# --- Normalizing Flow (Fixed Mask Type Error) ---
struct CouplingLayer <: Module
    mask::AbstractArray # Must be CuArray on GPU
    s_net::Chain
    t_net::Chain
end
@functor CouplingLayer

function (m::CouplingLayer)(x)
    # All math here must be on the same device
    x_m = x .* m.mask
    s = m.s_net(x_m) .* (1.0f0 .- m.mask)
    t = m.t_net(x_m) .* (1.0f0 .- m.mask)
    return x_m + (1.0f0 .- m.mask) .* (x .* exp.(s) + t)
end

# =========================
# 3. The nn Namespace
# =========================
module nn
import ..LinearLayer, ..TransformerBlock, ..VAEModule, ..CouplingLayer
import Flux

const Linear = LinearLayer
const Transformer = TransformerBlock
const VAE = VAEModule
const Flow = CouplingLayer
const Sequential = Flux.Chain
const Flatten = Flux.flatten
const ReLU = Flux.relu
end

# =========================
# 4. Device Management
# =========================
function to_device(model)
    if CUDA.functional()
        # f32() ensures no Float64 sneak in, gpu() moves to VRAM
        return f32(model) |> gpu
    end
    return f32(model)
end

# =========================
# 5. Main Demo
# =========================
function main()
    println("=== JPyTorch Stable Zoo Demo ===")

    # 1. Transformer
    m1 = TransformerBlock(
        Flux.MultiHeadAttention(64, nheads=4),
        LayerNorm(64), LayerNorm(64),
        Chain(Dense(64=>128, relu), Dense(128=>64))
        ) |> to_device
    x1 = CUDA.randn(Float32, 64, 10, 1)
    println("1. Transformer Out: ", size(m1(x1)))

    # 2. VAE
    m2 = nn.VAE(
        Chain(Dense(784=>128, relu)),
        Dense(128=>20), Dense(128=>20),
        Chain(Dense(20=>784, Flux.sigmoid))
        ) |> to_device
    x2 = CUDA.randn(Float32, 784, 2)
    res = m2(x2)
    println("2. VAE Recon Out:   ", size(res.reconstruction))

    # 3. Flow (Fixing the non-bitstype Mask)
    # We move the mask to the GPU explicitly
    mask = (CUDA.rand(Float32, 784) .> 0.5f0) |> gpu
    m3 = nn.Flow(mask, Chain(Dense(784=>784, tanh)), Chain(Dense(784=>784))) |> to_device
    println("3. Flow Out:        ", size(m3(x2)))

    println("\n✅ All GPU kernels compiled successfully.")
end

main()

