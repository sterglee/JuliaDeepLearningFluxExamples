using Flux
using Flux: onehotbatch, DataLoader, logitcrossentropy, @functor, onecold
using Statistics, Random, Printf, NNlib

# --- 1. CONFIG ---
const SEQ_LEN = 250
const VOCAB_SIZE = 5 # A, C, G, T, N
const D_MODEL = 64
const N_HEADS = 4
const D_HEAD = div(D_MODEL, N_HEADS)

# --- 2. TRANSFORMER CORE (Flux-Native) ---

struct MultiHeadSelfAttention
    Wq; Wk; Wv; Wo
end
@functor MultiHeadSelfAttention

function MultiHeadSelfAttention(d_model::Int)
    return MultiHeadSelfAttention(
        Dense(d_model, d_model), Dense(d_model, d_model),
        Dense(d_model, d_model), Dense(d_model, d_model)
    )
end

function (m::MultiHeadSelfAttention)(x)
    b_size = size(x, 3)
    q, k, v = m.Wq(x), m.Wk(x), m.Wv(x)

    function prepare(t)
        t = reshape(t, D_HEAD, N_HEADS, SEQ_LEN, b_size)
        return reshape(permutedims(t, (1, 3, 2, 4)), D_HEAD, SEQ_LEN, :)
    end

    q_h, k_h, v_h = prepare(q), prepare(k), prepare(v)
    scores = batched_mul(permutedims(q_h, (2, 1, 3)), k_h) ./ sqrt(Float32(D_HEAD))
    attn = softmax(scores, dims=1)
    out = batched_mul(v_h, attn)
    
    out = reshape(out, D_HEAD, SEQ_LEN, N_HEADS, b_size)
    return m.Wo(reshape(permutedims(out, (1, 3, 2, 4)), D_MODEL, SEQ_LEN, b_size))
end

struct TransformerBlock
    mha; norm1; ff; norm2
end
@functor TransformerBlock

function TransformerBlock(d_model)
    return TransformerBlock(
        MultiHeadSelfAttention(d_model),
        LayerNorm(d_model),
        Chain(Dense(d_model, 4d_model, relu), Dense(4d_model, d_model)),
        LayerNorm(d_model)
    )
end

function (t::TransformerBlock)(x)
    x = t.norm1(x .+ t.mha(x))
    x = t.norm2(x .+ t.ff(x))
    return x
end

# --- 3. DIFFUSION MODEL ---

struct DenoisingTransformer
    embed; pos_enc; layers; out_head
end
@functor DenoisingTransformer

function (m::DenoisingTransformer)(x)
    h = m.embed(x) .+ m.pos_enc
    h = m.layers(h)
    return m.out_head(h) 
end

function build_diffusion_model()
    return DenoisingTransformer(
        Embedding(VOCAB_SIZE, D_MODEL),
        randn(Float32, D_MODEL, SEQ_LEN),
        Chain(TransformerBlock(D_MODEL), TransformerBlock(D_MODEL)),
        Dense(D_MODEL, VOCAB_SIZE) 
    )
end

# --- 4. DATA & UTILS ---

function corrupt_sequence(x, noise_level=0.2)
    noisy_x = copy(x)
    mask = rand(size(x)...) .< noise_level
    noisy_x[mask] .= 5 # Use 'N' token as mask
    return noisy_x
end

# Simulated genomic data (5,000 sequences)
function get_genomic_data()
    return rand(1:VOCAB_SIZE, SEQ_LEN, 5000)
end

# --- 5. MAIN PIPELINE ---

function main()
    Random.seed!(42)
    xt = get_genomic_data()
    
    model = build_diffusion_model()
    opt_state = Flux.setup(Adam(5e-4), model)

    println("Training Genomic Diffusion Model...")

    for epoch in 1:10
        noise_level = 0.2
        noisy_xt = corrupt_sequence(xt, noise_level)
        target = onehotbatch(xt, 1:VOCAB_SIZE)

        loss, grads = Flux.withgradient(model) do m
            y_hat = m(noisy_xt)
            logitcrossentropy(y_hat, target)
        end
        
        Flux.update!(opt_state, model, grads[1])
        @printf("Epoch %2d | Loss: %.4f\n", epoch, loss)
    end

    # Sampling: Generate from 100% noise
    println("\nGenerating Synthetic Promoter...")
    sample_noise = fill(5, SEQ_LEN, 1)
    logits = model(sample_noise)
    indices = onecold(logits, 1:VOCAB_SIZE)
    
    bases = ['A', 'C', 'G', 'T', 'N']
    println("Sequence: ", join([bases[i] for i in indices]))
end

main()

