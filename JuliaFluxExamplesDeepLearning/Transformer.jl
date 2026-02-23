using Flux, Optimisers, ProgressMeter

# 1. Define a helper to extract just the output from MHA
# MHA returns (content, weights), we only want content for the next layer.
struct TransformerBlock
    mha
    ln1
    ffn
    ln2
end

# Make it a Flux functional layer
Flux.@layer TransformerBlock

function (m::TransformerBlock)(x)
    # 1. Multi-Head Attention (Returns a tuple, we take the first element)
    attn_out = m.mha(x)[1]
    x = m.ln1(x + attn_out) # Residual connection + Norm

    # 2. Feed Forward
    ffn_out = m.ffn(x)
    x = m.ln2(x + ffn_out) # Residual connection + Norm
    return x
end

# 2. Build the model
mha = MultiHeadAttention(256, nheads=4)
ln1 = LayerNorm(256)
ffn = Chain(Dense(256 => 512, relu), Dense(512 => 256))
ln2 = LayerNorm(256)

model = TransformerBlock(mha, ln1, ffn, ln2)

# 3. Training setup
X = randn(Float32, 256, 10, 32)
Y = randn(Float32, 256, 10, 32)

opt_state = Optimisers.setup(Optimisers.Adam(1e-4), model)

# 4. Loop
for epoch in 1:10
    loss, grads = Flux.withgradient(model) do m
        Flux.mse(m(X), Y)
    end
    opt_state, model = Optimisers.update(opt_state, model, grads[1])
    println("Epoch $epoch: $loss")
end

