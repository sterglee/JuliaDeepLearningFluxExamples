using Flux
using LinearAlgebra

# 1. Define the Struct
struct DeepSeekMLA
    up_q      # Up-projection for Query
    up_kv     # Up-projection for Key-Value
    out_proj  # Output projection
    n_heads::Int
    h_dim::Int
end

# 2. Make it a Flux "Functor" so Flux can see the internal weights
Flux.@functor DeepSeekMLA

# 3. Define the Forward Pass (The "call" behavior)
function (m::DeepSeekMLA)(x, cos_p, sin_p)
    # ERROR FIX: Access fields using 'm.' prefix
    # 'up_q' is a field of the struct 'm', not a global variable.
    q = m.up_q(x)   
    kv = m.up_kv(x) 
    
    # 4. Reshape for Multi-Head: (head_dim, n_heads, seq_len)
    # seq_len is derived from the second dimension of the input x
    seq_len = size(x, 2)
    
    q_heads = reshape(q, m.h_dim, m.n_heads, seq_len)
    k_heads = reshape(kv, m.h_dim, m.n_heads, seq_len)
    
    # 5. Apply RoPE (Rotary Positional Embeddings)
    q_rope = apply_rope(q_heads, cos_p, sin_p)
    k_rope = apply_rope(k_heads, cos_p, sin_p)
    
    # 6. Attention Score Calculation
    # Aligning dimensions for batched_mul: (n_heads, seq_len, head_dim) * (n_heads, head_dim, seq_len)
    q_p = permutedims(q_rope, (2, 3, 1))
    k_p = permutedims(k_rope, (2, 1, 3))
    
    scores = batched_mul(q_p, k_p) ./ sqrt(Float32(m.h_dim))
    weights = softmax(scores, dims=3)
    
    # 7. Final Projection
    flat_weights = reshape(weights, m.n_heads * seq_len, seq_len)
    return m.out_proj(flat_weights)
end

# ---------------------------------------------------------
# Initialization Example (How to actually call it)
# ---------------------------------------------------------
d_model, n_heads, h_dim = 512, 8, 64

# Initialize the instance 'mla_instance'
mla_instance = DeepSeekMLA(
    Dense(d_model => n_heads * h_dim), # up_q
    Dense(d_model => n_heads * h_dim), # up_kv
    Dense(n_heads * 128 => d_model),   # out_proj (assuming seq_len 128)
    n_heads, 
    h_dim
)

# Dummy inputs
x_input = randn(Float32, d_model, 128)
c_p = randn(Float32, h_dim ÷ 2, 1, 128)
s_p = randn(Float32, h_dim ÷ 2, 1, 128)

# Call the instance like a function
output = mla_instance(x_input, c_p, s_p)
println("Success! Output size: ", size(output))

