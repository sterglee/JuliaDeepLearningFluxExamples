using Flux
using LinearAlgebra
using Statistics
using Random

# ==========================================
# 1. Setup Data and Parameters
# ==========================================
# Sequence length (N) and input dimension (D)
N = 6
D = 8

# Create random inputs: (D x N) matrix
Random.seed!(3)
X = randn(Float32, D, N)

# Function to initialize random parameters for a single head
function init_head_params(d_in, d_out)
    # Omega matrices (D_out x D_in), Beta biases (D_out x 1)
    return (
        randn(Float32, d_out, d_in), # omega_q
        randn(Float32, d_out, d_in), # omega_k
        randn(Float32, d_out, d_in), # omega_v
        randn(Float32, d_out, 1),    # beta_q
        randn(Float32, d_out, 1),    # beta_k
        randn(Float32, d_out, 1)     # beta_v
        )
end

# Initialize parameters for two heads (each outputting dimension D/2 = 4)
Random.seed!(0)
D_head = Int(D/2)
params1 = init_head_params(D, D_head)
params2 = init_head_params(D, D_head)

# Final concatenation matrix Omega_c (D x D)
omega_c = randn(Float32, D, D)

# ==========================================
# 2. Multihead Self-Attention Logic
# ==========================================

# Single head scaled dot-product attention
function head_attention(X, ω_q, ω_k, ω_v, β_q, β_k, β_v)
    d_h = size(ω_q, 1)

    # 1. Compute Q, K, V
    Q = ω_q * X .+ β_q
    K = ω_k * X .+ β_k
    V = ω_v * X .+ β_v

    # 2. Scaled dot-product scores
    scaled_scores = (K' * Q) ./ sqrt(Float32(d_h))

    # 3. Softmax and weighted sum
    attention_matrix = softmax(scaled_scores)
    return V * attention_matrix
end

function multihead_scaled_self_attention(X, p1, p2, ω_c)
    # Run Head 1
    X_prime_1 = head_attention(X, p1...)

    # Run Head 2
    X_prime_2 = head_attention(X, p2...)

    # Concatenate heads vertically (since signals are in columns)
    X_concat = vcat(X_prime_1, X_prime_2)

    # Final linear transformation
    return ω_c * X_concat
end

# ==========================================
# 3. Execution and Verification
# ==========================================

X_prime = multihead_scaled_self_attention(X, params1, params2, omega_c)

println("Multihead Attention Output (X'):")
display(X_prime)

# Verification against true values from notebook
# (Note: These true values depend on the specific random seed/numpy behavior)
println("\nOutput Dimensions: ", size(X_prime)) # Should be (8, 6)

