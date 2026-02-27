using Flux
using LinearAlgebra
using Statistics
using Random

# ==========================================
# 1. Setup Data and Parameters
# ==========================================
# Number of inputs (N) and dimensions (D)
N = 3
D = 4

# Create random inputs: (D x N) matrix for efficiency in Julia (column-major)
# Equivalent to 'all_x' in the notebook
Random.seed!(3)
X = randn(Float32, D, N)

# Parameters (Weights and Biases)
Random.seed!(0)
omega_q = randn(Float32, D, D)
omega_k = randn(Float32, D, D)
omega_v = randn(Float32, D, D)

beta_q = randn(Float32, D, 1)
beta_k = randn(Float32, D, 1)
beta_v = randn(Float32, D, 1)

# ==========================================
# 2. Manual Logic (Equiv. to individual loops)
# ==========================================

# TODO: Compute queries, keys, and values (Eq 12.2 and 12.4)
# In Julia, we can broadcast or use matrix multiplication
Queries = omega_q * X .+ beta_q
Keys    = omega_k * X .+ beta_k
Values  = omega_v * X .+ beta_v

# ==========================================
# 3. Matrix-Based Self Attention (Eq 12.6 - 12.8)
# ==========================================

function self_attention(X, ω_v, ω_q, ω_k, β_v, β_q, β_k)
    # 1. Compute queries, keys, and values
    Q = ω_q * X .+ β_q
    K = ω_k * X .+ β_k
    V = ω_v * X .+ β_v

    # 2. Compute dot products (Attention Scores)
    # Result is an (N x N) matrix
    raw_attention = K' * Q

    # 3. Apply softmax per column (Eq 12.5)
    # Flux.softmax works on columns by default
    attention_matrix = softmax(raw_attention)

    # 4. Weight values by attentions (Eq 12.3)
    X_prime = V * attention_matrix

    return X_prime, attention_matrix
end

# ==========================================
# 4. Scaled Dot-Product Self Attention (Eq 12.9)
# ==========================================

function scaled_dot_product_self_attention(X, ω_v, ω_q, ω_k, β_v, β_q, β_k)
    d = size(X, 1) # Dimension D

    Q = ω_q * X .+ β_q
    K = ω_k * X .+ β_k
    V = ω_v * X .+ β_v

    # Compute dot products and scale by 1/√d
    # Scaling prevents the softmax from becoming too "extreme"
    scaled_scores = (K' * Q) ./ sqrt(d)

    attention_matrix = softmax(scaled_scores)
    X_prime = V * attention_matrix

    return X_prime, attention_matrix
end

# ==========================================
# 5. Execution and Validation
# ==========================================

X_prime_basic, att_basic = self_attention(X, omega_v, omega_q, omega_k, beta_v, beta_q, beta_k)
X_prime_scaled, att_scaled = scaled_dot_product_self_attention(X, omega_v, omega_q, omega_k, beta_v, beta_q, beta_k)

println("Output X' (Basic Attention):")
display(X_prime_basic)

println("\nAttention Matrix (Notice extreme values):")
display(att_basic)

println("\nOutput X' (Scaled Attention):")
display(X_prime_scaled)

println("\nAttention Matrix (Scaled - more balanced):")
display(att_scaled)

# Test for Permutation Covariance
# If we swap the first two columns of X, the output columns should also be swapped
X_permuted = X[:, [2, 1, 3]]
X_prime_perm, _ = scaled_dot_product_self_attention(X_permuted, omega_v, omega_q, omega_k, beta_v, beta_q, beta_k)

println("\nPermutation Covariance Check (Should be true):")
println(isapprox(X_prime_scaled[:, [2, 1, 3]], X_prime_perm))

