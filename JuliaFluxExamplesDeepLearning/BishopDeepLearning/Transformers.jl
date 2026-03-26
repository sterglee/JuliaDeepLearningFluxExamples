using LinearAlgebra
using Random
using Statistics

# --------------------------
# Συναρτήσεις για multi-head attention
# --------------------------

# Scaled dot-product attention
function attention(Q::Matrix{Float64}, K::Matrix{Float64}, V::Matrix{Float64})
    dk = size(K, 2)
    scores = Q * K' / sqrt(dk)          # dot-product & scaling
    weights = exp.(scores)              # elementwise exp
    weights ./= sum(weights, dims=2)    # normalize κάθε σειρά (softmax)
    return weights * V
end

# Multi-head attention με keyword args
function multi_head_attention(X::Matrix{Float64}; H::Int=2, Dv::Union{Nothing, Int}=nothing)
    N, D = size(X)

    # Αν δεν δώσουμε Dv, το χωρίζουμε ισότιμα
    if Dv === nothing
        Dv = div(D, H)
    end

    heads = []

    for h in 1:H
        # Τυχαίες "learnable" βάσεις (τυπικά trainable, εδώ για demo)
        Wq = randn(D, Dv)
        Wk = randn(D, Dv)
        Wv = randn(D, Dv)

        Q = X * Wq
        K = X * Wk
        V = X * Wv

        push!(heads, attention(Q, K, V))
    end

    # Concatenate heads (οριζόντια)
    Y_concat = hcat(heads...)

    # Τελική γραμμική προβολή
    Wo = randn(H*Dv, D)
    Y = Y_concat * Wo

    return Y
end

# --------------------------
# Παράδειγμα χρήσης
# --------------------------

Random.seed!(42)
X = randn(5, 8)  # 5 tokens, 8 features

# Κλήση με keyword arguments
Y = multi_head_attention(X; H=3, Dv=4)
println("Διάσταση εξόδου: ", size(Y))
println(Y)

