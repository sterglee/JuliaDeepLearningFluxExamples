using LinearAlgebra
using Statistics
using Random

"""
tls(A, b, thresh=eps())

Solves Ax ≈ b using Truncated Total Least Squares via SVD.
(Logic adapted for TLS)
"""
function tls(A, b, thresh=eps(Float64))
    m, n = size(A)
    # Augmented matrix [A | b]
    Z = hcat(A, b)
    U, S, V = svd(Z)

    # Identify singular values above threshold
    k = sum(S .< thresh)
    q = (n + 1) - k + 1
    if q > n + 1; q = n + 1; end

    V12 = V[1:n, q:end]
    V22 = V[n+1, q:end]

    # Compute TLS solution
    x = -V12 * V22' ./ norm(V22)^2
    return x[:]
end

function run_tls_comparison()
    # --- 1. Parameters ---
    N = 150             # Number of observations
    l = 90              # Problem size
    sv = 0.01f0         # Variance of additive noise
    sv2 = 0.2f0         # Input noise factor
    Iter = 100          # Number of iterations

    t1, t2, t3, t4 = zeros(l), zeros(l), zeros(l), zeros(l)
    theta = randn(l)    # True parameter

    println("Running TLS Comparison Simulation...")

    for i in 1:Iter
        # --- 2. Noise-Free Input Scenario ---
        A = randn(N, l)
        noise = sqrt(sv) * randn(N)
        y1 = A * theta + noise

        # OLS: that = (AᵀA)⁻¹Aᵀy1 (Implemented as A \ y1)
        t1 += A \ y1

        # --- 3. Noisy Input Scenario ---
        E = sv2 * randn(N, l)
        A2 = A + E # Corrupted data matrix

        # Biased OLS: solving with noisy A2
        t2 += A2 \ y1

        # --- 4. Total Least Squares (TLS) ---
        # Method A: Smallest Singular Value Compensation
        Ahatext = hcat(A2, y1)
        S_vals = svdvals(Ahatext)
        sl1 = minimum(filter(x -> x > 1e-10, S_vals)) # Smallest non-zero SV

        # that3 = (A2ᵀA2 - σ²I)⁻¹A2ᵀy1
        t3 += (A2' * A2 - sl1^2 * I) \ (A2' * y1)

        # Method B: Full TLS using SVD subspace
        t4 += tls(A2, y1)
    end

    # --- 5. Result Analysis ---
    # Calculate norms of the estimation errors
    println("Error Norms (Average over $Iter iterations):")
    println("OLS (Clean A):      ", norm(t1 ./ Iter - theta))
    println("OLS (Noisy A):      ", norm(t2 ./ Iter - theta))
    println("Compensated OLS:    ", norm(t3 ./ Iter - theta))
    println("Full TLS (SVD):     ", norm(t4 ./ Iter - theta))
end

run_tls_comparison()

