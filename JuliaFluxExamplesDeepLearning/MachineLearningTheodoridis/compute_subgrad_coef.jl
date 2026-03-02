using LinearAlgebra
using Statistics
using Random

# ============================================================
# 1. Kernel Infrastructure (Multiple Dispatch)
# ============================================================
abstract type Kernel end

struct Gaussian <: Kernel
    sigma::Float64
end

struct LinearKernel <: Kernel end

function kappa(x, y, k::Gaussian)
    # Standard RBF: exp(-||x-y||² / (2σ²))
    return exp(-norm(x - y)^2 / (2 * k.sigma^2))
end

function kappa(x, y, ::LinearKernel)
    return dot(x, y)
end

function build_kernel_matrix(X, kernel::Kernel)
    M = size(X, 1) # Number of samples
    K = zeros(Float64, M, M)
    for j in 1:M, i in 1:j
        val = kappa(X[i, :], X[j, :], kernel)
        K[i, j] = K[j, i] = val
    end
    return K
end

# ============================================================
# 2. SMO State and Logic
# ============================================================
mutable struct SMOState
    a::Vector{Float64}
    b::Float64
    u::Vector{Float64} # Cache for f(x_i)
    M::Int
end

function SMOState(M)
    SMOState(zeros(M), 0.0, zeros(M), M)
end

function take_step!(i, j, y, C, eps, state::SMOState, K)
    if i == j return 0 end

    ai_old, aj_old = state.a[i], state.a[j]
    yi, yj = y[i], y[j]
    Ei, Ej = state.u[i] - yi, state.u[j] - yj
    s = yi * yj

    # 1. Compute Bounds L and H
    if yi != yj
        L = max(0.0, aj_old - ai_old)
        H = min(C, C + aj_old - ai_old)
    else
        L = max(0.0, ai_old + aj_old - C)
        H = min(C, ai_old + aj_old)
    end
    if L >= H return 0 end

    # 2. Compute second derivative (eta)
    kii, kjj, kij = K[i,i], K[j,j], K[i,j]
    eta = kii + kjj - 2*kij
    if eta <= 1e-12 return 0 end

    # 3. Update aj and clip
    aj_new = clamp(aj_old + yj * (Ei - Ej) / eta, L, H)
    if abs(aj_new - aj_old) < eps * (aj_new + aj_old + eps)
        return 0
    end

    # 4. Update ai
    ai_new = ai_old + s * (aj_old - aj_new)

    # 5. Update threshold (bias) b
    b_old = state.b
    b1 = Ei + yi*(ai_new - ai_old)*kii + yj*(aj_new - aj_old)*kij + b_old
    b2 = Ej + yi*(ai_new - ai_old)*kij + yj*(aj_new - aj_old)*kjj + b_old

    if 0 < ai_new < C
        state.b = b1
        elseif 0 < aj_new < C
        state.b = b2
    else
        state.b = (b1 + b2) / 2
    end

    # 6. Save alphas and update prediction cache incrementally (O(M))
    state.a[i], state.a[j] = ai_new, aj_new

    delta_i = (ai_new - ai_old) * yi
    delta_j = (aj_new - aj_old) * yj
    delta_b = state.b - b_old

    for k in 1:state.M
        state.u[k] += delta_i * K[i, k] + delta_j * K[j, k] - delta_b
    end

    return 1
end

# ============================================================
# 3. Main Solver
# ============================================================
function smo_classification(X, y, C, eps, kernel::Kernel)
    M = size(X, 1)
    state = SMOState(M)
    K_mat = build_kernel_matrix(X, kernel)

    # Initial f(x) = 0 for all points
    state.u .= 0.0

    passes = 0
    max_passes = 10
    while passes < max_passes
        num_changed = 0
        for i in 1:M
            for j in 1:M
                num_changed += take_step!(i, j, y, C, eps, state, K_mat)
            end
        end
        passes = (num_changed == 0) ? passes + 1 : 0
    end

    return state.a, state.b
end

# ============================================================
# 4. Execution Block
# ============================================================
function main()
    Random.seed!(42)

    # Generate Synthetic Data: 40 points total, 2 features
    X = [randn(20, 2) .- 2.5; randn(20, 2) .+ 2.5]
    y = vcat(fill(-1.0, 20), fill(1.0, 20))

    C = 1.0
    eps = 1e-4
    kernel = Gaussian(1.0)

    println("--- Training SVM via SMO ---")
    @time a_sol, b_sol = smo_classification(X, y, C, eps, kernel)

    println("\nResults:")
    println("----------------------------")
    println("Support Vectors: ", count(x -> x > 1e-5, a_sol))
    println("Bias (b): ", round(b_sol, digits=4))

    # Test Prediction
    test_pt = [2.0, 2.0]
    # Decision Score = Σ (αi * yi * K(xi, test)) - b
    k_vals = [kappa(X[i, :], test_pt, kernel) for i in 1:size(X, 1)]
        score = dot(a_sol .* y, k_vals) - b_sol

        println("Test Point: ", test_pt)
        println("Score: ", round(score, digits=4))
        println("Predicted Class: ", score > 0 ? "+1" : "-1")

        return a_sol, b_sol
    end

    # Start the script
    a_sol, b_sol = main()

