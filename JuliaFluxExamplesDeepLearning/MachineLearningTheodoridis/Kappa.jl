using LinearAlgebra
using Statistics
using Random

# ============================================================
# 1. Kernel Infrastructure (Multiple Dispatch)
# ============================================================
abstract type Kernel end

struct Gaussian <: Kernel sigma::Float64 end
struct Polynomial <: Kernel d::Int end
struct VovkInf <: Kernel p::Int end
struct VovkPoly <: Kernel end
struct Dirichlet <: Kernel n::Int end
struct Periodic <: Kernel n::Int; sigma::Float64 end

# --- Kernel Dispatch Methods ---

function kappa(x, y, k::Gaussian)
    return exp(-norm(x - y)^2 / k.sigma^2)
end

function kappa(x, y, k::Polynomial)
    return (dot(x, y) + 1)^k.d
end

function kappa(x, y, k::VovkInf)
    xy = dot(x, y)
    return (1 - xy)^k.p / (1 - xy)
end

function kappa(x, y, ::VovkPoly)
    return 1 / (1 - dot(x, y))
end

function kappa(x, y, k::Dirichlet)
    d_sq = norm(x - y)^2
    return sin((2 * k.n + 1) * d_sq / 2) / (sin(d_sq / 2) + 0.0001)
end

function kappa(x, y, k::Periodic)
    d_sq = norm(x - y)^2
    val = 0.0
    for j in 0:k.n
        val += exp(-j^2 * k.sigma^2 / 2) * cos(j * d_sq)
    end
    return val
end

function build_kernel_matrix(X, kernel::Kernel)
    M = size(X, 1)
    K = zeros(Float64, M, M)
    for j in 1:M, i in 1:j
        val = kappa(X[i, :], X[j, :], kernel)
        K[i, j] = K[j, i] = val
    end
    return K
end

# ============================================================
# 2. SMO State and Solver Logic
# ============================================================
mutable struct SMOState
    a::Vector{Float64}
    b::Float64
    u::Vector{Float64}
    M::Int
end

function take_step!(i, j, y, C, eps, state::SMOState, K)
    if i == j return 0 end

    ai_old, aj_old = state.a[i], state.a[j]
    yi, yj = y[i], y[j]
    Ei, Ej = state.u[i] - yi, state.u[j] - yj
    s = yi * yj

    if yi != yj
        L, H = max(0.0, aj_old - ai_old), min(C, C + aj_old - ai_old)
    else
        L, H = max(0.0, ai_old + aj_old - C), min(C, ai_old + aj_old)
    end
    if L >= H return 0 end

    eta = K[i,i] + K[j,j] - 2*K[i,j]
    if eta <= 1e-12 return 0 end

    aj_new = clamp(aj_old + yj * (Ei - Ej) / eta, L, H)
    if abs(aj_new - aj_old) < eps * (aj_new + aj_old + eps) return 0 end

    ai_new = ai_old + s * (aj_old - aj_new)

    b_old = state.b
    b1 = Ei + yi*(ai_new - ai_old)*K[i,i] + yj*(aj_new - aj_old)*K[i,j] + b_old
    b2 = Ej + yi*(ai_new - ai_old)*K[i,j] + yj*(aj_new - aj_old)*K[j,j] + b_old
    state.b = (0 < ai_new < C) ? b1 : (0 < aj_new < C) ? b2 : (b1 + b2) / 2

    delta_i, delta_j, delta_b = (ai_new - ai_old)*yi, (aj_new - aj_old)*yj, state.b - b_old
    state.a[i], state.a[j] = ai_new, aj_new

    # Incremental update of prediction cache
    for k in 1:state.M
        state.u[k] += delta_i * K[i, k] + delta_j * K[j, k] - delta_b
    end
    return 1
end

function smo_classification(X, y, C, eps, kernel::Kernel)
    M = size(X, 1)
    state = SMOState(zeros(M), 0.0, zeros(M), M)
    K_mat = build_kernel_matrix(X, kernel)

    passes, max_passes = 0, 10
    while passes < max_passes
        num_changed = 0
        for i in 1:M, j in 1:M
            num_changed += take_step!(i, j, y, C, eps, state, K_mat)
        end
        passes = (num_changed == 0) ? passes + 1 : 0
    end
    return state.a, state.b
end

# ============================================================
# 3. Main Execution
# ============================================================
function main()
    Random.seed!(42)

    # 1. Setup Data (30 samples, 2 features)
    X = [randn(15, 2) .- 2.0; randn(15, 2) .+ 2.0]
    y = vcat(fill(-1.0, 15), fill(1.0, 15))

    # 2. Choose Kernel (Change this to test different kernels)
    # Options: Gaussian(sigma), Polynomial(d), Periodic(n, sigma), etc.
    my_kernel = Gaussian(1.0)

    println("--- Training SVM with $(typeof(my_kernel)) kernel ---")
    a_sol, b_sol = smo_classification(X, y, 1.0, 1e-4, my_kernel)

    println("Training Complete.")
    println("Support Vectors: ", count(x -> x > 1e-5, a_sol))
    println("Bias (b): ", round(b_sol, digits=4))

    # 3. Test Prediction
    test_pt = [2.0, 2.0]
    k_vals = [kappa(X[i, :], test_pt, my_kernel) for i in 1:size(X, 1)]
        score = dot(a_sol .* y, k_vals) - b_sol

        println("----------------------------")
        println("Test Point: ", test_pt)
        println("Decision Score: ", round(score, digits=4))
        println("Predicted Class: ", score > 0 ? "+1" : "-1")

        return a_sol, b_sol
    end

    # Run the full script
    a_sol, b_sol = main()


