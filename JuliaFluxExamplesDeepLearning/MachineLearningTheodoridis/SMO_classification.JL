using LinearAlgebra
using Statistics
using Random

# -------------------------------------------------------------------
# 1. Kernel Infrastructure
# -------------------------------------------------------------------
abstract type Kernel end

struct Gaussian <: Kernel
    sigma::Float64
end

struct LinearKernel <: Kernel end

struct Poly <: Kernel
    d::Int
end

# Multi-dispatch kappa function
function kappa(x, y, k::Gaussian)
    return exp(-norm(x - y)^2 / k.sigma^2)
end

function kappa(x, y, ::LinearKernel)
    return dot(x, y)
end

function kappa(x, y, k::Poly)
    return (1 + dot(x, y))^k.d
end

# Efficient Kernel Matrix Builder
function build_kernel_matrix(X, kernel::Kernel)
    M = size(X, 1)
    K = zeros(M, M)
    for j in 1:M
        for i in 1:j
            val = kappa(X[i, :], X[j, :], kernel)
            K[i, j] = K[j, i] = val
        end
    end
    return K
end

# -------------------------------------------------------------------
# 2. SMO State Management
# -------------------------------------------------------------------
mutable struct SMOState
    a::Vector{Float64}
    b::Float64
    u::Vector{Float64}
    KKT::Vector{Int}
    NB::Vector{Int}
    M::Int

    function SMOState(M)
        new(zeros(M), 0.0, zeros(M), zeros(Int, M), zeros(Int, M), M)
    end
end

# Update Metadata (KKT and Non-Bound status)
function update_metadata!(state::SMOState, y, C, tol)
    for k in 1:state.M
        r2 = (state.u[k] - y[k]) * y[k]
        # KKT conditions: 0 if not satisfied
        state.KKT[k] = ((r2 < -tol && state.a[k] < C) || (r2 > tol && state.a[k] > 0)) ? 0 : 1
        # NB: 1 if non-bound (0 < a < C)
        state.NB[k] = (state.a[k] > 0 && state.a[k] < C) ? 1 : 0
    end
end

# -------------------------------------------------------------------
# 3. SMO Core Logic (takeStep and examineExample)
# -------------------------------------------------------------------

function take_step!(i, j, y, C, epsilon, state::SMOState, K_mat)
    if i == j return 0 end

    E_i = state.u[i] - y[i]
    E_j = state.u[j] - y[j]
    s = y[i] * y[j]

    # Compute L and H bounds
    if y[i] != y[j]
        L = max(0.0, state.a[j] - state.a[i])
        H = min(C, C + state.a[j] - state.a[i])
    else
        L = max(0.0, state.a[j] + state.a[i] - C)
        H = min(C, state.a[j] + state.a[i])
    end
    if L == H return 0 end

    kii, kjj, kij = K_mat[i,i], K_mat[j,j], K_mat[i,j]
    eta = kii + kjj - 2*kij
    aj_old, ai_old = state.a[j], state.a[i]

    aj_new = 0.0
    if eta > 0
        aj_new = clamp(aj_old + y[j] * (E_i - E_j) / eta, L, H)
    else
        # Handle non-positive definite cases (numerical error handling)
        f1 = y[i] * (E_i + state.b) - ai_old * kii - s * aj_old * kij
        f2 = y[j] * (E_j + state.b) - s * ai_old * kij - aj_old * kjj
        L1 = ai_old + s * (aj_old - L)
        H1 = ai_old + s * (aj_old - H)
        Psi_L = L1*f1 + L*f2 + 0.5*L1^2*kii + 0.5*L^2*kjj + s*L*L1*kij
        Psi_H = H1*f1 + H*f2 + 0.5*H1^2*kii + 0.5*H^2*kjj + s*H*H1*kij
        aj_new = (Psi_L < Psi_H - epsilon) ? L : (Psi_L > Psi_H + epsilon) ? H : aj_old
    end

    if abs(aj_new - aj_old) < epsilon * (aj_new + aj_old + epsilon)
        return 0
    end

    ai_new = ai_old + s * (aj_old - aj_new)

    # Update threshold b
    b1 = E_i + y[i]*(ai_new - ai_old)*kii + y[j]*(aj_new - aj_old)*kij + state.b
    b2 = E_j + y[i]*(ai_new - ai_old)*kij + y[j]*(aj_new - aj_old)*kjj + state.b
    state.b = (0 < ai_new < C) ? b1 : (0 < aj_new < C) ? b2 : (b1 + b2) / 2

    # Update a and prediction cache u
    state.a[i], state.a[j] = ai_new, aj_new
    state.u .= (K_mat * (state.a .* y)) .- state.b

    update_metadata!(state, y, C, 0.001)
    return 1
end

function examine_example!(j, y, C, epsilon, state::SMOState, K_mat)
    if state.KKT[j] == 0
        # Second choice heuristic
        E_j = state.u[j] - y[j]
        i = argmax([abs(state.u[k] - y[k] - E_j) for k in 1:state.M])
        if take_step!(i, j, y, C, epsilon, state, K_mat) == 1 return 1 end

        # Loop over non-bound indices
        indices = shuffle(collect(1:state.M))
        for idx in indices
            if state.NB[idx] == 1
                if take_step!(idx, j, y, C, epsilon, state, K_mat) == 1 return 1 end
            end
        end

        # Loop over all indices
        for idx in indices
            if take_step!(idx, j, y, C, epsilon, state, K_mat) == 1 return 1 end
        end
    end
    return 0
end

# -------------------------------------------------------------------
# 4. Final Solver and Example Call
# -------------------------------------------------------------------

function smo_classification(X, y, C, epsilon, kernel::Kernel)
    M = length(y)
    state = SMOState(M)
    K_mat = build_kernel_matrix(X, kernel)

    # Initialize metadata
    state.u .= -state.b # Initial u = -b (where b=0)
    update_metadata!(state, y, C, 0.001)

    numChanged = 0
    examineAll = true

    while numChanged > 0 || examineAll
        numChanged = 0
        if examineAll
            for i in 1:M
                numChanged += examine_example!(i, y, C, epsilon, state, K_mat)
            end
        else
            for i in 1:M
                if state.NB[i] == 1
                    numChanged += examine_example!(i, y, C, epsilon, state, K_mat)
                end
            end
        end

        if examineAll
            examineAll = false
        elseif numChanged == 0
            examineAll = true
        end
    end

    return state.a, state.b
end

# --- RUNNING THE CODE ---

# Create dummy binary data
Random.seed!(42)
X_train = [randn(10, 2) .- 2.0; randn(10, 2) .+ 2.0]
y_train = [fill(-1.0, 10); fill(1.0, 10)]

# Parameters
C_param = 1.0
eps_param = 0.001
my_kernel = Gaussian(1.5)

# Train
a_sol, b_sol = smo_classification(X_train, y_train, C_param, eps_param, my_kernel)

# Output Results
sv_count = count(x -> x > 1e-5, a_sol)
println("SMO Classification Finished.")
println("Number of Support Vectors: ", sv_count)
println("Bias (b): ", round(b_sol, digits=4))


