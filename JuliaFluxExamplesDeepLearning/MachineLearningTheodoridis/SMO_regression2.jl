using LinearAlgebra
using Statistics
using Random

# -------------------------------------------------------------------
# 1. Kernel Infrastructure
# -------------------------------------------------------------------
abstract type Kernel end
struct Gaussian <: Kernel sigma::Float64 end

function kappa(x, y, k::Gaussian)
    return exp(-norm(x - y)^2 / k.sigma^2)
end

function build_kernel_matrix(X, kernel::Kernel)
    # X is M x N (Samples x Features)
    M = size(X, 1)
    K = zeros(M, M)
    for j in 1:M, i in 1:j
        val = kappa(X[i, :], X[j, :], kernel)
        K[i, j] = K[j, i] = val
    end
    return K
end

# -------------------------------------------------------------------
# 2. SMO State Container
# -------------------------------------------------------------------
mutable struct SMOState
    a1::Vector{Float64}
    a2::Vector{Float64}
    b::Float64
    u::Vector{Float64}
    KKT::Vector{Int}
    NB::Vector{Int}
    M::Int
    SMOState(M) = new(zeros(M), zeros(M), 0.0, zeros(M), zeros(Int, M), zeros(Int, M), M)
end

# -------------------------------------------------------------------
# 3. Helper Functions (Fixed Indexing)
# -------------------------------------------------------------------

function update_KKT!(state::SMOState, y, C, epsilon)
    for i in 1:state.M
        E_i = state.u[i] - y[i]
        if ((E_i >= epsilon) && (state.a2[i] < C)) ||
            ((E_i < epsilon) && (state.a2[i] > 0)) ||
                ((-E_i >= epsilon) && (state.a1[i] < C)) ||
                    ((-E_i < epsilon) && (state.a1[i] > 0))
                state.KKT[i] = 0
        else
            state.KKT[i] = 1
        end
    end
end

function take_step!(i, j, y, C, epsilon, state::SMOState, K_mat)
    if i == j return 0 end
    tol = 0.001

    E_i, E_j = y[i] - state.u[i], y[j] - state.u[j]
    gamma = state.a1[i] + state.a1[j] - state.a2[i] - state.a2[j]

    kii, kjj, kij = K_mat[i,i], K_mat[j,j], K_mat[i,j]
    eta = kii + kjj - 2*kij
    if eta <= 0 eta = 1e-3 end

    a1i_old, a1j_old = state.a1[i], state.a1[j]
    a2i_old, a2j_old = state.a2[i], state.a2[j]

    # Simplified Case 1 logic (Ported from your MATLAB while loop)
    L = max(0.0, gamma - C)
    H = min(gamma, C)

    if L < H
        state.a1[j] = clamp(a1j_old - (E_i - E_j)/eta, L, H)
        state.a1[i] = a1i_old - (state.a1[j] - a1j_old)
    end

    # Convergence Check
    diff = abs(state.a1[i]-a1i_old) + abs(state.a1[j]-a1j_old)
    if diff < tol return 0 end

    # Update u vector (prediction cache)
    for k in 1:state.M
        state.u[k] += (state.a1[i] - state.a2[i] - a1i_old + a2i_old) * K_mat[k, i] +
            (state.a1[j] - state.a2[j] - a1j_old + a2j_old) * K_mat[k, j]
    end

    update_KKT!(state, y, C, epsilon)
    return 1
end

function examine_example!(j, y, C, epsilon, state::SMOState, K_mat)
    if state.KKT[j] == 0
        # Simple heuristic: try all other indices in random order
        indices = shuffle(collect(1:state.M))
        for i in indices
            if take_step!(i, j, y, C, epsilon, state, K_mat) == 1
                return 1
            end
        end
    end
    return 0
end

# -------------------------------------------------------------------
# 4. Main Solver
# -------------------------------------------------------------------
function smo_regression(X, y, C, epsilon, kernel::Kernel)
    M = size(X, 1) # Samples
    state = SMOState(M)
    K_mat = build_kernel_matrix(X, kernel)

    # Initialize u with b=0
    state.u .= 0.0
    update_KKT!(state, y, C, epsilon)

    numChanged = 0
    examineAll = true
    for iter in 1:100 # Safety limit
        numChanged = 0
        if examineAll
            for i in 1:M
                numChanged += examine_example!(i, y, C, epsilon, state, K_mat)
            end
        else
            for i in 1:M
                if state.a1[i] > 0 || state.a2[i] > 0 # Non-bound check
                    numChanged += examine_example!(i, y, C, epsilon, state, K_mat)
                end
            end
        end

        if examineAll
            examineAll = false
            elseif numChanged == 0
            examineAll = true
        end
        if numChanged == 0 && examineAll == false break end
    end

    rmse = sqrt(mean((state.u .- y).^2))
    return state.a1, state.a2, state.b, rmse
end

# -------------------------------------------------------------------
# 5. Correct Call Example
# -------------------------------------------------------------------

# Data: 10 samples, 1 feature
X = reshape(collect(1.0:10.0), 10, 1)
y = [2.1, 3.9, 6.2, 8.0, 10.5, 12.1, 13.8, 16.1, 17.9, 20.2]

C = 1.0
epsilon = 0.1
kernel = Gaussian(2.0)

# The call that now works:
a1, a2, b, rmse = smo_regression(X, y, C, epsilon, kernel)

println("Success! RMSE: ", round(rmse, digits=4))

