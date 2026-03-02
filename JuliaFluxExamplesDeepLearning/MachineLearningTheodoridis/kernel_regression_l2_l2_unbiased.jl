using LinearAlgebra
using Random

# --- Kernel ---
abstract type Kernel end
struct Gaussian <: Kernel
    sigma::Float64
end

function kappa(x, y, k::Gaussian)
    return exp(-norm(x - y)^2 / (2*k.sigma^2))
end

function build_kernel_matrix(X, kernel::Kernel)
    M = size(X,1)
    K = zeros(Float64, M, M)
    for i in 1:M
        for j in 1:i
            K[i,j] = kappa(X[i,:], X[j,:], kernel)
            K[j,i] = K[i,j]
        end
    end
    return K
end

# --- SMO State ---
mutable struct SMOState
    a::Vector{Float64}
    b::Float64
    u::Vector{Float64}
end

function SMOState(M)
    SMOState(zeros(M), 0.0, zeros(M))
end

# --- SMO Step ---
function take_step!(i,j,y,C,eps,state,K)
    if i==j return 0 end
    ai_old, aj_old = state.a[i], state.a[j]
    yi, yj = y[i], y[j]
    Ei, Ej = state.u[i]-yi, state.u[j]-yj
    s = yi*yj

    if yi!=yj
        L = max(0.0, aj_old - ai_old)
        H = min(C, C + aj_old - ai_old)
    else
        L = max(0.0, ai_old + aj_old - C)
        H = min(C, ai_old + aj_old)
    end
    if L>=H return 0 end

    kii, kjj, kij = K[i,i], K[j,j], K[i,j]
    eta = kii+kjj-2*kij
    if eta<=1e-12 return 0 end

    aj_new = clamp(aj_old + yj*(Ei-Ej)/eta, L,H)
    if abs(aj_new-aj_old)<eps*(aj_new+aj_old+eps) return 0 end

    ai_new = ai_old + s*(aj_old - aj_new)

    # update bias
    b_old = state.b
    b1 = Ei + yi*(ai_new-ai_old)*kii + yj*(aj_new-aj_old)*kij + b_old
    b2 = Ej + yi*(ai_new-ai_old)*kij + yj*(aj_new-aj_old)*kjj + b_old
    if 0<ai_new<C
        state.b = b1
        elseif 0<aj_new<C
        state.b = b2
    else
        state.b = (b1+b2)/2
    end

    state.a[i], state.a[j] = ai_new, aj_new

    delta_i = (ai_new-ai_old)*yi
    delta_j = (aj_new-aj_old)*yj
    delta_b = state.b - b_old
    for k in 1:length(y)
        state.u[k] += delta_i*K[i,k] + delta_j*K[j,k] - delta_b
    end

    return 1
end

# --- SMO Solver ---
function smo_classification(X,y,C,eps,kernel::Kernel)
    M = size(X,1)
    state = SMOState(M)
    K = build_kernel_matrix(X,kernel)
    state.u .= 0.0

    passes = 0
    max_passes = 10
    while passes<max_passes
        num_changed = 0
        for i in 1:M
            for j in 1:M
                num_changed += take_step!(i,j,y,C,eps,state,K)
            end
        end
        passes = (num_changed==0) ? passes+1 : 0
    end

    return state.a, state.b
end

# --- MAIN ---
function main()
    Random.seed!(123)

    # <--- THIS IS CRUCIAL --->
    # Must have enough rows: 30 samples, 2 features
    X = [randn(15,2).-2.5; randn(15,2).+2.5]  # 30x2
    y = vcat(fill(-1.0,15), fill(1.0,15))    # length 30

    C = 1.0
    eps = 1e-4
    kernel = Gaussian(1.0)

    println("Training SVM...")
    a_sol, b_sol = smo_classification(X,y,C,eps,kernel)

    println("Training complete.")
    println("Support vectors: ", count(x->x>1e-5,a_sol))
    println("Bias b: ", round(b_sol,digits=4))

    test_pt = [2.0,2.0]
    score = sum(a_sol .* y .* [kappa(X[i,:],test_pt,kernel) for i in 1:size(X,1)]) - b_sol
        println("Test point: ", test_pt)
        println("Decision score: ", round(score,digits=4))
        println("Predicted class: ", score>0 ? "+1" : "-1")

        return a_sol, b_sol, score
    end

    a_sol, b_sol, score = main()


