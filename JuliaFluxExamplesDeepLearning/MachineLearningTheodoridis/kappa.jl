using LinearAlgebra

# -------------------------------------------------------------------
# 1. Define Kernel Types (Multiple Dispatch)
# -------------------------------------------------------------------
abstract type Kernel end

struct Gaussian <: Kernel
    sigma::Float64
end

struct GaussianC <: Kernel
    sigma::Float64
end

struct LinearKernel <: Kernel end

struct Poly <: Kernel
    d::Int
end

struct PolyC <: Kernel
    d::Int
end

# -------------------------------------------------------------------
# 2. Define Kappa Functions
# -------------------------------------------------------------------

# Standard Gaussian: exp(-||x-y||² / σ²)
function kappa(x, y, k::Gaussian)
    return exp(-norm(x - y)^2 / k.sigma^2)
end

# Complex Gaussian: 2 * real(exp(-sum((x - conj(y))²) / σ²))
function kappa(x, y, k::GaussianC)
    exponent = sum((x .- conj.(y)).^2)
    return 2 * real(exp(-exponent / k.sigma^2))
end

# Linear: x' * conj(y)
function kappa(x, y, ::LinearKernel)
    return dot(x, y)
end

# Polynomial: (1 + x' * conj(y))^d
function kappa(x, y, k::Poly)
    return (1 + dot(x, y))^k.d
end

# Complex Polynomial: 2 * real((1 + x' * conj(y))^d)
function kappa(x, y, k::PolyC)
    return 2 * real((1 + dot(x, y))^k.d)
end

# -------------------------------------------------------------------
# 3. Execution & Demonstration
# -------------------------------------------------------------------

# Create some dummy data
vec_a = [1.0, 2.0, 3.0]
vec_b = [1.1, 1.9, 3.2]

# Complex data for the _c versions
c_a = [1.0 + 1im, 2.0 - 0.5im]
c_b = [1.1 - 0.2im, 1.8 + 0.1im]

println("--- Results ---")

# Calling Gaussian
g_val = kappa(vec_a, vec_b, Gaussian(0.5))
println("Gaussian (σ=0.5):  ", g_val)

# Calling Linear
l_val = kappa(vec_a, vec_b, LinearKernel())
println("Linear:           ", l_val)

# Calling Poly
p_val = kappa(vec_a, vec_b, Poly(3))
println("Polynomial (d=3): ", p_val)

# Calling Complex Gaussian
gc_val = kappa(c_a, c_b, GaussianC(1.0))
println("Complex Gaussian: ", gc_val)

# Calling Complex Poly
pc_val = kappa(c_a, c_b, PolyC(2))
println("Complex Poly:     ", pc_val)


