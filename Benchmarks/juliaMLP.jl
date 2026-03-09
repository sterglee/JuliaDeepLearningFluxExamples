
# Julia 16 s
# Java 17 s
#C++ 166.17 s

using LinearAlgebra, Random

function train_mlp_fast(epochs)
    # Use Float32 for a significant speed boost over Float64
    n_samples = 1000
    X = randn(Float32, 2, n_samples)
    # Pre-calculate labels
    y = reshape(Float32.( (X[1,:] .^ 2 + X[2,:] .^ 2) .> 1.0f0 ), 1, n_samples)

    # Dimensions
    in_d, hid_d, out_d = 2, 4, 1
    lr = 0.1f0

    # Weights
    W1 = randn(Float32, hid_d, in_d) .* sqrt(2f0/in_d)
    b1 = zeros(Float32, hid_d, 1)
    W2 = randn(Float32, out_d, hid_d) .* sqrt(1f0/hid_d)
    b2 = zeros(Float32, out_d, 1)

    # PRE-ALLOCATE EVERYTHING (The secret to beating C++/Java)
    Z1 = zeros(Float32, hid_d, n_samples)
    A1 = zeros(Float32, hid_d, n_samples)
    Z2 = zeros(Float32, out_d, n_samples)
    A2 = zeros(Float32, out_d, n_samples)
    dA2 = zeros(Float32, out_d, n_samples)
    dA1 = zeros(Float32, hid_d, n_samples)

    # We use a view or a transpose object to avoid copying memory
    Xt = collect(X')

    for i in 1:epochs
        # --- Forward Pass ---
        mul!(Z1, W1, X)
        Z1 .+= b1
        @. A1 = max(0f0, Z1) # In-place ReLU

        mul!(Z2, W2, A1)
        Z2 .+= b2
        @. A2 = 1f0 / (1f0 + exp(-Z2)) # In-place Sigmoid

        # --- Backward Pass ---
        @. dA2 = (A2 - y) / Float32(n_samples)

        # Update W2: W2 = W2 - lr * (dA2 * A1')
        # mul!(C, A, B, alpha, beta) -> C = A*B*alpha + C*beta
        mul!(W2, dA2, A1', -lr, 1f0)
        b2 .-= lr .* sum(dA2, dims=2)

        # dA1 = (W2' * dA2) .* (Z1 > 0)
        mul!(dA1, W2', dA2)
        @. dA1 *= (Z1 > 0f0)

        # Update W1: W1 = W1 - lr * (dA1 * X')
        # This was the dimension error: dA1 is (4,1000), X is (2,1000)
        # dA1 * X' results in (4,2)
        mul!(W1, dA1, X', -lr, 1f0)
        b1 .-= lr .* sum(dA1, dims=2)
    end
end

# Warmup to compile
train_mlp_fast(10)

# Execution
epochs = 1_000_000
start_time = time_ns()
train_mlp_fast(epochs)
elapsed = (time_ns() - start_time) / 1e9
println("Optimized Julia Training Time: ", round(elapsed, digits=4), "s")

