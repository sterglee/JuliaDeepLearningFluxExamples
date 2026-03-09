using LinearAlgebra, Random, Base.Threads

function train_mlp_multithreaded(epochs)
    n_samples = 1000
    X = randn(Float32, 2, n_samples)
    y = reshape(Float32.( (X[1,:] .^ 2 + X[2,:] .^ 2) .> 1.0f0 ), 1, n_samples)

    in_d, hid_d, out_d = 2, 4, 1
    lr = 0.1f0

    W1 = randn(Float32, hid_d, in_d) .* sqrt(2f0/in_d)
    b1 = zeros(Float32, hid_d, 1)
    W2 = randn(Float32, out_d, hid_d) .* sqrt(1f0/hid_d)
    b2 = zeros(Float32, out_d, 1)

    # Buffers
    Z1 = zeros(Float32, hid_d, n_samples)
    A1 = zeros(Float32, hid_d, n_samples)
    Z2 = zeros(Float32, out_d, n_samples)
    A2 = zeros(Float32, out_d, n_samples)
    dA2 = zeros(Float32, out_d, n_samples)
    dA1 = zeros(Float32, hid_d, n_samples)

    # Force BLAS to use a specific number of threads for matrix mult
    # For small matrices like this, 1 thread is often faster to avoid overhead
    BLAS.set_num_threads(1)

    for i in 1:epochs
        # --- Forward ---
        mul!(Z1, W1, X)
        # Use @threads for large element-wise sweeps if n_samples was larger
        # but for 1000 samples, SIMD via @fastmath is better
        @fastmath @inbounds @. Z1 += b1
        @fastmath @inbounds @. A1 = max(0f0, Z1)

        mul!(Z2, W2, A1)
        @fastmath @inbounds @. Z2 += b2
        @fastmath @inbounds @. A2 = 1f0 / (1f0 + exp(-Z2))

        # --- Backward ---
        @fastmath @inbounds @. dA2 = (A2 - y) / Float32(n_samples)

        mul!(W2, dA2, A1', -lr, 1f0)
        # Manual sum reduction can be faster than sum(dims=2)
        b2 .-= lr .* sum(dA2, dims=2)

        mul!(dA1, W2', dA2)
        @fastmath @inbounds @. dA1 *= (Z1 > 0f0)

        mul!(W1, dA1, X', -lr, 1f0)
        b1 .-= lr .* sum(dA1, dims=2)
    end
end

# Warmup
train_mlp_multithreaded(10)

epochs = 1_000_000
start_time = time_ns()
train_mlp_multithreaded(epochs)
elapsed = (time_ns() - start_time) / 1e9
println("Julia Multithreaded Time: ", round(elapsed, digits=4), "s")

