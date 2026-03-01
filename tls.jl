using LinearAlgebra

"""
tls(A, b, thresh=eps(Float64))

Solves the linear equation Ax ≈ b using the Truncated Total Least Squares algorithm.

# Arguments
- `A`: The m x n data matrix.
- `b`: The m x 1 observation vector.
- `thresh`: Threshold for singular value truncation (default is machine epsilon).
    """
    function tls(A::AbstractMatrix, b::AbstractVector, thresh::Float64=eps(Float64))
        m, n = size(A)

        # 1. Size Validation
        if length(b) != m
            throw(DimensionMismatch("A and b size mis-match: A has $m rows, but b has $(length(b)) elements."))
        end

        # 2. Construct Augmented Matrix [A | b]
        # In Julia, hcat is more idiomatic for augmenting columns
        Z = hcat(A, b)

        # 3. Compute SVD of the augmented matrix
        # SVD returns U, S, and V. Note: Julia's V is already V, not V'.
        U, S, V = svd(Z)

        # 4. Truncation Logic
        # Identify the number of singular values below the threshold
        k = sum(S .< thresh)

        # Define the index for the noise subspace
        # q represents the start of the columns in V associated with the smallest singular values
        q = (n + 1) - k + 1
        if q > n + 1
            q = n + 1 # Fallback to the last column if none are below thresh
        end

        # 5. Extract Sub-matrices of V
        # V is (n+1) x (n+1). V12 is n x (k), V22 is 1 x (k)
        V12 = V[1:n, q:end]
        V22 = V[n+1, q:end]

        # 6. Compute TLS solution
        # The formula x = -V12 * V22' / ||V22||²
        # Note: V22 is treated as a row vector here.
        x = -V12 * V22' ./ norm(V22)^2

        return x[:] # Return as a flat vector
    end

    # --- Example Usage ---
    # Generate a simple system with noise in both A and b
    A_true = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    x_true = [1.0, -1.0]
    b_true = A_true * x_true

    # Add noise to both
    A_noisy = A_true + 0.05 * randn(3, 2)
    b_noisy = b_true + 0.05 * randn(3)

    x_tls = tls(A_noisy, b_noisy)
    println("TLS Solution: ", x_tls)
    println("True Solution: ", x_true)


