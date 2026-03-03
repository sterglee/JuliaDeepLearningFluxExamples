using SpecialFunctions

"""
gamma_multivariate_ln(x, p)

Compute the natural logarithm of the multivariate gamma function Γₚ(x).
    - `x`: A number or an array of numbers.
    - `p`: The dimension (scalar).

    Formula:
    log Γₚ(x) = p(p-1)/4 * log(π) + Σⱼ₌₁ᵖ log Γ(x + (1-j)/2)
    """
    function gamma_multivariate_ln(x, p::Int)
        # Ensure x is an array for consistency, even if a scalar is passed
        x_vec = collect(x)
        K = length(x_vec)

        # Precompute the constant term
        constant_term = p * (p - 1) * 0.25 * log(pi)

        # Calculate the sum of log-gammas
        # j ranges from 1 to p
        # For each x in x_vec, we sum loggamma(x + 0.5*(1-j))
        val = zeros(Float64, K)

        for k in 1:K
            sum_logs = 0.0
            for j in 1:p
                # logabsgamma returns (result, sign); we take the first element [1]
                sum_logs += logabsgamma(x_vec[k] + 0.5 * (1 - j))[1]
            end
            val[k] = constant_term + sum_logs
        end

        # If input was a single value, return a scalar instead of a 1-element array
        return length(val) == 1 ? val[1] : val
    end

    # --- Demo / Testing ---
    println("Scalar test (x=5, p=3): ", gamma_multivariate_ln(5, 3))
    println("Array test (x=[5, 10], p=2): ", gamma_multivariate_ln([5, 10], 2))

