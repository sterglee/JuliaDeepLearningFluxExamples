using LinearAlgebra

"""
sb1_rvm(X, t, init_alpha, init_beta, kernel, length_scale, use_bias, max_its, mon_its)

Julia implementation of the RVM wrapper.
Note: This requires `sb1_kernel_function` and `sb1_estimate` to be defined.
"""
function sb1_rvm(X, t, init_alpha, init_beta, kernel_, length_scale, use_bias, max_its, mon_its)

  println("Constructing RVM ...")
  println("Evaluating kernel ...")

  # 1. Generate the Basis Matrix (Design Matrix)
  # PHI is N x N where PHI[i, j] = K(x_i, x_j)
  phi = sb1_kernel_function(X, X, kernel_, length_scale)
  n_samples, d = size(X)

  # 2. Add Bias (Intercept) term
  # We append a column of ones to the end of the matrix
  if use_bias
    phi = hcat(phi, ones(n_samples, 1))
  end

  println("Created basis matrix PHI: $(size(phi, 1)) x $(size(phi, 2))")
  println("Kernel: '$kernel_' | Scale: $length_scale")

  # 3. Call the Sparse Bayesian Estimation routine
  # This is where the heavy lifting (Evidence Procedure / Fast Marginal Likelihood) happens
  println("Calling hyperparameter estimation routine ...")
  weights, used, marginal, alpha, beta, gamma = sb1_estimate(
    phi, t, init_alpha, init_beta, max_its, mon_its
    )

  # 4. Handle the Bias term for the output
  # In Julia, 'used' contains the indices of the rows of X that are RVs.
  # If the bias was 'used', it will be index N + 1.
  bias = 0.0
  if use_bias
    # Find if the last column index (N+1) is in the 'used' list
    bias_idx = findfirst(==(n_samples + 1), used)

    if bias_idx !== nothing
      bias = weights[bias_idx]

      # Remove the bias from the weights and used-indices vectors
      # so they only reflect data-point relevance
      deleteat!(weights, bias_idx)
      deleteat!(used, bias_idx)
    end
  end

  return weights, used, bias, marginal, alpha, beta, gamma
end

