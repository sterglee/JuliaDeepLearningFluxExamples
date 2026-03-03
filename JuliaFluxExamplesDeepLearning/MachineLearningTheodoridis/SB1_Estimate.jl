using LinearAlgebra
using Printf
using Random

# --- Helper Functions ---

function sb1_diagnostic(level, fmt, args...)
  @printf(fmt, args...)
end

function sb1_posterior_mode(PHI, t, w_init, alpha, max_its)
  # Placeholder for Bernoulli logic (Classification)
  M = size(PHI, 2)
  Hessian = (PHI' * PHI) + Diagonal(alpha)
  U = cholesky(Hermitian(Hessian)).U
  Ui = inv(U)
  w = (Ui * (Ui' * (PHI' * t)))
  # Placeholder data likelihood for Bernoulli
  data_likely = -0.5 * sum((t - PHI*w).^2)
  return w, Ui, data_likely
end

# --- Main Algorithm ---

"""
sb1_estimate(PHI, t, alpha_init, beta_init, max_its, mon_its=0)

Translated Sparse Bayesian Estimate (V1.10).
"""
function sb1_estimate(PHI, t, alpha_val, beta_val, max_its, mon_its=0)
  MIN_DELTA_LOGALPHA = 1e-3
  ALPHA_MAX = 1e9

  regression = (beta_val != 0)
  fixed_beta = false
  beta = Float64(beta_val)
  max_its_pm = 25

  if regression
    fixed_beta = beta_val < 0
    beta = abs(beta)
  end

  n_samples, m_total = size(PHI)
  w = zeros(m_total)
  alpha = fill(Float64(alpha_val), m_total)
  gamma = ones(m_total)

  phit_t = PHI' * t
  last_it = false

  # Initialize variables for return scope
  useful = trues(m_total)
  marginal = -Inf
  alpha_used = Float64[]

  for i in 1:max_its
    # 1. Pruning
    useful = (alpha .< ALPHA_MAX)
    m_curr = sum(useful)

    # If no basis functions remain, break to avoid errors
    if m_curr == 0
      @warn "All basis functions pruned."
      break
    end

    alpha_used = alpha[useful]
    phi_used = PHI[:, useful]
    w[.!useful] .= 0

    data_likely = 0.0
    ui = zeros(m_curr, m_curr)

    # 2. Compute Weights and Stats
    if regression
      # Hessian = (Phi' * Phi) * beta + diag(alpha)
      # We use Hermitian to ensure the Cholesky solver recognizes symmetry
      hessian = (phi_used' * phi_used) .* beta + Diagonal(alpha_used)
      u_fact = cholesky(Hermitian(hessian)).U
      ui = inv(u_fact) # Inverse of Cholesky factor

      w_useful = (ui * (ui' * phit_t[useful])) * beta
      w[useful] .= w_useful

      ed = sum((t .- phi_used * w_useful).^2)
      data_likely = (n_samples * log(beta) - beta * ed) / 2
    else
      w_useful, ui, data_likely = sb1_posterior_mode(phi_used, t, w[useful], alpha_used, max_its_pm)
      w[useful] .= w_useful
      ed = 0.0 # Not used in classification re-estimation here
    end

    # 3. Covariance & Well-determinedness
    # In MATLAB: logdetH = -2*sum(log(diag(Ui)))
    # This is the log-determinant of the Hessian
    log_det_h = -2.0 * sum(log.(diag(ui)))

    # diag(Sigma) = sum(Ui.^2, 2)
    diag_sig = sum(ui.^2, dims=2)[:]
    gamma_used = 1.0 .- alpha_used .* diag_sig
    gamma[useful] .= gamma_used

    # 4. Marginal Likelihood
    # L = DataLikely - 0.5 * (log|H| - log|A| + w'*A*w)
    marginal = data_likely - 0.5 * (log_det_h - sum(log.(alpha_used)) + (w[useful]' * (alpha_used .* w[useful])))

    # 5. Diagnostics
    if last_it || (mon_its > 0 && i % mon_its == 0)
      if regression
        sb1_diagnostic(1, "%5d> L = %.3f\t Gamma = %.2f (nz = %d)\t s=%.3f\n",
                       i, marginal, sum(gamma_used), m_curr, sqrt(1/beta))
      else
        sb1_diagnostic(1, "%5d> L = %.3f\t Gamma = %.2f (nz = %d)\n",
                       i, marginal, sum(gamma_used), m_curr)
      end
    end

    if !last_it
      # 6. Hyperparameter Re-estimation
      log_alpha_old = log.(alpha[useful])

      # MacKay update: alpha = gamma / w^2
      # Add eps() to avoid division by zero
      alpha[useful] .= gamma_used ./ (w[useful].^2 .+ eps())

      # Check convergence
      au = alpha[useful]
      valid_idx = au .> 0
      max_d_alpha = maximum(abs.(log_alpha_old[valid_idx] .- log.(au[valid_idx])))

      if max_d_alpha < MIN_DELTA_LOGALPHA
        last_it = true
        sb1_diagnostic(1, "Terminating: max log(alpha) change is %g (<%g).\n",
                       max_d_alpha, MIN_DELTA_LOGALPHA)
      end

      if regression && !fixed_beta
        # Note: ed must be current
        ed = sum((t .- phi_used * w[useful]).^2)
        beta = (n_samples - sum(gamma_used)) / (ed + eps())
      end
    else
      break
    end
  end

  if !last_it
    sb1_diagnostic(1, "Terminating due to max iterations (did not converge)\n")
  end

  # Tidy up results
  final_useful = (alpha .< ALPHA_MAX)
  final_weights = w[final_useful]
  used_indices = findall(final_useful)

  sb1_diagnostic(1, "Hyperparameter estimation complete\n")
  sb1_diagnostic(2, "non-zero parameters:\t%d\n", length(final_weights))

  return final_weights, used_indices, marginal, alpha[final_useful], beta, gamma[final_useful]
end

# --- Main Demo ---

function main()
  Random.seed!(42)
  println("--- Running SB1_Estimate Demo ---")

  N, M = 150, 100
  X = randn(N, M)
  true_w = zeros(M)
  true_w[[10, 20, 30]] = [10.0, -5.0, 2.0]

  # Sigma = 0.1 => Beta = 100
  t = X * true_w + randn(N) * 0.1

  weights, used, ml, alpha, beta, gamma = sb1_estimate(X, t, 1e-3, 1.0, 500, 50)

  println("\n--- Results ---")
  println("Indices found: ", used)
  println("Weights: ", round.(weights, digits=3))
  @printf("Estimated Beta: %.2f (True: 100.0)\n", beta)
end

main()

