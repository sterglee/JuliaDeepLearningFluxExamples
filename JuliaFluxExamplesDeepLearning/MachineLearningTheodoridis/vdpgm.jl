using LinearAlgebra, Statistics, Random, SpecialFunctions, Distributions

# --- Options Handling ---
# We use a mutable struct to mimic MATLAB's dynamic fields with default values
Base.@kwdef mutable struct VDPGMOptions
  algorithm::String = "vdp"
  collapsed_means::Bool = false
  do_sort::Bool = false
  get_q_of_z::Bool = false
  weight::Float64 = 1.0
  get_log_likelihood::Bool = false
  use_kd_tree::Bool = true
  threshold::Float64 = 1e-5
  sis::Int = 0
  initial_depth::Int = 3
  initial_K::Int = 1
  ite::Float64 = Inf
  do_split::Bool = false
  do_merge::Bool = false
  do_greedy::Bool = true
  max_target_ratio::Float64 = 0.5
  init_of_split::String = "pc"
  recursive_expanding_depth::Int = 2
  recursive_expanding_threshold::Float64 = 0.1
  recursive_expanding_frequency::Int = 3
  test_data::Union{Nothing, Matrix{Float64}} = nothing
  hp_posterior::Any = nothing  # Placeholder for complex posterior struct
end

# --- The Main Function ---
function vdpgm(given_data::Matrix{Float64}, opts::VDPGMOptions=VDPGMOptions())
  start_time = time()

  # In Julia, dense matrices are standard; if given_data is sparse, convert it
  # data = issparse(given_data) ? collect(given_data) : given_data
  D, N = size(given_data)

  # 1. Prediction Mode (if hp_posterior is provided)
  if opts.hp_posterior !== nothing
    results = Dict()
    if opts.get_q_of_z
      results[:q_of_z] = mk_q_of_z(given_data, opts.hp_posterior, opts)
    end
    if opts.test_data !== nothing
      results[:predictive_posterior] = log_predictive_dist(given_data, opts.hp_posterior, opts)
    end
    return results
  end

  # 2. Initialization
  # hp_prior = mk_hp_prior(given_data, opts)

  if opts.sis > 0
    # q_of_z = sequential_importance_sampling(given_data, hp_prior, opts)
  else
    # q_of_z = rand_q_of_z(given_data, opts.initial_K, opts)
  end

  # 3. Optimization Loop (Greedy vs Split-Merge)
  # This is where the core VB inference occurs
  # ... logic for greedy or split_merge ...

  # 4. Result Construction
  results = (
    algorithm = opts.algorithm,
    elapsed_time = time() - start_time,
    # hp_posterior = hp_posterior,
    # K = length(hp_posterior.eta),
    opts = opts
    )

  return results
end

# --- Log Predictive Distribution ---
function log_predictive_dist(data, hp_posterior, opts)
  # This translates the log_predictive_dist and log_T_dist logic
  test_data = opts.test_data
  D, n_test = size(test_data)
  K = length(hp_posterior.m_cols) # assuming m is stored as columns

  log_prob = fill(-Inf, K, n_test)
  E_pi = mk_E_pi(hp_posterior, opts)

  for c in 1:K
    if E_pi[c] > 0
      f = hp_posterior.eta[c] + 1 - D
      # Σ = B * (ξ+1) / (ξ * f)
      sigma = hp_posterior.B[c] * (hp_posterior.xi[c] + 1) / (hp_posterior.xi[c] * f)

      log_prob[c, :] .= log(E_pi[c]) .+ log_T_dist(test_data, hp_posterior.m[:, c], sigma, f)
    end
  end

  return log_sum_exp(log_prob, dims=1)
end

# --- Log Multivariate Student-T Density ---
function log_T_dist(X, m, sigma, f)
  D, n = size(X)
  diff = X .- m

  # term1: Γ((f+D)/2) / ( (fπ)^(D/2) * Γ(f/2) * sqrt(|Σ|) )
  log_const = lgamma((f + D) / 2) - (D / 2) * log(f * π) - lgamma(f / 2) - 0.5 * logdet(sigma)

  # Quadratic part: (1 + (1/f)(x-m)' Σ⁻¹ (x-m)) ^ (-(f+D)/2)
  # Efficient calculation: sum(diff .* (inv(sigma) * diff), dims=1)
  quad = sum(diff .* (sigma \ diff), dims=1)
  log_kernel = -(f + D) / 2 .* log.(1 .+ (1/f) .* quad)

  return log_const .+ log_kernel
end

# --- Helper: LogSum


