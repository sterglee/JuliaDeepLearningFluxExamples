using Flux
using Statistics
using Plots
using LinearAlgebra
using Random

Random.seed!(42)

# -------------------------------------------------
# 1. Setup Data
# -------------------------------------------------
true_func(x) = sin.(3f0 .* x) .+ 0.5f0 .* x

x_train = reshape(Float32.(collect(-2:0.5:2)), 1, :)
y_train = vec(true_func(x_train) .+ 0.1f0 .* randn(Float32, size(x_train)))

# -------------------------------------------------
# 2. Random Features (Fixed hidden layer)
# -------------------------------------------------
n_hidden = 50
W_random = randn(Float32, n_hidden, 1)
b_random = randn(Float32, n_hidden)

function get_features(x)
    relu.(W_random * x .+ b_random)
end

# -------------------------------------------------
# 3. Bayesian Linear Regression on Random Features
# -------------------------------------------------
function bayesian_inference(x_test, x_train, y_train, σ_noise, σ_prior)

    H_train = get_features(x_train)        # (H × N)
    H_test  = get_features(x_test)         # (H × 1)

    # Precision matrix
    A = (1f0/σ_noise^2) .* (H_train * H_train') .+
        (1f0/σ_prior^2) .* I(n_hidden)

    Σ = inv(A)

    # Posterior mean of weights
    w_hat = (1f0/σ_noise^2) .* Σ * H_train * y_train

    # Predictive mean
    μ_star = only(w_hat' * H_test)

    # Predictive variance
    σ_star_sq = σ_noise^2 + only(H_test' * Σ * H_test)

    return Float32(μ_star), Float32(sqrt(σ_star_sq))
end

# -------------------------------------------------
# 4. Generate Predictions
# -------------------------------------------------
x_range = reshape(Float32.(collect(-3:0.1:3)), 1, :)

means = Float32[]
stds  = Float32[]

for i in 1:size(x_range, 2)
    m, s = bayesian_inference(x_range[:, i:i], x_train, y_train, 0.2f0, 1.0f0)
    push!(means, m)
    push!(stds, s)
end

# -------------------------------------------------
# 5. Visualization
# -------------------------------------------------
plot(vec(x_range), means,
     ribbon = 2 .* stds,
     fillalpha = 0.2,
     label = "Uncertainty (2σ)",
     lw = 2)

scatter!(vec(x_train), y_train,
         label = "Training Data")

plot!(vec(x_range),
      vec(true_func(x_range)),
      label = "True Function",
      ls = :dash)

title!("Bayesian Neural Network: Predictive Uncertainty")

