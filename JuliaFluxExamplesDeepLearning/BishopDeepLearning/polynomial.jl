using Random, LinearAlgebra
using Plots
using MLJLinearModels

# --- Δημιουργία συνθετικών δεδομένων ---
Random.seed!(42)
N = 10
x = range(0, 1, length=N)
t = sin.(2π .* x) .+ 0.1 .* randn(N)

# --- Πολυωνυμική βάση ---
degree = 9
# Use collect() to ensure we have a Matrix, and power broadcasting
X_poly = hcat([x.^i for i in 0:degree]...)

# --- Πολυωνυμικό fit χωρίς regularization ---
# Standard OLS via backslash
w = X_poly \ t
x_fit = range(0, 1, length=100)
X_fit_poly = hcat([x_fit.^i for i in 0:degree]...)
y_fit = X_fit_poly * w

# --- Ridge Regression με MLJLinearModels ---
# Note: MLJLinearModels usually expects an intercept unless fit_intercept=false.
# Since our X_poly already includes a column of 1s (x^0), we set fit_intercept=false.
λ = 0.001  # Lower lambda for a more visible fit
ridge_model = RidgeRegression(λ, fit_intercept=false)

# MLJLinearModels.fit takes the model, the matrix X, and the target t
# It returns the learned coefficients (θ)
θ = fit(ridge_model, X_poly, t)

# For prediction, we manually multiply or use the predict function if available
y_ridge = X_fit_poly * θ

# --- Οπτικοποίηση ---
scatter(x, t, color=:blue, label="Training Data", markersize=5)
plot!(x_fit, sin.(2π .* x_fit), color=:black, linestyle=:dash, label="True Function")
plot!(x_fit, y_fit, color=:red, label="OLS (Degree 9 - Overfit)", ylims=(-1.5, 1.5))
plot!(x_fit, y_ridge, color=:green, label="Ridge (Regularized)", linewidth=2)
xlabel!("x")
ylabel!("t")
title!("Polynomial Fit vs Regularized Fit")

