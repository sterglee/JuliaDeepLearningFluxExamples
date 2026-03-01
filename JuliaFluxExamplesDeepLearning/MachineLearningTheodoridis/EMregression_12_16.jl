using LinearAlgebra
using Random
using Plots

function em_linear_regression()
    Random.seed!(0)  # Match MATLAB rng('default')

    # --- True signal ---
    x = 0:1e-4:2
    y = 0.2 .- x .+ 0.9 .* x.^2 .+ 0.7 .* x.^3 .- 0.2 .* x.^5

    # --- Training samples ---
    N = 500
    K = 5
    a, b = 0.0, 2.0

    x1 = range(a, stop=b - (b/N), length=N)
    sigma_eta = 0.05
    n = sqrt(sigma_eta) .* randn(N)

    theta_true = [0.2; -1.0; 0.9; 0.7; -0.2]

    # Measurement matrix
    Phi = hcat(ones(N), x1, x1.^2, x1.^3, x1.^5)
    Phi_gram = Phi' * Phi

    # Noisy observations
    y1 = Phi * theta_true + n

    # --- EM algorithm ---
    EMiter = 20
    betaj = 1.0
    alphaj = 1.0
    sigma_eta_EM = zeros(EMiter)
    Phiy = Phi' * y1

    mu_theta = zeros(K)

    for i in 1:EMiter
        Sigma_theta = inv(betaj * Phi_gram + alphaj * I(K))
        mu_theta = betaj * Sigma_theta * Phiy

        alphaj = K / (norm(mu_theta)^2 + tr(Sigma_theta))
        betaj = N / (norm(y1 - Phi * mu_theta)^2 + tr(Sigma_theta))

        sigma_eta_EM[i] = 1 / betaj
    end

    # --- Prediction ---
    Np = 10
    x2 = (b-a) .* rand(Np)
    Phip = hcat(ones(Np), x2, x2.^2, x2.^3, x2.^5)

    # Predicted mean
    y_pred = Phip * mu_theta

    # Predicted variance
    temp = inv(sigma_eta_EM[end] * I(K) + (1/alphaj) * Phi_gram)
    y_pred_var = sigma_eta_EM[end] .+ sigma_eta_EM[end] .* diag(Phip * (1/alphaj * temp) * Phip')

    # --- Plot true curve and predictions with error bars ---
    p1 = plot(x, y, color=:black, label="True Curve", lw=2)
    scatter!(x2, y_pred, color=:blue, label="Predictions")
    for i in 1:Np
        plot!([x2[i], x2[i]], [y_pred[i]-y_pred_var[i], y_pred[i]+y_pred_var[i]], color=:red, lw=1)
    end
    xlabel!("x"); ylabel!("y"); title!("EM Linear Regression Predictions with Error Bars")

    # --- Plot Noise Variance over Iterations ---
    p2 = plot(1:EMiter, sigma_eta_EM, color=:blue, lw=2, label="Estimated Noise Variance")
    hline!(p2, [sigma_eta], color=:red, linestyle=:dash, label="True Noise Variance")
    xlabel!("EM Iteration"); ylabel!("Noise Variance"); title!("Noise Variance Convergence")

    display(p1)
    display(p2)
end

em_linear_regression()


