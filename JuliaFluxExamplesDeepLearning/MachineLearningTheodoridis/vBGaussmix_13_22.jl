using LinearAlgebra, Random, Distributions, Plots, SpecialFunctions

# ----------------------------
# 1. Generate data (top-level)
# ----------------------------
Random.seed!(42)
N = 300
μ_true = [-2.5 -4.0 2.0 0.1 3.0;
          2.5 -2.0 -1.0 0.2 3.0]
Σ_true = [
    [0.5 0.081; 0.081 0.7],
    [0.4 0.02; 0.02 0.3],
    [0.6 0.531; 0.531 0.9],
    [0.5 0.22; 0.22 0.8],
    [0.88 0.2; 0.2 0.22]
    ]
X = hcat([rand(MvNormal(μ_true[:, i], Σ_true[i]), N ÷ 5) for i in 1:5]...)
    l, N_total = size(X)
    K = 25

    # ----------------------------
    # 2. Define VB-GMM function
    # ----------------------------
    function run_variational_gmm(X, K; iterations=100)
        l, N = size(X)
        beta = 1.0
        nu_prior = ones(K)
        W0_inv = Matrix{Float64}(I, l, l)

        EQk = [Matrix{Float64}(I, l, l) for _ in 1:K]
            Pk = fill(1/K, K)
            ElnQk = ones(K)
            μ_tilde = randn(l, K)

            for i in 1:iterations
                # E-Step
                pik = zeros(K, N)
                for k in 1:K
                    for n in 1:N
                        diff = X[:, n] - μ_tilde[:, k]
                        quad_form = dot(diff, EQk[k] * diff)
                        pik[k, n] = Pk[k] * exp(0.5 * ElnQk[k] - 0.5 * quad_form)
                    end
                end
                pnk = pik ./ sum(pik, dims=1)
                pnk_sumk = vec(sum(pnk, dims=2))

                # M-Step
                for k in 1:K
                    if pnk_sumk[k] > 1e-5
                        Q_tilde = beta * I + EQk[k] * pnk_sumk[k]
                        Q_inv = inv(Q_tilde)
                        μ_tilde[:, k] = Q_inv * EQk[k] * (X * pnk[k, :]')

                        nuk = nu_prior[k] + pnk_sumk[k]
                        Wk_inv = copy(W0_inv)
                        for n in 1:N
                            d = X[:, n] - μ_tilde[:, k]
                            Wk_inv += pnk[k, n] * (d * d' + Q_inv)
                        end
                        Wk = inv(Wk_inv)
                        EQk[k] = nuk * Wk
                        ElnQk[k] = sum(digamma.(0.5 .* (nuk .- (0:l-1)))) + l*log(2) + logdet(Wk)[1]
                        Pk[k] = pnk_sumk[k] / N
                    else
                        Pk[k] = 0.0
                    end
                end
            end

            return μ_tilde, EQk, Pk
        end

        # ----------------------------
        # 3. Run VB-GMM at top level
        # ----------------------------
        μ_final, EQ_final, P_final = run_variational_gmm(X, K)  # <- Must be top-level

        # ----------------------------
        # 4. Plotting
        # ----------------------------
        plt = scatter(X[1, :], X[2, :], marker=:dot, color=:black, alpha=0.3, label="Data", aspect_ratio=:equal)
        display(plt)


