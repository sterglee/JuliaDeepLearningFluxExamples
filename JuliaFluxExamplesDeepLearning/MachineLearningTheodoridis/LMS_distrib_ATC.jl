using LinearAlgebra
using Statistics
using Random
using Plots

# -----------------------------
# 1. Distributed LMS Function
# -----------------------------
function LMS_distrib_sparse(inputvec, y, data, errfun)

    L = data[:L]
    N = data[:N]
    nodes = data[:nodes]
    mu = data[:mu]
    normalized = get(data, :normalized, false)
    gamma = data[:gamma]
    C = data[:C]

    # Make mu a function if scalar
    if !(typeof(mu) <: Function)
        μval = mu
        mu = i -> μval
    end

    # Initialize node estimates
    x = [zeros(L) for _ in 1:nodes]
        if haskey(data, :initial_estimate)
            x = [copy(data[:initial_estimate]) for _ in 1:nodes]
            end

            # Initialize error storage
            err = [zeros(N) for _ in 1:nodes]

                # Sparse regularization
                epsil = 0.1
                gradf(xv) = (1 ./ (epsil .+ abs.(xv))) .* sign.(xv)
                f(xv) = sum(abs.(xv) ./ (epsil .+ abs.(xv)))

                gradfmat = zeros(nodes)
                fmat = zeros(nodes)

                for i in 1:N
                    if gamma == 0
                        for n in 1:nodes
                            gradfmat[n] = norm(gradf(x[n]))^2
                            fmat[n] = f(x[n])
                        end
                    end

                    # Adaptation stage
                    x_c = similar(x)
                    for n in 1:nodes
                        a_n = inputvec(n, i)
                        er = y[n][i] - dot(a_n, x[n])
                        if normalized
                            x_c[n] = x[n] + mu(i)/norm(a_n)^2 * a_n * er
                        else
                            x_c[n] = x[n] + mu(i) * a_n * er
                        end
                    end

                    # Combination stage
                    for n in 1:nodes
                        neighb = findall(c -> c != 0, C[:, n])
                        x[n] = zeros(L)
                        for ii in neighb
                            x[n] .+= C[ii, n] * x_c[ii]
                        end
                    end

                    # Error computation
                    for n in 1:nodes
                        if length(methods(errfun).ms) == 2
                            err[n][i] = errfun(i, x[n])
                        else
                            err[n][i] = errfun(x[n])
                        end
                    end
                end

                return err, x
            end

            # -----------------------------
            # 2. Example Call
            # -----------------------------

            # Simulation parameters
            L = 5
            N = 1000
            nodes = 3
            mu_val = 0.01
            gamma = 0.0

            # Combination matrix (simple averaging)
            C = ones(nodes, nodes) ./ nodes

            # Input and desired signal
            Xnodes = [randn(L, N) for n in 1:nodes]      # white input
                θ_true = randn(L)                             # unknown system
                y = [Xnodes[n]' * θ_true .+ 0.01*randn(N) for n in 1:nodes]

                    # Data dictionary
                    data = Dict(
                        :L => L,
                        :N => N,
                        :mu => mu_val,
                        :nodes => nodes,
                        :gamma => gamma,
                        :C => C,
                        :normalized => true,
                        :initial_estimate => zeros(L)
                        )

                    # Error function
                    errfun = x -> norm(x - θ_true)^2

                    # Input vector function
                    inputvec = (node, i) -> Xnodes[node][:, i]

                    # -----------------------------
                    # 3. Call the function
                    # -----------------------------
                    err, x_est = LMS_distrib_sparse(inputvec, y, data, errfun)

                    # -----------------------------
                    # 4. Plot learning curves
                    # -----------------------------
                    plot()
                    for n in 1:nodes
                        plot!(10 .* log10.(err[n]), label="Node $n", xlabel="Iteration", ylabel="MSE (dB)", title="Distributed LMS Learning Curves")
                    end
                    display(plot!)


