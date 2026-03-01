using Random
using Distributions
using LinearAlgebra
using Plots

function naive_bayes_comparison()
    # ----------------------------
    # 1. Data Generation
    # ----------------------------
    Random.seed!(0)

    # Mean vectors and diagonal covariance matrices
    m1 = [0.0, 2.0]; S1 = [4.0 0.0; 0.0 1.0]
    m2 = [0.0, 0.0]; S2 = [4.0 0.0; 0.0 1.0]

    n_points_per_class = 5000

    dist1 = MvNormal(m1, S1)
    dist2 = MvNormal(m2, S2)

    # Generate data (2 x 10000)
    X = hcat(rand(dist1, n_points_per_class), rand(dist2, n_points_per_class))
    labels = vcat(ones(Int, n_points_per_class), 2 * ones(Int, n_points_per_class))

    p_total = size(X, 2)
    P1 = n_points_per_class / p_total
    P2 = P1

    # ----------------------------
    # 2. Full Bayes Classification
    # ----------------------------
    # Calculate multivariate PDF for each point
    pdf1 = [pdf(dist1, X[:, i]) for i in 1:p_total]
        pdf2 = [pdf(dist2, X[:, i]) for i in 1:p_total]

            class_full = [P1 * pdf1[i] > P2 * pdf2[i] ? 1 : 2 for i in 1:p_total]
                Pe_full = mean(class_full .!= labels)

                # ----------------------------
                # 3. Naive Bayes Classification
                # ----------------------------
                # Naive assumption: f(x1, x2) = f(x1) * f(x2)
                # Define marginal distributions (univariate)
                d1_x1, d1_x2 = Normal(m1[1], sqrt(S1[1,1])), Normal(m1[2], sqrt(S1[2,2]))
                d2_x1, d2_x2 = Normal(m2[1], sqrt(S2[1,1])), Normal(m2[2], sqrt(S2[2,2]))

                pdf1_naive = [(pdf(d1_x1, X[1, i]) * pdf(d1_x2, X[2, i])) for i in 1:p_total]
                    pdf2_naive = [(pdf(d2_x1, X[1, i]) * pdf(d2_x2, X[2, i])) for i in 1:p_total]

                        class_naive = [P1 * pdf1_naive[i] > P2 * pdf2_naive[i] ? 1 : 2 for i in 1:p_total]
                            Pe_naive = mean(class_naive .!= labels)

                            # ----------------------------
                            # 4. Results and Plotting
                            # ----------------------------
                            println("Full Bayes Error:  ", round(Pe_full, digits=4))
                            println("Naive Bayes Error: ", round(Pe_naive, digits=4))

                            # Plot results
                            p1 = scatter(X[1, labels.==1], X[2, labels.==1], color=:blue, alpha=0.1, label="Class 1")
                            scatter!(X[1, labels.==2], X[2, labels.==2], color=:red, alpha=0.1, label="Class 2")
                            title!("True Labels and Data Distribution")

                            p2 = scatter(X[1, class_naive.==1], X[2, class_naive.==1], color=:blue, alpha=0.1, label="Pred Class 1")
                            scatter!(X[1, class_naive.==2], X[2, class_naive.==2], color=:red, alpha=0.1, label="Pred Class 2")
                            title!("Naive Bayes Classification Results")

                            display(plot(p1, p2, layout=(1,2), size=(900, 400)))
                        end

                        naive_bayes_comparison()


