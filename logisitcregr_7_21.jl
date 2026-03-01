using Random
using Distributions
using LinearAlgebra
using Plots

function logistic_regression_example()
    # ----------------------------
    # 1. Data Generation
    # ----------------------------
    Random.seed!(0)

    # Mean vectors and covariance matrices
    m1 = [0.0, 2.0];  S1 = [4.0 1.8; 1.8 1.0]
    m2 = [0.0, 0.0];  S2 = [4.0 -1.8; -1.8 1.0]

    n_points = 1500

    dist1 = MvNormal(m1, S1)
    dist2 = MvNormal(m2, S2)

    # Training set (2 x 3000)
    X = hcat(rand(dist1, n_points), rand(dist2, n_points))
    y = vcat(zeros(Int, n_points), ones(Int, n_points))      # Labels: 0 or 1

    # Test set
    X_test = hcat(rand(dist1, n_points), rand(dist2, n_points))
    y_test = vcat(zeros(Int, n_points), ones(Int, n_points))

    # ----------------------------
    # 2. Bayes Classification (The Ideal Benchmark)
    # ----------------------------
    p1 = [pdf(dist1, X_test[:, i]) for i in 1:size(X_test, 2)]
        p2 = [pdf(dist2, X_test[:, i]) for i in 1:size(X_test, 2)]

            # Prior probabilities (assuming equal classes)
            P1, P2 = 0.5, 0.5

            class_test_bayes = [P1*p1[i] > P2*p2[i] ? 0 : 1 for i in 1:length(p1)]
                Pe_bayes = mean(class_test_bayes .!= y_test)
                println("Bayes Classification Error: ", Pe_bayes)

                # ----------------------------
                # 3. Logistic Regression via Gradient Descent
                # ----------------------------
                # Add bias row (Intercept) to make it 3 x 3000
                X_aug = vcat(X, ones(1, size(X, 2)))
                X_test_aug = vcat(X_test, ones(1, size(X_test, 2)))

                rho = 0.0001          # Reduced step size for stability
                theta = zeros(3)      # Initial weights [w1, w2, b]
                e_thresh = 1e-6
                max_iter = 10000
                e = 1.0
                iter = 0

                # Sigmoid function
                σ(z) = 1.0 ./ (1.0 .+ exp.(-z))

                while e > e_thresh && iter < max_iter
                    iter += 1
                    theta_old = copy(theta)

                    # 1. Calculate predictions (1 x 3000)
                    z = theta' * X_aug
                    s = σ(z)

                    # 2. Calculate gradient (3 x 1)
                    # Gradient = X * (predictions - labels)^T
                    grad = X_aug * (s .- y')'

                    # 3. Update
                    theta -= rho .* grad

                    # 4. Check convergence
                    e = norm(theta .- theta_old)
                end

                println("Logistic Regression Converged in $iter iterations.")
                println("Final Theta: ", round.(theta, digits=3))

                # ----------------------------
                # 4. Evaluate and Plot
                # ----------------------------
                s_test = σ(theta' * X_test_aug)
                predictions = [val > 0.5 ? 1 : 0 for val in s_test]

                    Pe_log = mean(predictions[:] .!= y_test)
                    println("Logistic Regression Test Error: ", Pe_log)

                    # Visualization
                    plt = scatter(X_test[1, y_test .== 0], X_test[2, y_test .== 0],
                                  color=:blue, label="Class 0", alpha=0.3)
                    scatter!(X_test[1, y_test .== 1], X_test[2, y_test .== 1],
                             color=:red, label="Class 1", alpha=0.3)

                    # Plot Decision Boundary: w1*x + w2*y + b = 0  => y = (-w1*x - b) / w2
                    x_range = range(minimum(X_test[1,:]), maximum(X_test[1,:]), length=100)
                    boundary_y = (-theta[1] .* x_range .- theta[3]) ./ theta[2]
                    plot!(x_range, boundary_y, color=:black, lw=3, label="Decision Boundary")

                    title!("Logistic Regression Results\nError Rate: $(round(Pe_log, digits=4))")
                    xaxis!("Feature 1")
                    yaxis!("Feature 2")
                    display(plt)
                end

                logistic_regression_example()

