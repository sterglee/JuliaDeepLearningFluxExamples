using Distributions
using Random
using Plots
using DecisionTree

function boosting_example()
    # ----------------------------
    # 1. Generate Training Data
    # ----------------------------
    Random.seed!(0)

    # Class means and covariances
    m11 = [0.0, 3.0];   S11 = [0.2 0.0; 0.0 2.0]
    m12 = [11.0, -2.0]; S12 = [3.0 0.0; 0.0 0.5]
    m21 = [3.0, -2.0];  S21 = [5.0 0.0; 0.0 0.5]
    m22 = [7.5, 4.0];   S22 = [7.0 0.0; 0.0 0.5]

    n_pts = 100

    dist11 = MvNormal(m11, S11)
    dist12 = MvNormal(m12, S12)
    dist21 = MvNormal(m21, S21)
    dist22 = MvNormal(m22, S22)

    X = hcat(rand(dist11, n_pts), rand(dist12, n_pts),
             rand(dist21, n_pts), rand(dist22, n_pts))  # 2 x 400

    labels = vcat(ones(Int, n_pts), ones(Int, n_pts),
                  2*ones(Int, n_pts), 2*ones(Int, n_pts))   # 400-element vector

    # Plot training data
    scatter(X[1, labels .== 1], X[2, labels .== 1], color=:blue, label="Class 1")
    scatter!(X[1, labels .== 2], X[2, labels .== 2], color=:red, label="Class 2")
    title!("Training Data")
    axis_equal = true

    # ----------------------------
    # 2. Generate Test Data
    # ----------------------------
    Random.seed!(100)

    X_test = hcat(rand(dist11, n_pts), rand(dist12, n_pts),
                  rand(dist21, n_pts), rand(dist22, n_pts))

    labels_test = vcat(ones(Int, n_pts), ones(Int, n_pts),
                       2*ones(Int, n_pts), 2*ones(Int, n_pts))

    # Plot test data
    scatter(X_test[1, labels_test .== 1], X_test[2, labels_test .== 1], color=:blue, label="Class 1")
    scatter!(X_test[1, labels_test .== 2], X_test[2, labels_test .== 2], color=:red, label="Class 2")
    title!("Test Data")

    # ----------------------------
    # 3. AdaBoost using DecisionTree.jl
    # ----------------------------
    n_base_classifiers = 200  # Using smaller number for speed; MATLAB used 12000

    # Build AdaBoost ensemble
    trees, alphas = build_adaboost_stumps(labels, X', n_base_classifiers)

    # ----------------------------
    # 4. Cumulative Loss Function
    # ----------------------------
    function cumulative_loss(alphas, trees, X_data, y_actual)
        n_samples = size(X_data, 1)
        n_iters = length(alphas)
        losses = zeros(n_iters)
        for i in 1:n_iters
            preds = apply_adaboost_stumps(trees[1:i], alphas[1:i], X_data)
            losses[i] = sum(preds .!= y_actual) / n_samples
        end
        return losses
    end

    L_train = cumulative_loss(alphas, trees, X', labels)
    L_test = cumulative_loss(alphas, trees, X_test', labels_test)

    # ----------------------------
    # 5. Plot Learning Curves
    # ----------------------------
    plot(1:n_base_classifiers, L_train, label="Training Error", color=:red, lw=2)
    plot!(1:n_base_classifiers, L_test, label="Test Error", color=:blue, lw=2)
    xlabel!("Number of Base Classifiers")
    ylabel!("Error Rate")
    title!("AdaBoost Learning Curves")
end

# Run the example
boosting_example()

