using Distributions
using Random
using Plots
using DecisionTree

function decision_tree_example()
    # ----------------------------
    # 1. Generate Training Data
    # ----------------------------
    Random.seed!(0)

    # Means and covariances
    m11 = [0.0, 3.0];   S11 = [0.2 0.0; 0.0 2.0]
    m12 = [11.0, -2.0]; S12 = [3.0 0.0; 0.0 0.5]
    m21 = [3.0, -2.0];  S21 = [5.0 0.0; 0.0 0.5]
    m22 = [7.5, 4.0];   S22 = [7.0 0.0; 0.0 0.5]
    m3 = [7.0, 2.0];    S3 = [8.0 0.0; 0.0 0.5]

    n_pts = 500

    dist11 = MvNormal(m11, S11)
    dist12 = MvNormal(m12, S12)
    dist21 = MvNormal(m21, S21)
    dist22 = MvNormal(m22, S22)
    dist3  = MvNormal(m3, S3)

    X = hcat(rand(dist11, n_pts), rand(dist12, n_pts),
             rand(dist21, n_pts), rand(dist22, n_pts), rand(dist3, n_pts))  # 2 x 2500

    labels = vcat(ones(Int, n_pts), ones(Int, n_pts),
                  2*ones(Int, n_pts), 2*ones(Int, n_pts), 3*ones(Int, n_pts))

    # Plot training data
    scatter(X[1, labels .== 1], X[2, labels .== 1], color=:blue, label="Class 1")
    scatter!(X[1, labels .== 2], X[2, labels .== 2], color=:red, label="Class 2")
    scatter!(X[1, labels .== 3], X[2, labels .== 3], color=:green, label="Class 3")
    title!("Training Data")
    axis_equal = true

    # ----------------------------
    # 2. Generate Test Data
    # ----------------------------
    Random.seed!(100)

    X_test = hcat(rand(dist11, n_pts), rand(dist12, n_pts),
                  rand(dist21, n_pts), rand(dist22, n_pts), rand(dist3, n_pts))

    labels_test = vcat(ones(Int, n_pts), ones(Int, n_pts),
                       2*ones(Int, n_pts), 2*ones(Int, n_pts), 3*ones(Int, n_pts))

    # Plot test data
    scatter(X_test[1, labels_test .== 1], X_test[2, labels_test .== 1], color=:blue, label="Class 1")
    scatter!(X_test[1, labels_test .== 2], X_test[2, labels_test .== 2], color=:red, label="Class 2")
    scatter!(X_test[1, labels_test .== 3], X_test[2, labels_test .== 3], color=:green, label="Class 3")
    title!("Test Data")

    # ----------------------------
    # 3. Train Full Decision Tree
    # ----------------------------
    max_depth = 10  # Maximum depth (simulate unpruned tree)
    dtree = DecisionTreeClassifier(max_depth=max_depth)
    fit!(dtree, X', labels)

    # Predict labels for test set
    predicted_full = predict(dtree, X_test')

    # Compute probability of error
    Pe_full = sum(predicted_full .!= labels_test) / length(labels_test)
    println("Error (full tree): ", Pe_full)

    # ----------------------------
    # 4. Simulate Pruning by Limiting Tree Depth
    # ----------------------------
    max_prune_level = 20
    Pe = zeros(max_prune_level+1)

    for k in 0:max_prune_level
        # Limit depth to k (simulate pruning)
        dtree_pruned = DecisionTreeClassifier(max_depth=k == 0 ? 1 : k)
        fit!(dtree_pruned, X', labels)
        predicted = predict(dtree_pruned, X_test')
        Pe[k+1] = sum(predicted .!= labels_test) / length(labels_test)
    end

    # Plot probability of error vs pruning level
    plot(0:max_prune_level, Pe, marker=:circle, xlabel="Prune Level (Max Depth)", ylabel="Test Error",
         title="Decision Tree Pruning vs Error", color=:blue)
end

decision_tree_example()

