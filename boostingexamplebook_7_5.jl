using LinearAlgebra
using Random
using Distributions
using DecisionTree
using Plots

# 1. Setup and Training Data Generation
Random.seed!(0)
l_dim = 20
n_per_group = 100

# Defining means and covariances
m11, m12, m21 = zeros(l_dim), ones(l_dim), [zeros(l_dim÷2); ones(l_dim÷2)]
S = eye = Matrix{Float64}(I, l_dim, l_dim)

# Generating multivariate normal samples
X1 = rand(MvNormal(m11, S), n_per_group)
X2 = rand(MvNormal(m12, S), n_per_group)
X3 = rand(MvNormal(m21, S), n_per_group)

# Training Features and Labels
X = hcat(X1, X2, X3)'  # Shape: (300, 20)
labels = [ones(Int, 200); 2*ones(Int, 100)]

# 2. Test Data Generation
Random.seed!(100)
X1_test = rand(MvNormal(m11, S), n_per_group)
X2_test = rand(MvNormal(m12, S), n_per_group)
X3_test = rand(MvNormal(m21, S), n_per_group)

X_test = hcat(X1_test, X2_test, X3_test)'
labels_test = [ones(Int, 200); 2*ones(Int, 100)]

# 3. Training the AdaBoost Ensemble
# Parameters: labels, features, number of iterations, pruning/depth
n_iterations = 2000
model = build_adaboost_stumps(labels, X, n_iterations)

# 4. Evaluating Cumulative Loss
function cumulative_loss(model, X_data, y_actual)
    n_samples = size(X_data, 1)
    n_iters = length(model.coeffs)
    losses = zeros(n_iters)

    # In AdaBoost, we track the ensemble prediction as we add more trees
    for i in 1:n_iters
        # Create a sub-ensemble of the first 'i' classifiers
        sub_model = Ensemble(model.coeffs[1:i], model.trees[1:i])
        preds = apply_adaboost_stumps(sub_model, X_data)
        losses[i] = sum(preds .!= y_actual) / n_samples
    end
    return losses
end

println("Calculating cumulative loss...")
L_train = cumulative_loss(model, X, labels)
L_test = cumulative_loss(model, X_test, labels_test)

# 5. Visualization
p = plot(1:n_iterations, L_train, color=:red, label="Training Error", lw=1.5)
plot!(1:n_iterations, L_test, color=:blue, label="Test Error", lw=1.5)
xlabel!("Number of base classifiers")
ylabel!("Classification Error")
title!("AdaBoost Convergence (Theodoridis Example)")

display(p)


