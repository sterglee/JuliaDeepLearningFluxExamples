using LinearAlgebra
using Statistics
using Random
using Distributions
using Plots

function bayes_classification_example()
    # ----------------------------
    # 1. Setup
    # ----------------------------
    Random.seed!(0)  # For reproducibility

    # Mean vectors and covariance matrix
    m1 = [0.0, 0.0]
    m2 = [2.0, 2.0]
    S = [1.0 0.25; 0.25 1.0]

    n_points_per_class = 500

    # ----------------------------
    # 2. Generate Data
    # ----------------------------
    dist1 = MvNormal(m1, S)
    dist2 = MvNormal(m2, S)

    X1 = rand(dist1, n_points_per_class)
    X2 = rand(dist2, n_points_per_class)

    X = hcat(X1, X2)  # 2 x 1000
    labels = vcat(ones(Int, n_points_per_class), 2*ones(Int, n_points_per_class))

    # Plot original data
    scatter(X[1, labels .== 1], X[2, labels .== 1], color=:blue, label="Class 1")
    scatter!(X[1, labels .== 2], X[2, labels .== 2], color=:red, label="Class 2")
    title!("Original Data")

    # ----------------------------
    # 3. Classic Bayes Classification
    # ----------------------------
    P1 = n_points_per_class / size(X, 2)
    P2 = P1

    p1 = pdf.(Ref(dist1), eachcol(X))
    p2 = pdf.(Ref(dist2), eachcol(X))

    class = Int[]
    for i in 1:length(p1)
        push!(class, (P1*p1[i] > P2*p2[i]) ? 1 : 2)
    end

    # Probability of error
    Pe = sum(class .!= labels) / length(labels)
    println("Classic Bayes Probability of Error: ", Pe)

    # Plot classified data
    scatter(X[1, class .== 1], X[2, class .== 1], color=:blue, label="Class 1")
    scatter!(X[1, class .== 2], X[2, class .== 2], color=:red, label="Class 2")
    title!("Classic Bayes Classification")

    # ----------------------------
    # 4. Average Risk Minimization
    # ----------------------------
    L = [0.0 1.0; 0.005 0.0]  # Loss matrix

    class_loss = Int[]
    for i in 1:length(p1)
        push!(class_loss, (L[1,2]*P1*p1[i] > L[2,1]*P2*p2[i]) ? 1 : 2)
    end

    # Average risk
    Ar = 0.0
    for i in 1:length(labels)
        if class_loss[i] != labels[i]
            if labels[i] == 1
                Ar += L[1,2]
            else
                Ar += L[2,1]
            end
        end
    end
    Ar /= length(labels)
    println("Average Risk (Loss-weighted): ", Ar)

    # Plot average risk classification
    scatter(X[1, class_loss .== 1], X[2, class_loss .== 1], color=:blue, label="Class 1")
    scatter!(X[1, class_loss .== 2], X[2, class_loss .== 2], color=:red, label="Class 2")
    title!("Average Risk Minimization Classification")
end

# Run the example
bayes_classification_example()

