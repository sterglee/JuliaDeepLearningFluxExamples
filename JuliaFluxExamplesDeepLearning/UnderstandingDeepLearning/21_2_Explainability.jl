using Flux
using GLM
using DataFrames
using LinearAlgebra
using Statistics
using Plots
using Random

# --- 1. Black-Box Model ---
# Creating a simple non-linear model to explain
Random.seed!(42)
n_features = 4
black_box_model = Chain(
    Dense(n_features, 32, relu),
    Dense(32, 1, sigmoid)
    )

# --- 2. LIME Explanation Function ---
function explain_lime(model, x_query; n_samples=500, sigma=0.1)
    # 1. Generate Perturbations
    # We use Float32 for noise to match the Flux model's expected input type
    noise = randn(Float32, length(x_query), n_samples) .* Float32(sigma)
    X_perturbed = x_query .+ noise  # Shape: (Features, Samples)

    # 2. Get Black-Box Predictions
    # We treat the model as an opaque function
    y_perturbed = vec(model(X_perturbed))

    # 3. Calculate Weights (Distance-based)
    # Points closer to the query point have higher weight
    distances = [norm(X_perturbed[:, i] .- x_query) for i in 1:n_samples]
        weights = exp.(-(distances .^ 2) ./ (2 * sigma^2))

        # --- 4. Prepare for GLM (The Fix) ---
        # Convert all data to Float64 to satisfy GLM's internal solver (delbeta!)
        X_64 = Float64.(X_perturbed')
        Y_64 = Float64.(y_perturbed)
        W_64 = Float64.(weights)

        # Build DataFrame for regression
        df = DataFrame(X_64, :auto)
        df.target = Y_64

        # Define linear formula: target ~ x1 + x2 + x3 + x4
        feature_names = names(df, Not(:target))
        formula = Term(:target) ~ sum(term.(feature_names))

        # Fit the local linear surrogate model
        #
        lime_model = lm(formula, df, wts=W_64)

        return lime_model
    end

    # --- 3. Execution ---
    # Define a query point (must be Float32 for Flux)
    x_instance = Float32[0.5, -0.2, 0.8, 0.1]

    # Get prediction
    prediction = black_box_model(x_instance)[1]
    println("Black-box prediction for instance: ", round(prediction, digits=4))

        # Generate Explanation using our function
        # Ensure the name here matches the function definition above!
        explanation = explain_lime(black_box_model, x_instance)

        # --- 4. Display Results ---
        # Coefficients represent local feature importance
        importance = coef(explanation)[2:end]
        println("\nLIME Local Coefficients (Feature Importance):")
        for (i, val) in enumerate(importance)
            println("Feature $i: ", round(val, digits=4))
        end

        #
        bar(1:n_features, importance,
            title="LIME Local Explanation",
            xlabel="Feature Index",
            ylabel="Contribution",
            legend=false)

