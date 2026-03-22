using NearestNeighbors
using StaticArrays
using StatsBase  # ERROR FIX: Required for the 'mode' function
using Random
using Plots

# 1. Generate Synthetic Dataset (5 Classes, 1000 samples)
function generate_knn_data(n=1000, centers_count=5)
    Random.seed!(42)
    # create random centers for our 5 classes
    centers = [randn(2) .* 5.0 for _ in 1:centers_count]

        X = Vector{SVector{2, Float32}}()
        y = Vector{Int}()

        for i in 1:n
            class_idx = rand(1:centers_count)
            # Add some noise around the center
            point = centers[class_idx] + randn(2)
            push!(X, SVector{2, Float32}(point))
            push!(y, class_idx)
        end
        return X, y
    end

    # 2. k-NN Classification Logic
    function knn_predict(tree, training_labels, query_points, k)
        # knn returns indices of the k nearest neighbors for each query point
        idxs, dists = knn(tree, query_points, k, true)

        predictions = Int[]
        for neighbor_idxs in idxs
            # Get labels of the k neighbors
            neighbor_labels = training_labels[neighbor_idxs]

            # ERROR FIX: mode() from StatsBase returns the most frequent element
            push!(predictions, mode(neighbor_labels))
        end
        return predictions
    end

    # 3. Main Logic
    function main_knn_complete()
        println("--- Generating Data and Building KDTree ---")
        X_train, y_train = generate_knn_data(1000, 5)

        # KDTree allows O(log n) neighbor lookups
        tree = KDTree(X_train)

        # Setup a grid for decision boundary visualization
        x_range = -12:0.2:12
        y_range = -12:0.2:12
        grid_points = [SVector{2, Float32}(xi, yi) for xi in x_range, yi in y_range]
            flat_grid = vec(grid_points)

            println("Predicting grid labels for emergence visualization...")
                grid_preds = knn_predict(tree, y_train, flat_grid, 5)

                # Reshape predictions back to grid dimensions
                heatmap_data = reshape(grid_preds, length(x_range), length(y_range))

                # Visualization of the emergent boundaries
                p = heatmap(x_range, y_range, heatmap_data',
                            title="Emergent k-NN Boundaries (k=5)",
                            xlabel="Feature 1", ylabel="Feature 2",
                            color=:viridis, alpha=0.4)

                # Overlay original training points
                scatter!(p, [p[1] for p in X_train], [p[2] for p in X_train],
                             group=y_train, markersize=2, markerstrokewidth=0, label="Train Points")

                    display(p)
                    println("Execution Complete.")
                end

                # To run the script:
                 main_knn_complete()

