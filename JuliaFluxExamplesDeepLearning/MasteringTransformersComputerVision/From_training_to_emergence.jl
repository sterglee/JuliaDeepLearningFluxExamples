using NearestNeighbors
using StaticArrays
using StatsBase   # For mode() function
using Random
using Plots
using MLDataUtils # For train_test_split equivalent

# 1. Generate Synthetic Dataset (1000 samples, 5 classes)
function generate_data(n=1000, centers=5)
    Random.seed!(42)
    # create random centers for our 5 classes
    center_coords = [randn(2) .* 5.0 for _ in 1:centers]
    
    X = Vector{SVector{2, Float32}}()
    y = Vector{Int}()
    
    for i in 1:n
        class_idx = rand(1:centers)
        point = center_coords[class_idx] + randn(2)
        push!(X, SVector{2, Float32}(point))
        push!(y, class_idx)
    end
    # Convert to matrix form for splitting if needed, or keep as SVectors for speed
    return X, y
end

# 2. k-NN Prediction Logic
# Replicates the majority-vote behavior explained by GPT-4
function predict_knn(tree, train_labels, queries, k)
    # idxs contains the indices of the k nearest training points for each query
    idxs, _ = knn(tree, queries, k, true)
    
    predictions = Int[]
    for neighbor_indices in idxs
        # Get labels of neighbors and find the most frequent (mode)
        push!(predictions, mode(train_labels[neighbor_indices]))
    end
    return predictions
end

# 3. Main Execution and Visualization
function main_emergence_plot()
    # Setup data
    X, y = generate_data(1000, 5)
    
    # Build KDTree for efficient O(log n) neighbor search
    kdtree = KDTree(X)
    
    # Create meshgrid for decision boundaries (step size 0.1)
    x_range = -12:0.1:12
    y_range = -12:0.1:12
    grid_points = [SVector{2, Float32}(xi, yi) for xi in x_range, yi in y_range]
    flat_grid = vec(grid_points)
    
    println("Computing emergent boundaries for k=5...")
    grid_preds = predict_knn(kdtree, y, flat_grid, 5)
    
    # Reshape for heatmap
    Z = reshape(grid_preds, length(x_range), length(y_range))

    # Plotting decision boundaries and data points
    p = heatmap(x_range, y_range, Z', 
                c=:viridis, alpha=0.3, 
                title="Emergent k-NN Decision Boundaries (k=5)",
                xlabel="Feature 1", ylabel="Feature 2", legend=false)
    
    scatter!(p, [p[1] for p in X], [p[2] for p in X], 
             marker_z=y, color=:viridis, markersize=2, 
             markerstrokewidth=0, label="Data Points")
    
    display(p)
end

main_emergence_plot()
