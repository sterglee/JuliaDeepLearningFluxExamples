using Random
using Distributions
using LinearAlgebra
using GLMakie

# --------------------------
# 1. Δημιουργία συνθετικών δεδομένων
# --------------------------
Random.seed!(42)

n_samples = 200
μ1, σ1 = [2.0, 2.0], 1.0
μ2, σ2 = [5.0, 5.0], 1.5

dist1 = MvNormal(μ1, σ1 * I)
dist2 = MvNormal(μ2, σ2 * I)

x1 = rand(dist1, n_samples)' # Matrix N x 2
x2 = rand(dist2, n_samples)'

X = vcat(x1, x2)
# True labels for the original points
y_true = vcat(fill(1, n_samples), fill(2, n_samples))

# --------------------------
# 2. Παράμετροι Απόφασης
# --------------------------
# Lkj: rows = actual, cols = predicted
# Here, misclassifying Class 1 as 2 costs 1,
# but misclassifying Class 2 as 1 costs 100!
L = [0 1; 100 0]
θ = 0.7  # Reject threshold

# --------------------------
# 3. Συναρτήσεις Απόφασης
# --------------------------
function posterior_prob(x)
    p1 = pdf(dist1, x)
    p2 = pdf(dist2, x)
    return [p1, p2] ./ (p1 + p2)
end

function classify_with_reject(x, L, θ)
    probs = posterior_prob(x)
    max_prob = maximum(probs)

    if max_prob < θ
        return 0.0  # Reject
    else
        # Expected loss for choosing class j
        expected_loss = [sum(L[:, j] .* probs) for j in 1:2]
            return Float64(argmin(expected_loss))
        end
    end

    # --------------------------
    # 4. Οπτικοποίηση με GLMakie
    # --------------------------
    fig = Figure(size = (900, 700))
    ax = Axis(fig[1, 1],
              title = "Decision Theory: Expected Loss (L21=100) & Reject (θ=$θ)",
              xlabel = "x1", ylabel = "x2")

    # Create a grid for the decision regions
    x_grid = range(minimum(X[:,1])-1, maximum(X[:,1])+1, length=200)
    y_grid = range(minimum(X[:,2])-1, maximum(X[:,2])+1, length=200)

    # Evaluate decision at every pixel
    grid_preds = [classify_with_reject([x, y], L, θ) for x in x_grid, y in y_grid]

        # 1. Plot Decision Regions (Heatmap)
        # 0 = Reject (Grey), 1 = Class 1 (Blue), 2 = Class 2 (Red)
        heatmap!(ax, x_grid, y_grid, grid_preds,
                 colormap = [:lightgrey, :lightblue, :lightpink],
                 alpha = 0.5)

        # 2. Plot Decision Boundaries
        contour!(ax, x_grid, y_grid, grid_preds, color = :black, linewidth = 1)

        # 3. Plot Original Data Points
        scatter!(ax, x1[:, 1], x1[:, 2], color = :blue, label = "Class 1 Samples", markersize = 8)
        scatter!(ax, x2[:, 1], x2[:, 2], color = :red, label = "Class 2 Samples", markersize = 8)

        # 4. Add Legend
        # Create custom elements for the legend to show the "Reject" zone
        poly_reject = PolyElement(color = :lightgrey, strokecolor = :black)
        axislegend(ax,
                   [[poly_reject], MarkerElement(marker=:circle, color=:blue), MarkerElement(marker=:circle, color=:red)],
                   ["Reject Zone", "Class 1", "Class 2"],
                   position = :rb)

        # --------------------------
        # 5. Αποθήκευση
        # --------------------------
        save("decision_theory_reject_option.png", fig)
        println("Figure saved as decision_theory_reject_option.png")
        display(fig)

