using LinearAlgebra
using Random
using GLMakie

# -------------------------------
# 1. Δημιουργία συνθετικού dataset
# -------------------------------
Random.seed!(42)
N = 100
D = 2
K = 3

X = randn(N, D)

function assign_class(x)
    if x[1] + x[2] > 1
        return 1
        elseif x[1] - x[2] > 0
        return 2
    else
        return 3
    end
end

T = zeros(N, K)
true_classes = zeros(Int, N)
for i in 1:N
    c = assign_class(X[i, :])
    T[i, c] = 1.0
    true_classes[i] = c
end

# -------------------------------
# 2. Least-Squares Training
# -------------------------------
X_aug = hcat(ones(N), X)
W = X_aug \ T  # Υπολογισμός βαρών (D+1 x K)

# -------------------------------
# 3. Visualization με GLMakie
# -------------------------------
fig = Figure(size = (800, 600))
ax = Axis(fig[1, 1],
          title = "Least Squares Linear Classifier (GLMakie)",
          xlabel = "x1", ylabel = "x2")

# Δημιουργία πλέγματος για την επιφάνεια απόφασης
x_min, x_max = minimum(X[:,1]) - 0.5, maximum(X[:,1]) + 0.5
y_min, y_max = minimum(X[:,2]) - 0.5, maximum(X[:,2]) + 0.5

x_range = range(x_min, x_max, length=200)
y_range = range(y_min, y_max, length=200)

# Matrix για την αποθήκευση των προβλέψεων σε κάθε pixel
grid_preds = zeros(length(x_range), length(y_range))

for (i, x_val) in enumerate(x_range)
    for (j, y_val) in enumerate(y_range)
        # Πρόβλεψη: y = [1 x1 x2] * W
        scores = [1.0, x_val, y_val]' * W
        # Το argmax επιστρέφει CartesianIndex, παίρνουμε το index της στήλης
        grid_preds[i, j] = argmax(vec(scores))
    end
end

# 1. Heatmap για τις περιοχές απόφασης
heatmap!(ax, x_range, y_range, grid_preds,
         colormap = [:lightblue, :lightpink, :lightgreen],
         alpha = 0.4)

# 2. Decision Boundaries (Contours)
contour!(ax, x_range, y_range, grid_preds, color = :black, linewidth = 1)

# 3. Scatter plot των δεδομένων εκπαίδευσης
colors = [:blue, :red, :green]
for k in 1:K
    idx = findall(==(k), true_classes)
    scatter!(ax, X[idx, 1], X[idx, 2],
             color = colors[k],
             strokewidth = 1, strokecolor = :white,
             label = "Class $k", markersize = 12)
end

axislegend(ax, position = :rt)

# -------------------------------
# 4. Αποθήκευση και Εμφάνιση
# -------------------------------
save("least_squares_classifier.png", fig)
println("Plot saved as least_squares_classifier.png")

display(fig)

