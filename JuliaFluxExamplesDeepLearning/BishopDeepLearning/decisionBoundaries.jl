using LinearAlgebra
using Random
using GLMakie
using Statistics

# -----------------------------
# 1. Δημιουργία συνθετικών δεδομένων
# -----------------------------
Random.seed!(1234)
N = 100          # αριθμός δειγμάτων
D = 2            # διάσταση εισόδου
K = 3            # αριθμός κλάσεων (Blue, Red, Green)

centers = [ [2.0, 2.0], [-2.0, -2.0], [2.0, -2.0] ]

X = zeros(N, D)
t = zeros(N, K)   # 1-of-K encoding
class_labels = zeros(Int, N)

for n in 1:N
    k = rand(1:K)           
    X[n, :] = centers[k] .+ 0.6 * randn(D) # Increased noise for better visual overlap
    t[n, k] = 1.0
    class_labels[n] = k
end

# -----------------------------
# 2. Εφαρμογή Least-Squares
# -----------------------------
# W = (X_augᵀ X_aug)⁻¹ X_augᵀ T
X_aug = hcat(ones(N), X)   
W = pinv(X_aug) * t   

# -----------------------------
# 3. Οπτικοποίηση (Decision Surface)
# -----------------------------
fig = Figure(size = (900, 700))
ax = Axis(fig[1, 1], 
    title = "Least Squares Linear Discriminant (3 Classes)",
    xlabel = "Feature x1", ylabel = "Feature x2")

# Δημιουργία πλέγματος (Grid) για τις περιοχές απόφασης
x_range = range(-5, 5, length=200)
y_range = range(-5, 5, length=200)

# Matrix to store the predicted class ID (1, 2, or 3) for each pixel
grid_preds = zeros(length(x_range), length(y_range))

for (i, x) in enumerate(x_range)
    for (j, y) in enumerate(y_range)
        # Linear Model: y(x) = Wᵀ [1, x1, x2]
        # We use vec() to turn the 1xK matrix into a Vector so argmax returns an Int
        scores = vec([1.0, x, y]' * W)
        grid_preds[i, j] = argmax(scores)
    end
end

# 1. Σχεδίαση Heatmap (Περιοχές κλάσεων)
heatmap!(ax, x_range, y_range, grid_preds, 
    colormap = [:lightblue, :lightpink, :lightgreen], 
    alpha = 0.5)

# 2. Σχεδίαση Περιγράμματος (Decision Boundaries)
contour!(ax, x_range, y_range, grid_preds, color = :black, linewidth = 1.5)

# 3. Σχεδίαση των αρχικών δεδομένων
colors = [:blue, :red, :green]
for k in 1:K
    idxs = findall(==(k), class_labels)
    scatter!(ax, X[idxs, 1], X[idxs, 2], 
        color = colors[k], 
        strokewidth = 1.5, strokecolor = :white,
        label = "Class $k", markersize = 14)
end

axislegend(ax, position = :rt, framevisible = true)

# -----------------------------
# 4. Αποθήκευση και Εμφάνιση
# -----------------------------
# Save the figure as a high-quality bitmap
save("linear_multiclass_results.png", fig)

println("Plot saved successfully to 'linear_multiclass_results.png'")
display(fig)