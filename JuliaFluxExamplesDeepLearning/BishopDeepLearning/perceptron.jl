using Random
using LinearAlgebra
using GLMakie

# -----------------------------
# 1. Δημιουργία τυχαίων δεδομένων
# -----------------------------
Random.seed!(42)
N = 100

# Generate data as Matrices for easier plotting in Makie
x1_mat = randn(N, 2) .+ 2.0
x2_mat = randn(N, 2) .+ 5.0

X = vcat(x1_mat, x2_mat)
# Convert to a vector of vectors for the learning loop
X_vecs = [X[i, :] for i in 1:size(X, 1)]
    y_true = vcat(ones(N), zeros(N))

    # -----------------------------
    # 2. Αρχικοποίηση & Εκπαίδευση
    # -----------------------------
    w = randn(2)
    w0 = 0.0
    η = 0.1
    epochs = 20

    # Perceptron functions
    predict(x, w, w0) = (dot(w, x) + w0) >= 0 ? 1.0 : 0.0

    for epoch in 1:epochs
        for i in 1:length(X_vecs)
            y_i = predict(X_vecs[i], w, w0)
            error = y_true[i] - y_i
            w .+= η * error .* X_vecs[i]
            w0 += η * error
        end
    end

    # -----------------------------
    # 3. Οπτικοποίηση με GLMakie
    # -----------------------------
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1],
              title = "Perceptron Linear Discriminant (GLMakie)",
              xlabel = "x1", ylabel = "x2")

    # Plot Class 1 and Class 2
    scatter!(ax, x1_mat[:, 1], x1_mat[:, 2], color = (:blue, 0.5), label = "Class 1", markersize = 12)
    scatter!(ax, x2_mat[:, 1], x2_mat[:, 2], color = (:red, 0.5), label = "Class 2", markersize = 12)

    # Calculate Decision Boundary
    x_min, x_max = minimum(X[:, 1]) - 1, maximum(X[:, 1]) + 1
    x_vals = [x_min, x_max]
    # Derived from w1*x + w2*y + w0 = 0  => y = (-w1*x - w0) / w2
    y_vals = (-w[1] .* x_vals .- w0) ./ w[2]

    # Plot Boundary Line
    lines!(ax, x_vals, y_vals, color = :black, linewidth = 3, label = "Decision Boundary")

    # Add Legend
    axislegend(ax, position = :rt)

    # -----------------------------
    # 4. Αποθήκευση (Save)
    # -----------------------------
    # display(fig) # Opens the interactive window
    save("perceptron_glmakie.png", fig)

    println("Plot saved as perceptron_glmakie.png")

