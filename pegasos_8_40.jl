using LinearAlgebra, Random, Statistics, Plots

# --- 1. The Pegasos Algorithm Function ---
function pegasos(X, Y; λ=0.1, k=10, maxIter=2000, tol=1e-5)
    N, d = size(X)

    # Initialization
    w = rand(d)
    w ./= (sqrt(λ) * norm(w))

    # Store history for averaging (as per Pegasos theory)
    w_history = zeros(maxIter, d)
    t_final = maxIter

    for t in 1:maxIter
        b = mean(Y - X * w) # Update bias

        # Stochastic Step: Select subset of size k
        idx = rand(1:N, k)
        At, yt = X[idx, :], Y[idx]

        # Identify margin violations: y_i(w'x_i + b) < 1
        violations = (At * w .+ b) .* yt .< 1

        η_t = 1.0 / (λ * t) # Learning rate

        # Calculate sub-gradient
        grad = zeros(d)
        if any(violations)
            grad = sum(yt[violations] .* At[violations, :], dims=1)[:]
        end

        # Update and Project
        w_new = (1 - η_t * λ) * w + (η_t / k) * grad
        w_new *= min(1.0, 1.0 / (sqrt(λ) * norm(w_new)))

        w_history[t, :] = w_new

        if norm(w_new - w) < tol
            t_final = t
            break
        end
        w = w_new
    end

    # Return averaged weights for stability
    w_avg = mean(w_history[1:t_final, :], dims=1)[:]
    b_avg = mean(Y - X * w_avg)

    return w_avg, b_avg
end

# --- 2. Main Example Call ---
function run_pegasos_example()
    # Generate two separable clusters
    Random.seed!(123)
    X = [randn(50, 2) .+ 2.5; randn(50, 2) .- 2.5]
    Y = [ones(50); -ones(50)]

    println("Training Pegasos SVM...")
    w, b = pegasos(X, Y, λ=0.01, k=10, maxIter=1000)

    # Calculate Accuracy
    preds = sign.(X * w .+ b)
    acc = 100 * sum(preds .== Y) / length(Y)
    println("Final Training Accuracy: $acc%")

    # Visualize Decision Boundary
    p = scatter(X[1:50,1], X[1:50,2], color=:blue, label="Class +1")
    scatter!(X[51:100,1], X[51:100,2], color=:red, label="Class -1")

    # Solve for w1*x + w2*y + b = 0 => y = -(w1*x + b)/w2
    x_range = range(-5, 5, length=100)
    y_boundary = -(w[1] .* x_range .+ b) ./ w[2]
    plot!(x_range, y_boundary, color=:black, lw=2, label="Decision Boundary")

    display(p)
end

run_pegasos_example()

