# ---------------------------------------------------------------
# Exercise 13.24
# RVM Classification (Julia version)
# ---------------------------------------------------------------

using Random
using LinearAlgebra
using Statistics
using Distributions
using Plots

# ---------------------------
# Random seed (MATLAB rng default)
# ---------------------------
Random.seed!(0)

# ---------------------------
# Data parameters
# ---------------------------
l = 2              # dimension
N = 150            # number of samples

# ---------------------------
# Generate data
# ---------------------------
x1 = 10 .* rand(l, N) .- 5
y1 = zeros(Int, N)

for i in 1:N
    t = 0.05 * (x1[1,i]^3 + x1[1,i]^2 + x1[1,i] + 1)
    if t + 2*randn() > x1[2,i]
        y1[i] = 1
    else
        y1[i] = 0
    end
end

# ---------------------------
# Plot training data
# ---------------------------
scatter(
    x1[1, y1 .== 0],
    x1[2, y1 .== 0],
    color=:gray,
    markersize=4,
    label="Class 0"
    )

scatter!(
    x1[1, y1 .== 1],
    x1[2, y1 .== 1],
    color=:red,
    markersize=4,
    label="Class 1"
    )

# Plot box limits
xmin, xmax = extrema(x1[1,:])
ymin, ymax = extrema(x1[2,:])
box = 1.1 .* [xmin, xmax, ymin, ymax]
xlims!(box[1], box[2])
ylims!(box[3], box[4])

# ---------------------------
# RVM Parameters
# ---------------------------
width = 3.0
maxIts = 1000
initAlpha = (1/N)^2
initBeta = 0.0
useBias = true

# ===============================================================
# 🔧 RVM TRAINING PLACEHOLDER
# ===============================================================
# Replace this function with actual RVM implementation
# ===============================================================

function rvm_train(X, y; width=3.0, maxIts=1000)

    # Dummy placeholder model (for structure only)
    # Replace with real RVM training

    used = collect(1:10)  # pretend 10 relevance vectors
    weights = randn(length(used))
    bias = randn()

    return weights, used, bias
end

# Train RVM
weights, used, bias = rvm_train(x1', y1, width=width, maxIts=maxIts)

# ---------------------------
# Gaussian kernel
# ---------------------------
function gaussian_kernel(X1, X2, width)
    n1 = size(X1,1)
    n2 = size(X2,1)
    K = zeros(n1, n2)
    for i in 1:n1
        for j in 1:n2
            K[i,j] = exp(-norm(X1[i,:] - X2[j,:])^2 / (2*width^2))
        end
    end
    return K
end

# ---------------------------
# Evaluate over grid
# ---------------------------
gsteps = 50
range1 = range(box[1], box[2], length=gsteps)
range2 = range(box[3], box[4], length=gsteps)

grid = [(x,y) for y in range2, x in range1]
    Xgrid = hcat([collect(g) for g in grid]...)'

        # Kernel matrix between grid and relevance vectors
        PHI = gaussian_kernel(Xgrid, x1[:,used]', width)

        y_grid = PHI * weights .+ bias

        # Sigmoid probabilities
        p_grid = 1 ./(1 .+ exp.(-y_grid))

        # Reshape for contour
        P = reshape(p_grid, gsteps, gsteps)

        # Decision boundary
        contour!(
            range1,
            range2,
            P,
            levels=[0.5],
            linewidth=2,
            color=:black,
            label="Decision boundary"
            )

        # Plot relevance vectors
        scatter!(
            x1[1,used],
            x1[2,used],
            markershape=:circle,
            markersize=8,
            markercolor=:black,
            label="Relevance Vectors"
            )

        display(plot!)

