using Statistics
using LinearAlgebra
using DataFrames
using MLDatasets
using Plots

# -----------------------------
# 1. Helper Functions
# -----------------------------

function label_it(y, w)
    cls = unique(y)
    cl_w = [sum(w[y .== cl]) for cl in cls]
    return cls[argmax(cl_w)]
end

function to_freq(y, w)
    cls = unique(y)
    n = length(y)
    p = [count(==(cl), y) / n for cl in cls]
    weight = [sum(w[y .== cl]) for cl in cls]
    return p, weight
end

function split_data(X, y, idx, threshold)
    idx_l = X[:, idx] .<= threshold
    idx_r = .!idx_l
    return (X[idx_l, :], y[idx_l]), (X[idx_r, :], y[idx_r]), idx_l, idx_r
end

function get_thresholds(X)
    nf = size(X, 2)
    thresholds = Tuple{Int, Float64}[]
    for i in 1:nf
        ux = sort(unique(X[:, i]))
        if length(ux) > 1
            # Midpoints between consecutive unique values
            t_vals = 0.5 .* (ux[1:end-1] .+ ux[2:end])
            for t in t_vals
                push!(thresholds, (i, t))
            end
        end
    end
    return thresholds
end

function get_imp(p, w, metric)
    if metric == "entropy"
        # -sum(p * log2(p)) weighted
        return -sum(p .* log2.(p .+ 1e-10)) * sum(w)
    elseif metric == "error"
        return (1.0 - maximum(p)) * w[argmax(p)]
    elseif metric == "gini"
        return sum(p .* (1.0 .- p)) * sum(w)
    end
    return 0.0
end

function info_gain(X, y, w, split, metric)
    idx, threshold = split
    (Xl, yl), (Xr, yr), mask_l, mask_r = split_data(X, y, idx, threshold)

    isempty(yl) && return 0.0
    isempty(yr) && return 0.0

    Np = sum(w)
    Nl, Nr = sum(w[mask_l]), sum(w[mask_r])

    pl, wl = to_freq(yl, w[mask_l])
    pr, wr = to_freq(yr, w[mask_r])
    p, weight = to_freq(y, w)

    Il = get_imp(pl, wl, metric)
    Ir = get_imp(pr, wr, metric)
    Ip = get_imp(p, weight, metric)

    return Ip - (Nl/Np * Il) - (Nr/Np * Ir)
end

# -----------------------------
# 2. Tree Structures
# -----------------------------

mutable struct BTNode
    X::Matrix{Float64}
    y::Vector{Int}
    w::Vector{Float64}
    level::Int
    max_depth::Int
    metric::String

    label::Int
    split::Union{Nothing, Tuple{Int, Float64}}
    lchild::Union{Nothing, BTNode}
    rchild::Union{Nothing, BTNode}

    function BTNode(X, y, w, level, max_depth, metric)
        node = new(X, y, w, level, max_depth, metric)
        node.label = label_it(y, w)
        node.split = nothing
        node.lchild = nothing
        node.rchild = nothing
        return node
    end
end

function update!(node::BTNode)
    # Stopping criteria
    sps = get_thresholds(node.X)
    if length(node.y) <= 1 || node.level >= node.max_depth || isempty(sps)
        return node
    end

    # Find best split
    IGs = [info_gain(node.X, node.y, node.w, sp, node.metric) for sp in sps]
    best_sp = sps[argmax(IGs)]
    node.split = best_sp

    # Create children
    (Xl, yl), (Xr, yr), mask_l, mask_r = split_data(node.X, node.y, best_sp[1], best_sp[2])

    if !isempty(yl) && !isempty(yr)
        node.lchild = update!(BTNode(Xl, yl, node.w[mask_l], node.level + 1, node.max_depth, node.metric))
        node.rchild = update!(BTNode(Xr, yr, node.w[mask_r], node.level + 1, node.max_depth, node.metric))
    end

    return node
end

# -----------------------------
# 3. Decision Tree Wrapper
# -----------------------------

struct DecisionTree
    max_depth::Int
    metric::String
    root::Ref{Union{Nothing, BTNode}}

    DecisionTree(; max_depth=3, metric="entropy") = new(max_depth, metric, Ref{Union{Nothing, BTNode}}(nothing))
end

function fit!(dt::DecisionTree, X, y, w)
    dt.root[] = update!(BTNode(X, y, w, 0, dt.max_depth, dt.metric))
end

function find_leaf(node::BTNode, x)
    if isnothing(node.split) || (isnothing(node.lchild) && isnothing(node.rchild))
        return node.label
    end

    idx, threshold = node.split
    if x[idx] <= threshold
        return isnothing(node.lchild) ? node.label : find_leaf(node.lchild, x)
    else
        return isnothing(node.rchild) ? node.label : find_leaf(node.rchild, x)
    end
end

predict(dt::DecisionTree, X::Matrix) = [find_leaf(dt.root[], X[i, :]) for i in 1:size(X, 1)]

# -----------------------------
# 4. Main
# -----------------------------

function main()
    # Using RDatasets or similar would be standard, but for this example:
    # We'll use dummy data mimicking the Iris petal dimensions
    X = [range(1, 5, length=100) range(0.1, 2, length=100)] .+ 0.2 .* randn(100, 2)
    y = Int.(X[:, 1] .> 3.0) # Binary classification dummy
    w = fill(1.0/length(y), length(y))

    model = DecisionTree(max_depth=3, metric="entropy")
    fit!(model, X, y, w)

    # Visualization
    x_rng = range(minimum(X[:,1]), maximum(X[:,1]), length=100)
    y_rng = range(minimum(X[:,2]), maximum(X[:,2]), length=100)

    z = [find_leaf(model.root[], [x, y]) for x in x_rng, y in y_rng]

    p = contourf(x_rng, y_rng, z', alpha=0.3, color=:coolwarm, title="Decision Tree Regions")
    scatter!(p, X[y.==0, 1], X[y.==0, 2], label="Class 0", color=:blue)
    scatter!(p, X[y.==1, 1], X[y.==1, 2], label="Class 1", color=:red)
    display(p)
end

main()

