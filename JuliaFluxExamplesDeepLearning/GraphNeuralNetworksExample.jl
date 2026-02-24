############################################################
# Graph Neural Network (GCN) using modern Flux (>=0.14)
# Fully self-contained, CPU only
############################################################

using LinearAlgebra
using Statistics
using Random
using Flux
using Flux: onehotbatch, onecold
using Flux.Losses: logitcrossentropy

Random.seed!(42)

############################################################
# 1. Normalize adjacency
############################################################

function normalize_adjacency(A::Matrix{Float32})
    N = size(A, 1)
    A_hat = A + Matrix{Float32}(I, N, N)
    D = diagm(0 => vec(sum(A_hat, dims=2)))
    D_inv_sqrt = Diagonal(1.0f0 ./ sqrt.(diag(D)))
    return D_inv_sqrt * A_hat * D_inv_sqrt
end

############################################################
# 2. GCN Layer
############################################################

struct GCNLayer
    W::Matrix{Float32}
    b::Vector{Float32}
end

Flux.@functor GCNLayer

function GCNLayer(in_features::Int, out_features::Int)
    W = randn(Float32, in_features, out_features) * sqrt(2 / in_features)
    b = zeros(Float32, out_features)
    GCNLayer(W, b)
end

function (layer::GCNLayer)(A_norm, X)
    A_norm * X * layer.W .+ layer.b'
end

############################################################
# 3. GCN Model
############################################################

struct GCN
    layer1::GCNLayer
    layer2::GCNLayer
end

Flux.@functor GCN

function GCN(in_features, hidden, classes)
    GCN(
        GCNLayer(in_features, hidden),
        GCNLayer(hidden, classes)
        )
end

function (m::GCN)(A_norm, X)
    h = relu.(m.layer1(A_norm, X))
    m.layer2(A_norm, h)
end

############################################################
# 4. Toy Graph
############################################################

A = Float32[
    0 1 1 0 0 0;
    1 0 1 0 0 0;
    1 1 0 0 0 0;
    0 0 0 0 1 1;
    0 0 0 1 0 1;
    0 0 0 1 1 0
    ]

A_norm = normalize_adjacency(A)

X = rand(Float32, 6, 3)

labels = [1,1,1,2,2,2]
Y = onehotbatch(labels, 1:2)'   # (6,2)

############################################################
# 5. Model + Optimizer
############################################################

model = GCN(3, 16, 2)

loss(model) = logitcrossentropy(model(A_norm, X)', Y')

opt = Flux.setup(Flux.Adam(0.01), model)

############################################################
# 6. Training (Modern Flux Style)
############################################################

println("Training...")

epochs = 200

for epoch in 1:epochs

    grads = Flux.gradient(model) do m
        loss(m)
    end

    Flux.update!(opt, model, grads[1])

    if epoch % 20 == 0
        println("Epoch $epoch | Loss = $(loss(model))")
    end
end

############################################################
# 7. Evaluation
############################################################

logits = model(A_norm, X)
pred = onecold(logits', 1:2)

println("\nTrue: ", labels)
println("Pred: ", pred)

println("Accuracy = ", mean(pred .== labels))

