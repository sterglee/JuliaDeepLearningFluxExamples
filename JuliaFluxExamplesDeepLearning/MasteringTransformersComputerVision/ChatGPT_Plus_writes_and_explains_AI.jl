using Flux
using Flux: train!, onehotbatch, onecold
using Statistics
using Random

# 1. Dataset Generation (Synthesizing the classification data)
Random.seed!(42)
function generate_data(n=1000)
    # Creating two clusters of data
    x1 = randn(2, n ÷ 2) .- 2.0
    x2 = randn(2, n ÷ 2) .+ 2.0
    X = hcat(x1, x2)

    # Labels: 0 for cluster 1, 1 for cluster 2
    y = vcat(zeros(Int, n ÷ 2), ones(Int, n ÷ 2))

    # One-hot encode labels for Flux (2 classes)
    Y = onehotbatch(y, 0:1)

    return Float32.(X), Y
end

X_train, Y_train = generate_data()

# 2. Model Definition
# A simple MLP similar to the logic in the notebook
model = Chain(
    Dense(2 => 8, relu),
    Dense(8 => 2),
    softmax
    )

# 3. Loss Function and Optimizer
loss(m, x, y) = Flux.crossentropy(m(x), y)
opt_state = Flux.setup(Flux.Adam(0.01), model)

# 4. Training Loop
epochs = 100
for epoch in 1:epochs
    Flux.train!(loss, model, [(X_train, Y_train)], opt_state)
    if epoch % 10 == 0
        current_loss = loss(model, X_train, Y_train)
        println("Epoch $epoch: Loss = $current_loss")
    end
end

# 5. Evaluation
y_pred = onecold(model(X_train), 0:1)
y_true = onecold(Y_train, 0:1)
accuracy = mean(y_pred .== y_true)

println("\nFinal Accuracy: $(accuracy * 100)%")

