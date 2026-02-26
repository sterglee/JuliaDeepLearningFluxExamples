using Flux
using Flux: onehotbatch, onecold, logitcrossentropy, Adam
using MLDatasets
using Statistics

# 1. Define LeNet
function build_lenet(num_classes)
    return Chain(
        Conv((5,5), 1=>20, relu),        # 28x28x1 -> 24x24x20
        MaxPool((2,2)),                  # 24x24x20 -> 12x12x20
        Conv((5,5), 20=>50, relu),       # 12x12x20 -> 8x8x50
        MaxPool((2,2)),                  # 8x8x50 -> 4x4x50
        Flux.flatten,                    # 800
        Dense(800, 500, relu),
        Dense(500, num_classes)          # Logits output
        )
end

# 2. Data Prep
train_data = MNIST(split=:train)
test_data  = MNIST(split=:test)

# Reshape to (W, H, C, N) -> (28, 28, 1, 60000)
train_x = reshape(Float32.(train_data.features), 28, 28, 1, :)
test_x  = reshape(Float32.(test_data.features), 28, 28, 1, :)

classes = 0:9
train_y_oh = onehotbatch(train_data.targets, classes)

loader = Flux.DataLoader((train_x, train_y_oh), batchsize=128, shuffle=true)

# 3. Model & Training
model = build_lenet(length(classes))
opt_state = Flux.setup(Adam(0.001), model)

println("Training LeNet...")
for epoch in 1:3
    for (x, y) in loader
        grads = Flux.gradient(model) do m
            logitcrossentropy(m(x), y)
        end
        Flux.update!(opt_state, model, grads[1])
    end
    acc = mean(onecold(model(test_x), classes) .== test_data.targets)
    println("Epoch $epoch: Test Accuracy = $(round(acc*100, digits=2))%")
end

