using Flux, MLDatasets, Statistics
using Flux: onehotbatch, flatten, BatchNorm, SamePad

# 0. Parameters
num_classes = 10
batch_size = 32
epochs = 10
lr = 0.0005f0

# 1. Prepare Data
train_data = CIFAR10(split=:train)
# Normalize and cast to Float32
x_train = Float32.(train_data.features ./ 255.0f0)
y_train = onehotbatch(train_data.targets, 0:9)
loader = Flux.DataLoader((x_train, y_train), batchsize=batch_size, shuffle=true)

# 2. Build Model
model = Chain(
    Conv((3, 3), 3 => 32, pad=SamePad(), stride=1),
    BatchNorm(32), leakyrelu,
    Conv((3, 3), 32 => 32, pad=SamePad(), stride=2),
    BatchNorm(32), leakyrelu,
    Conv((3, 3), 32 => 64, pad=SamePad(), stride=1),
    BatchNorm(64), leakyrelu,
    Conv((3, 3), 64 => 64, pad=SamePad(), stride=2),
    BatchNorm(64), leakyrelu,
    flatten,
    Dense(4096, 128), BatchNorm(128), leakyrelu, Dropout(0.5),
    Dense(128, num_classes),
    softmax
    )

# 3. Train with Timing
optim = Flux.setup(Flux.Adam(lr), model)

total_time = @elapsed begin
    for epoch in 1:epochs
        epoch_time = @elapsed begin
            Flux.trainmode!(model)
            for (x, y) in loader
                loss, grads = Flux.withgradient(model) do m
                    Flux.crossentropy(m(x), y)
                end
                Flux.update!(optim, model, grads[1])
            end
        end
        println("Epoch $epoch complete in $(round(epoch_time, digits=2))s")
    end
end
println("Total Training Time: $(round(total_time, digits=2))s")

