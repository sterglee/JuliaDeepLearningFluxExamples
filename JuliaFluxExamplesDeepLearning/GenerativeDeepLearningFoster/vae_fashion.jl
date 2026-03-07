using Flux, CUDA, MLDatasets, Statistics
using Flux: onehotbatch, flatten, BatchNorm, SamePad
using Base.Threads

# 0. Parameters
num_classes = 10
batch_size = 32
epochs = 10
lr = 0.0005f0

# 1. Prepare Data
train_data = CIFAR10(split=:train)
x_train = Float32.(train_data.features ./ 255.0f0)
y_train = onehotbatch(train_data.targets, 0:9)

# Use parallel=true for multithreaded data fetching
loader = Flux.DataLoader((x_train, y_train), batchsize=batch_size, shuffle=true, parallel=true)

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
    ) |> gpu

# 3. Training Loop with Multithreading and CUDA
optim = Flux.setup(Flux.Adam(lr), model)

for epoch in 1:epochs
    Flux.trainmode!(model)
    @elapsed for (x, y) in loader
        x_batch, y_batch = x |> gpu, y |> gpu
        loss, grads = Flux.withgradient(model) do m
            Flux.crossentropy(m(x_batch), y_batch)
        end
        Flux.update!(optim, model, grads[1])
    end
    println("Epoch $epoch complete")
end
