using Flux, MLDatasets, Statistics, CUDA
using Flux: onehotbatch, flatten, BatchNorm, SamePad, onecold
using Base.Threads # Essential for @threads and Atomic operations

# 0. Parameters
num_classes = 10
batch_size = 32
epochs = 10
lr = 0.0005f0

# 1. Prepare Data
train_data = CIFAR10(split=:train)
test_data = CIFAR10(split=:test)

# Normalize and cast to Float32
x_train = Float32.(train_data.features ./ 255.0f0)
y_train = onehotbatch(train_data.targets, 0:9)
x_test = Float32.(test_data.features ./ 255.0f0)
y_test = onehotbatch(test_data.targets, 0:9)

# Use parallel=true in DataLoader for multithreaded batch fetching
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
    ) |> gpu # Move to GPU if available

# 3. Train with Timing
optim = Flux.setup(Flux.Adam(lr), model)

println("Starting training with $(nthreads()) CPU threads...")

total_time = @elapsed begin
    for epoch in 1:epochs
        epoch_time = @elapsed begin
            Flux.trainmode!(model)
            for (x, y) in loader
                x_batch, y_batch = x |> gpu, y |> gpu
                loss, grads = Flux.withgradient(model) do m
                    Flux.crossentropy(m(x_batch), y_batch)
                end
                Flux.update!(optim, model, grads[1])
            end
        end

        # --- Multithreaded Evaluation ---
        Flux.testmode!(model)
        # Move model to CPU for comparison if x_test is large, or keep on GPU
        preds = model(x_test |> gpu) |> cpu

        # Use Atomic integers to safely count correct predictions across threads
        correct = Atomic{Int}(0)
        @threads for i in 1:size(y_test, 2)
            if onecold(preds[:, i]) == onecold(y_test[:, i])
                atomic_add!(correct, 1)
            end
        end

        accuracy = correct[] / size(y_test, 2)
        println("Epoch $epoch: Acc = $(round(accuracy * 100, digits=2))% | Time: $(round(epoch_time, digits=2))s")
    end
end

