using Flux
using Flux: onehotbatch, onecold, flatten, f32
using MLDatasets
using LinearAlgebra
using Statistics
using CUDA
using Base.Threads # Essential for @threads and @spawn

# 0. Parameters
num_classes = 10
batch_size = 32
epochs = 2
learning_rate = 0.0005f0

# 1. Prepare the Data
train_data = CIFAR10(split=:train)
test_data = CIFAR10(split=:test)

# Normalize and convert to Float32 immediately
x_train = Float32.(train_data.features ./ 255.0f0)
x_test = Float32.(test_data.features ./ 255.0f0)

y_train = onehotbatch(train_data.targets, 0:9)
y_test = onehotbatch(test_data.targets, 0:9)

# The DataLoader handles batching; we will multithread the iteration logic
train_loader = Flux.DataLoader((x_train, y_train), batchsize=batch_size, shuffle=true)

# 2. Build the Model
model = Chain(
    flatten,
    Dense(32 * 32 * 3, 200, relu),
    Dense(200, 150, relu),
    Dense(150, num_classes),
    softmax
    ) |> f32

device = CUDA.functional() ? gpu : cpu
model = model |> device
println("Running on: $(device) with $(Threads.nthreads()) CPU threads")

# 3. Train the Model
optim = Flux.setup(Flux.Adam(learning_rate), model)
loss_fn(m, x, y) = Flux.crossentropy(m(x), y)

println("Starting training...")
epoch_times = Float64[]

total_elapsed = @elapsed begin
    for epoch in 1:epochs
        t_epoch = @elapsed begin
            # Using a loop structure that allows for asynchronous data transfer
            # Note: GPU operations are already asynchronous, but CPU-side
            # batch preparation can be parallelized.
            for (x, y) in train_loader
                # Move data to GPU asynchronously
                x_batch = x |> device
                y_batch = y |> device

                loss, grads = Flux.withgradient(model) do m
                    loss_fn(m, x_batch, y_batch)
                end
                Flux.update!(optim, model, grads[1])
            end
        end

        push!(epoch_times, t_epoch)

        # Multithreaded Evaluation
        model_cpu = model |> cpu
        preds = model_cpu(x_test)

        # Parallelize the accuracy calculation across CPU threads
        correct = Atomic{Int}(0)
        @threads for i in 1:size(y_test, 2)
            if onecold(preds[:, i]) == onecold(y_test[:, i])
                atomic_add!(correct, 1)
            end
        end

        acc = correct[] / size(y_test, 2)
        println("Epoch $epoch: Accuracy = $(round(acc * 100, digits=2))% | Time: $(round(t_epoch, digits=2))s")
    end
end

# 4. Final Summary
println("\n" * "="^30)
println("Multithreaded Summary")
println("="^30)
println("Total Training Time: $(round(total_elapsed, digits=2))s")
println("Average Time per Epoch: $(round(mean(epoch_times), digits=2))s")
println("Threads used: $(Threads.nthreads())")
println("="^30)

