# julia time:  1.5 sec
# python tensorflow: 3.56

using Flux
using Flux: onehotbatch, onecold, flatten, f32
using MLDatasets
using LinearAlgebra
using Statistics
using CUDA

# 0. Parameters
num_classes = 10
batch_size = 32
epochs = 10
learning_rate = 0.0005f0

# 1. Prepare the Data
train_data = CIFAR10(split=:train)
test_data = CIFAR10(split=:test)

# Normalize and convert to Float32
x_train = Float32.(train_data.features ./ 255.0f0)
x_test = Float32.(test_data.features ./ 255.0f0)

y_train = onehotbatch(train_data.targets, 0:9)
y_test = onehotbatch(test_data.targets, 0:9)

train_loader = Flux.DataLoader((x_train, y_train), batchsize=batch_size, shuffle=true)

# 2. Build the Model
model = Chain(
    flatten,
    Dense(32 * 32 * 3, 200, relu),
    Dense(200, 150, relu),
    Dense(150, num_classes),
    softmax
) |> f32

# Device selection
device = CUDA.functional() ? gpu : cpu
model = model |> device
println("Running on: $(device)")

# 3. Train the Model
optim = Flux.setup(Flux.Adam(learning_rate), model)
loss_fn(m, x, y) = Flux.crossentropy(m(x), y)

println("Starting training...")
epoch_times = Float64[]

# Capture the total time for all epochs
total_elapsed = @elapsed begin
    for epoch in 1:epochs
        # Capture the time for this specific epoch
        t_epoch = @elapsed begin
            for (x, y) in train_loader
                x_batch, y_batch = x |> device, y |> device
                
                loss, grads = Flux.withgradient(model) do m
                    loss_fn(m, x_batch, y_batch)
                end
                Flux.update!(optim, model, grads[1])
            end
        end
        
        push!(epoch_times, t_epoch)

        # Evaluation
        model_cpu = model |> cpu
        acc = sum(onecold(model_cpu(x_test)) .== onecold(y_test)) / size(y_test, 2)
        println("Epoch $epoch: Accuracy = $(round(acc * 100, digits=2))% | Epoch Time: $(round(t_epoch, digits=2))s")
    end
end

# 4. Final Summary
println("\n" * "="^30)
println("Final Performance Summary")
println("="^30)
println("Total Training Time: $(round(total_elapsed, digits=2))s")
println("Average Time per Epoch: $(round(mean(epoch_times), digits=2))s")
println("="^30)
