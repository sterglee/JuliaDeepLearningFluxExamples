
#Julia time: 5.7
# Python time: 13.5

using Flux
using Flux: onehotbatch, onecold, flatten, f32, SamePad, BatchNorm
using MLDatasets
using LinearAlgebra
using Statistics
using CUDA
using Base.Threads

# 0. Parameters
num_classes = 10
batch_size = 32
epochs = 10
lr = 0.0005f0

# 1. Prepare the Data
train_data = CIFAR10(split=:train)
test_data = CIFAR10(split=:test)

x_train = Float32.(train_data.features ./ 255.0f0)
x_test = Float32.(test_data.features ./ 255.0f0)

y_train = onehotbatch(train_data.targets, 0:9)
y_test = onehotbatch(test_data.targets, 0:9)

train_loader = Flux.DataLoader((x_train, y_train), batchsize=batch_size, shuffle=true)

# 2. Build the CNN Model
# FIX: Ensured Dense layers have (input_dim, output_dim)
model = Chain(
    Conv((3, 3), 3 => 32, pad=SamePad(), stride=1),
    BatchNorm(32),
    leakyrelu,
    Conv((3, 3), 32 => 32, pad=SamePad(), stride=2),
    BatchNorm(32),
    leakyrelu,
    Conv((3, 3), 32 => 64, pad=SamePad(), stride=1),
    BatchNorm(64),
    leakyrelu,
    Conv((3, 3), 64 => 64, pad=SamePad(), stride=2),
    BatchNorm(64),
    leakyrelu,
    flatten,
    # CIFAR-10 (32x32) downsampled twice (stride 2) becomes 8x8. 8*8*64 = 4096
    Dense(4096, 128), 
    BatchNorm(128),
    leakyrelu,
    Dropout(0.5),
    Dense(128, num_classes),
    softmax
) |> f32 |> gpu

# 3. Train the Model with Timing
optim = Flux.setup(Flux.Adam(lr), model)

println("Starting training on: $(CUDA.functional() ? "GPU" : "CPU") with $(Threads.nthreads()) threads")
epoch_times = Float64[]

total_duration = @elapsed begin
    for epoch in 1:epochs
        Flux.trainmode!(model)
        
        t_epoch = @elapsed begin
            for (x, y) in train_loader
                x_batch, y_batch = x |> gpu, y |> gpu
                
                loss, grads = Flux.withgradient(model) do m
                    Flux.crossentropy(m(x_batch), y_batch)
                end
                Flux.update!(optim, model, grads[1])
            end
        end
        
        push!(epoch_times, t_epoch)

        # Evaluation
        Flux.testmode!(model)
        model_cpu = model |> cpu
        preds = model_cpu(x_test)
        
        correct = Atomic{Int}(0)
        @threads for i in 1:size(y_test, 2)
            if onecold(preds[:, i]) == onecold(y_test[:, i])
                atomic_add!(correct, 1)
            end
        end
        
        acc = correct[] / size(y_test, 2)
        println("Epoch $epoch: Acc = $(round(acc * 100, digits=2))% | Time: $(round(t_epoch, digits=2))s")
    end
end

# 4. Final Summary
println("\n" * "="^40)
println("Final Performance Summary")
println("-"^40)
println("Total Training Time:    $(round(total_duration, digits=2))s")
println("Average Time per Epoch: $(round(mean(epoch_times), digits=2))s")
println("Total Epochs:           $epochs")
println("="^40)

