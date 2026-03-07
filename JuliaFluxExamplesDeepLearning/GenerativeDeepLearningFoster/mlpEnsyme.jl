using Flux
using Flux: onehotbatch, onecold, flatten, f32
using MLDatasets
using LinearAlgebra
using Statistics
using CUDA
using Enzyme

# 0. Parameters
num_classes = 10
batch_size = 32
epochs = 2
learning_rate = 0.0005f0

# 1. Prepare the Data
train_data = CIFAR10(split=:train)
test_data = CIFAR10(split=:test)

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

device = CUDA.functional() ? gpu : cpu
model = model |> device

# 3. Train the Model
# We use the explicit parameters for Enzyme compatibility
optim = Flux.setup(Flux.Adam(learning_rate), model)

# Enzyme works best with a functional approach where we pass parameters explicitly
function loss_fn(m, x, y)
    return Flux.crossentropy(m(x), y)
end

println("Starting training with Enzyme...")
epoch_times = Float64[]

total_elapsed = @elapsed begin
    for epoch in 1:epochs
        t_epoch = @elapsed begin
            for (x, y) in train_loader
                x_batch, y_batch = x |> device, y |> device
                
                # Use Enzyme.Active to wrap the model for differentiation
                # Enzyme.Reverse is the standard mode for neural network backprop
                grads = Enzyme.gradient(Enzyme.Reverse, loss_fn, model, x_batch, y_batch)
                
                # Update the model using Flux's update rule
                Flux.update!(optim, model, grads[1])
            end
        end
        
        push!(epoch_times, t_epoch)

        # Evaluation
        model_cpu = model |> cpu
        acc = sum(onecold(model_cpu(x_test)) .== onecold(y_test)) / size(y_test, 2)
        println("Epoch $epoch: Accuracy = $(round(acc * 100, digits=2))% | Time: $(round(t_epoch, digits=2))s")
    end
end

println("\nTotal Training Time: $(round(total_elapsed, digits=2))s")

