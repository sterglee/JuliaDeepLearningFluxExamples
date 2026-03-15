
# Total Duration with CUDA: 3.75 seconds Julia, 22.048139 secs Tensorflow 
# Julia Native SVD Benchmark:0.032900 seconds (3.66 k allocations: 11.469 MiB)
# Python Native SVD Time: 5.9220 s

using Flux
using Flux: DataLoader, onehotbatch, onecold, flatten, logitcrossentropy
using MLDatasets
using Statistics
using Printf
using CUDA  # Required for GPU support

# --- 1. Constants ---
const BATCH_SIZE = 128
const EPOCHS = 5
const CLASSES = 10
const VALIDATION_SPLIT = 0.2

# Check if CUDA is functional
if CUDA.functional()
    @info "CUDA is available. Training on GPU."
    device = gpu
else
    @warn "CUDA not found. Falling back to CPU."
    device = cpu
end

# --- 2. Data Preparation ---
println("Loading CIFAR-10...")
train_data = MLDatasets.CIFAR10(split=:train)
x_all, y_all = train_data[:]

# Ensure Float32 for GPU compatibility
x_all = Float32.(x_all)
y_all = Int.(y_all) # Labels as Integers for one-hot

n_samples = size(x_all, 4)
n_val = Int(floor(n_samples * VALIDATION_SPLIT))
n_train = n_samples - n_val

x_train = x_all[:, :, :, 1:n_train]
y_train = y_all[1:n_train]
x_val   = x_all[:, :, :, n_train+1:end]
y_val   = y_all[n_train+1:end]

y_train_oh = onehotbatch(y_train, 0:9)
y_val_oh   = onehotbatch(y_val, 0:9)

# We keep the full dataset on CPU and move batches to GPU during training
# to save VRAM, or move it all now if it fits:
x_val_gpu = x_val |> device
y_val_gpu = y_val_oh |> device

train_loader = DataLoader((x_train, y_train_oh), batchsize=BATCH_SIZE, shuffle=true)

# --- 3. Build the Model ---
# 32x32 -> Conv(3x3) -> 30x30 -> MaxPool(2x2) -> 15x15
model = Chain(
    Conv((3, 3), 3 => 32, relu),
    MaxPool((2, 2)),
    Dropout(0.25),
    flatten,
    Dense(15 * 15 * 32 => 512, relu),
    Dropout(0.5),
    Dense(512 => CLASSES)
) |> device  # Move model to GPU

# --- 4. Training Setup ---
opt_state = Flux.setup(RMSProp(), model)

# Accuracy helper (must handle device movement)
function accuracy(x, y, m)
    pred = m(x)
    mean(onecold(cpu(pred), 0:9) .== onecold(cpu(y), 0:9))
end

# --- 5. Training Loop ---
println("Starting Training on $(device)...")

start_time = time()
for epoch in 1:EPOCHS
    for (batch_x, batch_y) in train_loader
        # Move batch to GPU
        x_g, y_g = batch_x |> device, batch_y |> device

        grads = Flux.gradient(model) do m
            result = m(x_g)
            logitcrossentropy(result, y_g)
        end
        Flux.update!(opt_state, model, grads[1])
    end

    # Validation (using the GPU-stored validation set)
    val_acc = accuracy(x_val_gpu, y_val_gpu, model)
    @printf("Epoch %d/%d: Val Acc: %.2f%%\n", epoch, EPOCHS, val_acc * 100)
end
total_time = time() - start_time

# --- 6. Final Evaluation ---
test_data = MLDatasets.CIFAR10(split=:test)
xt, yt = test_data[:]
xt_gpu = Float32.(xt) |> device
yt_gpu = onehotbatch(yt, 0:9) |> device

final_acc = accuracy(xt_gpu, yt_gpu, model)
println("\nFinal Test Accuracy: ", round(final_acc * 100, digits=2), "%")
@printf("Total Duration: %.6f seconds\n", total_time)

