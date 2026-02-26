using Flux, CUDA
using Flux: DataLoader, onehotbatch, onecold, flatten, logitcrossentropy
using MLDatasets, Statistics, Printf

# Check if CUDA is functional
if CUDA.functional()
    println("GPU detected: ", CUDA.name(CUDA.device()))
else
    @warn "No GPU detected, falling back to CPU."
end

# --- 1. Constants ---
const NUM_CLASSES = 10
const BATCH_SIZE = 16
const EPOCHS = 50

# --- 2. Data Loading & Normalization ---
function load_data()
    train_data = MLDatasets.CIFAR10(split=:train)
    test_data  = MLDatasets.CIFAR10(split=:test)

    x_train, y_train = train_data[:]
    x_test,  y_test  = test_data[:]

    # Global Mean/Std for normalization
    m, s = mean(x_train), std(x_train)

    x_train = (x_train .- m) ./ (s + 1f-7)
    x_test  = (x_test  .- m) ./ (s + 1f-7)

    # One-hot labels as Float32
    y_train = Float32.(onehotbatch(y_train, 0:9))
    y_test  = Float32.(onehotbatch(y_test, 0:9))

    return x_train, y_train, x_test, y_test
end

x_train, y_train, x_test, y_test = load_data()

# Move test data to GPU once (it stays there for evaluation)
x_test_gpu = x_test |> gpu
y_test_gpu = y_test |> gpu

# DataLoader stays on CPU; we move batches to GPU during the loop
train_loader = DataLoader((x_train, y_train), batchsize=BATCH_SIZE, shuffle=true)

# --- 3. Build Model & Move to GPU ---
# 32x32 -> 16x16 -> 8x8 -> 4x4
model = Chain(
    Conv((3,3), 3=>32, relu, pad=1),
    BatchNorm(32),
    Conv((3,3), 32=>32, relu, pad=1),
    BatchNorm(32),
    MaxPool((2,2)),
    Dropout(0.2f0),

    Conv((3,3), 32=>64, relu, pad=1),
    BatchNorm(64),
    Conv((3,3), 64=>64, relu, pad=1),
    BatchNorm(64),
    MaxPool((2,2)),
    Dropout(0.3f0),

    Conv((3,3), 64=>128, relu, pad=1),
    BatchNorm(128),
    Conv((3,3), 128=>128, relu, pad=1),
    BatchNorm(128),
    MaxPool((2,2)),
    Dropout(0.4f0),

    flatten,
    Dense(4 * 4 * 128 => NUM_CLASSES)
    ) |> f32 |> gpu  # Ensure Float32 and move parameters to VRAM

# --- 4. Training Setup ---
opt_state = Flux.setup(RMSProp(), model)

function get_accuracy(m, x, y)
    Flux.testmode!(m)
    # Calculation happens on GPU
    acc = mean(onecold(m(x), 0:9) .== onecold(y, 0:9))
    Flux.trainmode!(m)
    return acc
end

# --- 5. Training Loop ---
println("Starting GPU Training...")

for epoch in 1:EPOCHS
    for (batch_x, batch_y) in train_loader
        # Move current batch to GPU
        x_g, y_g = batch_x |> gpu, batch_y |> gpu

        grads = Flux.gradient(model) do m
            logitcrossentropy(m(x_g), y_g)
        end
        Flux.update!(opt_state, model, grads[1])

        # Free batch memory (optional, but good for large models)
        CUDA.unsafe_free!(x_g)
        CUDA.unsafe_free!(y_g)
    end

    val_acc = get_accuracy(model, x_test_gpu, y_test_gpu)
    @printf("Epoch %d: Test Accuracy: %.2f%%\n", epoch, val_acc * 100)
end

# --- 6. Final Evaluation ---
Flux.testmode!(model)
final_loss = logitcrossentropy(model(x_test_gpu), y_test_gpu)
@printf("\nFinal Result -> Loss: %.4f | Accuracy: %.2f%%\n",
        final_loss, get_accuracy(model, x_test_gpu, y_test_gpu) * 100)

