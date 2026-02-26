using Flux
using Flux: DataLoader, train!, onehotbatch, onecold, flatten, logitcrossentropy
using MLDatasets
using Statistics
using Printf

# --- 1. Constants & Hyperparameters ---
const IMG_SIZE = (32, 32, 3)
const BATCH_SIZE = 128
const EPOCHS = 5
const CLASSES = 10
const VALIDATION_SPLIT = 0.2

# --- 2. Data Preparation ---
println("Loading CIFAR-10...")
# Load full training set
train_data = MLDatasets.CIFAR10(split=:train)
x_all, y_all = train_data[:]

# Normalize to [0, 1] and ensure Float32
x_all = Float32.(x_all)

# Manual Validation Split (0.2)
n_samples = size(x_all, 4)
n_val = Int(floor(n_samples * VALIDATION_SPLIT))
n_train = n_samples - n_val

x_train = x_all[:, :, :, 1:n_train]
y_train = y_all[1:n_train]
x_val   = x_all[:, :, :, n_train+1:end]
y_val   = y_all[n_train+1:end]

# One-hot encode labels (0:9 for CIFAR-10)
y_train_oh = onehotbatch(y_train, 0:9)
y_val_oh   = onehotbatch(y_val, 0:9)

# Create DataLoader
train_loader = DataLoader((x_train, y_train_oh), batchsize=BATCH_SIZE, shuffle=true)

# --- 3. Build the Model (Equivalent to Keras Sequential) ---
# Calculation for Flatten: 32x32 -> Conv(3x3) -> 30x30 -> MaxPool(2x2) -> 15x15
# 15 * 15 * 32 = 7200
model = Chain(
    Conv((3, 3), 3 => 32, relu),
    MaxPool((2, 2)),
    Dropout(0.25),

    flatten,

    Dense(15 * 15 * 32 => 512, relu),
    Dropout(0.5),
    Dense(512 => CLASSES)
    # We omit softmax here because logitcrossentropy is more numerically stable
    )

# --- 4. Training Setup ---
# Using RMSProp as per your Keras code
opt_state = Flux.setup(RMSProp(), model)

# Accuracy helper
accuracy(x, y) = mean(onecold(model(x), 0:9) .== onecold(y, 0:9))

# --- 5. Training Loop ---
println("Starting Training (5 Epochs)...")

@time for epoch in 1:EPOCHS
    # Standard training loop using Flux.train! logic
    for (batch_x, batch_y) in train_loader
        grads = Flux.gradient(m -> logitcrossentropy(m(batch_x), batch_y), model)
        Flux.update!(opt_state, model, grads[1])
    end

    # Validation
    train_acc = accuracy(x_train, y_train_oh)
    val_acc   = accuracy(x_val, y_val_oh)

    @printf("Epoch %d/%d: Train Acc: %.2f%% | Val Acc: %.2f%%\n",
            epoch, EPOCHS, train_acc * 100, val_acc * 100)
end


# --- 6. Final Evaluation ---
test_data = MLDatasets.CIFAR10(split=:test)
x_test, y_test = test_data[:]
x_test = Float32.(x_test)
y_test_oh = onehotbatch(y_test, 0:9)

final_acc = accuracy(x_test, y_test_oh)
println("\nFinal Test Accuracy: ", round(final_acc * 100, digits=2), "%")

