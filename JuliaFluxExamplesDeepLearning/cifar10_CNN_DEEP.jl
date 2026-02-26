using Flux
using Flux: DataLoader, onehotbatch, onecold, flatten, logitcrossentropy
using MLDatasets, Statistics, Printf

# --- 1. Constants & Hyperparameters ---
const NUM_CLASSES = 10
const BATCH_SIZE = 64
const EPOCHS = 5

# --- 2. Data Loading & Normalization ---
function load_data()
    # Load CIFAR10 - MLDatasets provides images as Float32 by default
    train_data = MLDatasets.CIFAR10(split=:train)
    test_data  = MLDatasets.CIFAR10(split=:test)

    x_train, y_train = train_data[:]
    x_test,  y_test  = test_data[:]

    # Calculate Mean and Std over all pixels (equivalent to axis=(0,1,2,3))
    # We use 'f' suffix for Float32 literals
    m = mean(x_train)
    s = std(x_train)

    # Normalize using Float32 arithmetic
    x_train = (x_train .- m) ./ (s + 1f-7)
    x_test  = (x_test  .- m) ./ (s + 1f-7)

    # One-hot encode labels into Float32 matrices
    y_train = Float32.(onehotbatch(y_train, 0:9))
    y_test  = Float32.(onehotbatch(y_test, 0:9))

    return x_train, y_train, x_test, y_test
end

x_train, y_train, x_test, y_test = load_data()
train_loader = DataLoader((x_train, y_train), batchsize=BATCH_SIZE, shuffle=true)

# --- 3. Build Model (Modern Chain Syntax) ---
# Image dimensions: 32x32 -> MaxPool -> 16x16 -> MaxPool -> 8x8 -> MaxPool -> 4x4
model = Chain(
    # Block 1
    Conv((3,3), 3=>32, relu, pad=1),
    BatchNorm(32),
    Conv((3,3), 32=>32, relu, pad=1),
    BatchNorm(32),
    MaxPool((2,2)),
    Dropout(0.2f0),

    # Block 2
    Conv((3,3), 32=>64, relu, pad=1),
    BatchNorm(64),
    Conv((3,3), 64=>64, relu, pad=1),
    BatchNorm(64),
    MaxPool((2,2)),
    Dropout(0.3f0),

    # Block 3
    Conv((3,3), 64=>128, relu, pad=1),
    BatchNorm(128),
    Conv((3,3), 128=>128, relu, pad=1),
    BatchNorm(128),
    MaxPool((2,2)),
    Dropout(0.4f0),

    # Dense Output
    flatten,
    Dense(4 * 4 * 128 => NUM_CLASSES)
    ) |> f32  # Ensure all internal weights are Float32

# --- 4. Training Setup ---
# RMSProp optimizer to match Keras 'RMSprop'
opt_state = Flux.setup(RMSProp(), model)

# Define accuracy with mode switching for BatchNorm/Dropout
function get_accuracy(m, x, y)
    Flux.testmode!(m)
    acc = mean(onecold(m(x), 0:9) .== onecold(y, 0:9))
    Flux.trainmode!(m)
    return acc
end

# --- 5. Training Loop using train! ---
println("Starting Training for $EPOCHS epochs...")

    for epoch in 1:EPOCHS
        # Flux.train! takes (loss_function, model, data_iterator, opt_state)
        Flux.train!(model, train_loader, opt_state) do m, x, y
            logitcrossentropy(m(x), y)
        end

        # Progress report
        val_acc = get_accuracy(model, x_test, y_test)
        @printf("Epoch %d: Test Accuracy: %.2f%%\n", epoch, val_acc * 100)
    end

    # --- 6. Final Evaluation ---
    Flux.testmode!(model)
    final_loss = logitcrossentropy(model(x_test), y_test)
    final_acc = get_accuracy(model, x_test, y_test)

    @printf("\nFinal Result -> Loss: %.4f | Accuracy: %.2f%%\n", final_loss, final_acc * 100)

