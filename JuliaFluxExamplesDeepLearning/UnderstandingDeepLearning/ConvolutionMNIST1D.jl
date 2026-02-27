using Flux, Statistics, Random, Plots, Pickle

# ==========================================
# 1. Data Loading and Preprocessing
# ==========================================
# Load the MNIST-1D dataset
data = Pickle.load(open("mnist1d_data.pkl"))

# Flux expects 1D convolution data in the shape: (Width, Channels, Batch)
# MNIST-1D template length is 40
x_train_raw = Float32.(data["x"])'          # Transpose to (40, 4000)
x_train = reshape(x_train_raw, 40, 1, :)    # Shape: (40, 1, 4000)

x_val_raw = Float32.(data["x_test"])'
x_val = reshape(x_val_raw, 40, 1, :)        # Shape: (40, 1, 1000)

# Shift labels by 1 for Julia's 1-based indexing (0-9 becomes 1-10)
y_train_labels = data["y"] .+ 1
y_val_labels = data["y_test"] .+ 1

# Convert training labels to one-hot encoding for CrossEntropy
y_train_oh = Flux.onehotbatch(y_train_labels, 1:10)

# ==========================================
# 2. Model Architecture (The "TODO" Task)
# ==========================================
# Architecture requirement from Notebook 10.2:
# - Conv1: Kernel 3, Stride 2, Valid Padding, 15 Channels
# - Conv2: Kernel 3, Stride 2, Valid Padding, 15 Channels
# - Conv3: Kernel 3, Stride 2, Valid Padding, 15 Channels
# - Flatten & Linear to 10 outputs

model = Chain(
    # Layer 1: In=1 channel, Out=15 channels
    Conv((3,), 1 => 15, relu, stride=2, pad=0),

    # Layer 2: In=15, Out=15
    Conv((3,), 15 => 15, relu, stride=2, pad=0),

    # Layer 3: In=15, Out=15
    Conv((3,), 15 => 15, relu, stride=2, pad=0),

    # Flatten converts the spatial output to a vector
    Flux.flatten,

    # Final Linear layer (60 inputs to 10 outputs)
    Dense(60, 10)
    )

# ==========================================
# 3. Training Configuration
# ==========================================
# Define the DataLoader in the Main scope to avoid UndefVarError
train_loader = Flux.DataLoader((x_train, y_train_oh), batchsize=100, shuffle=true)

# Loss function: Logit Cross Entropy
loss_fn(x, y) = Flux.logitcrossentropy(model(x), y)

# Optimizer: SGD with Learning Rate 0.05 and Momentum 0.9
opt_state = Flux.setup(Flux.Momentum(0.05, 0.9), model)

# ==========================================
# 4. Training Loop
# ==========================================
epochs = 100
train_errors = Float32[]
val_errors = Float32[]

println("Starting training for $epochs epochs...")

    for epoch in 1:epochs
        # Training step
        for (batch_x, batch_y) in train_loader
            grads = Flux.gradient(m -> loss_fn(batch_x, batch_y), model)
            Flux.update!(opt_state, model, grads[1])
        end

        # Calculate performance statistics
        # Use onecold to convert model probabilities back to class indices
        train_pred = Flux.onecold(model(x_train))
        val_pred = Flux.onecold(model(x_val))

        train_err = 100 * (1 - mean(train_pred .== y_train_labels))
        val_err = 100 * (1 - mean(val_pred .== y_val_labels))

        push!(train_errors, train_err)
        push!(val_errors, val_err)

        if epoch % 10 == 0 || epoch == 1
            println("Epoch $epoch: Train Error $(round(train_err, digits=2))%, Val Error $(round(val_err, digits=2))%")
        end

        # Note: StepLR equivalent (reducing LR every 20 epochs)
        if epoch % 20 == 0
            new_lr = opt_state[1].eta * 0.5
            Flux.adjust!(opt_state, new_lr)
        end
    end

    # ==========================================
    # 5. Result Visualization
    # ==========================================
    plot(train_errors, label="Train Error", color=:red, lw=2)
    plot!(val_errors, label="Validation Error", color=:blue, lw=2,
          title="MNIST-1D 1D-CNN Performance",
          xlabel="Epoch", ylabel="Error %", ylim=(0, 100))
