
#time 1.5 s per epoch
using Flux
using CUDA
using Statistics
using Dates

# 1. Configuration & Constants
const VOCAB_SIZE = 10000
const EMBEDDING_DIM = 100
const N_UNITS = 128
const MAX_LEN = 200
const BATCH_SIZE = 32
const EPOCHS = 5

# Set the compute device
device = gpu

# 2. Define the Model Architecture
# We use f32 to ensure Float32 precision (required for efficient GPU usage)
model = Chain(
    Embedding(VOCAB_SIZE => EMBEDDING_DIM),
    LSTM(EMBEDDING_DIM => N_UNITS),
    Dense(N_UNITS => VOCAB_SIZE)
    ) |> f32 |> device

# 3. Data Preparation (Demonstration with dummy data)
# Flux LSTMs expect a Vector of Matrices: [Matrix(features, batch) for t in timesteps]
function get_dummy_data()
    # Indices must be Int (to index Embedding), but the labels for loss will be OHE
    x_raw = [rand(1:VOCAB_SIZE, BATCH_SIZE) for _ in 1:MAX_LEN]
        y_raw = rand(1:VOCAB_SIZE, BATCH_SIZE)

        # Move x to device (indices are kept as Int on GPU)
        x_gpu = [xt |> device for xt in x_raw]

            # CRITICAL: onehotbatch must be moved to device to avoid Ptr errors
            y_gpu = Flux.onehotbatch(y_raw, 1:VOCAB_SIZE) |> device

            return x_gpu, y_gpu
        end

        x_train, y_train = get_dummy_data()

        # 4. Loss and Optimization
        opt_state = Flux.setup(Adam(0.001), model)

        function loss_func(m, x, y)
            # 1. Reset LSTM hidden state for a new sequence
            Flux.reset!(m)

            # 2. Sequential forward pass
            # (Applying model to each timestep. Flux handles the state update internally)
            out = nothing
            for t in x
                out = m(t)
            end

            # 3. Calculate Loss (logitcrossentropy is more stable for Float32)
            return Flux.logitcrossentropy(out, y)
        end

        # 5. Training Loop with Timing
        println("Starting Training on: ", CUDA.name(CUDA.device()))

        # Warm-up (Compiles the gradient code to ensure accurate timing later)
        loss_func(model, x_train, y_train)

        for epoch in 1:EPOCHS
            local val
            # CUDA.@elapsed provides the most accurate timing for GPU operations
            time_taken = CUDA.@elapsed begin
                val, grads = Flux.withgradient(model) do m
                    loss_func(m, x_train, y_train)
                end
                Flux.update!(opt_state, model, grads[1])
            end

            println("Epoch $epoch | Loss: $(round(val, digits=4)) | Time: $(round(time_taken, digits=4))s")
        end

