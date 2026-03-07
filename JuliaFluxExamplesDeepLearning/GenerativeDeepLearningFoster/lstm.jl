
# time: 1.5s per epoch
using Flux
using CUDA
using Statistics
using Dates

# 0. Parameters (From snippet_1)
VOCAB_SIZE = 10000
MAX_LEN = 200
EMBEDDING_DIM = 100
N_UNITS = 128
BATCH_SIZE = 32
EPOCHS = 25

# 1. Define Model
# Moving to GPU and using Float32 as requested
device = gpu
model = Chain(
    Embedding(VOCAB_SIZE => EMBEDDING_DIM),
    LSTM(EMBEDDING_DIM => N_UNITS),
    Dense(N_UNITS => VOCAB_SIZE)
    ) |> f32 |> device

# 2. Dummy Data for Demonstration
# In Julia/Flux, RNN inputs are typically a vector of matrices (one per timestep)
# Each matrix is (features, batch_size)
x_train = [rand(1:VOCAB_SIZE, BATCH_SIZE) for _ in 1:MAX_LEN] |> device
    y_train = rand(1:VOCAB_SIZE, BATCH_SIZE) |> device

    # 3. Loss and Optimizer
    opt_state = Flux.setup(Adam(0.001), model)

    function loss_func(m, x, y)
        # Reset hidden state for each new sequence
        Flux.reset!(m)
        # Pass through the sequence (last output used for prediction)
        # Foldl or a loop is used for sequential processing in Flux
        out = nothing
        for t in x
            out = m(t)
        end
        return Flux.logitcrossentropy(out, Flux.onehotbatch(y, 1:VOCAB_SIZE))
    end

    # 4. Training with Timing
    println("Starting Training on: ", CUDA.name(CUDA.device()))

    # Warm-up (to exclude JIT compilation from timing)
    loss_func(model, x_train, y_train)

    # Actual timed training loop
    for epoch in 1:EPOCHS
        start_time = now()

        val, grads = Flux.withgradient(model) do m
            loss_func(m, x_train, y_train)
        end
        Flux.update!(opt_state, model, grads[1])

        end_time = now()
        elapsed = canonicalize(Dates.CompoundPeriod(end_time - start_time))

        println("Epoch $epoch | Loss: $(round(val, digits=4)) | Time: $elapsed")
    end

