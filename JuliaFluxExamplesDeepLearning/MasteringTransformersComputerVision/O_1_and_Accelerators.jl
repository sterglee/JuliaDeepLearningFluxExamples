using Flux
using LinearAlgebra

# -----------------------------
# Setup parameters
# -----------------------------
const d = 12288
const n_steps = 100
const hidden_size = 512

input_seq = randn(Float32, d, n_steps)

# -----------------------------
# Define the RNN Cell
# -----------------------------
rnn_cell = Flux.RNNCell(d, hidden_size)

# Initial hidden state
initial_state = randn(Float32, hidden_size)

println("--- Running Fixed RNN Sequential Simulation ---")

# -----------------------------
# Simulation Function
# -----------------------------
function simulate_rnn(cell, input, state)
    current_state = state

    @inbounds for t in 1:size(input, 2)
        x_t = view(input, :, t)  # avoids allocation

        # Correct order: (input, state)
        current_state, _ = cell(x_t, current_state)
    end

    return current_state
end

# -----------------------------
# Run + Benchmark
# -----------------------------
rnn_time = @elapsed begin
    final_state = simulate_rnn(rnn_cell, input_seq, initial_state)
end

println("RNN sequential ($n_steps steps) time: $rnn_time seconds")
println("Final state size: ", size(final_state))

