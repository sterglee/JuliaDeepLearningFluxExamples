using Flux
using Random
using LinearAlgebra

# --- 1. Define the Network Architecture ---
# This replicates the deep network with specific hidden dimensions
# D_i = input, D_1 = 1st hidden, D_2 = 2nd hidden, D_3 = 3rd hidden, D_o = output
function build_deep_network(D_i, D_1, D_2, D_3, D_o)
    return Chain(
        Dense(D_i, D_1, relu),  # Layer 1
        Dense(D_1, D_2, relu),  # Layer 2
        Dense(D_2, D_3, relu),  # Layer 3
        Dense(D_3, D_o)         # Output Layer (Linear)
        )
end

# --- 2. Manual Initialization (Optional) ---
# The notebook often uses specific weights. Here is how to set them manually:
function set_custom_weights!(model)
    # Accessing parameters: layers are model[1], model[2], etc.
    # model[1].weight is Matrix{Float32}, model[1].bias is Vector{Float32}

    # Example: Initialize first layer with specific values
    D_1, D_i = size(model[1].weight)
    model[1].weight .= randn(Float32, D_1, D_i) * 0.1f0
    model[1].bias .= zeros(Float32, D_1)
end

# --- 3. Running Inference ---
# Setup dimensions as per the notebook example
D_i, D_1, D_2, D_3, D_o = 3, 4, 4, 3, 2
n_data = 5

# Generate random input data (Features x Samples)
#
X = randn(Float32, D_i, n_data)

# Create the model
model = build_deep_network(D_i, D_1, D_2, D_3, D_o)

# Pass data through the network
Y = model(X)

println("Input Shape: ", size(X))
println("Output Shape: ", size(Y))
println("\nOutput Data points:")
display(Y)

