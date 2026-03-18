module JTensorFlow

using Flux
using Optimisers
using Statistics

# --- 1. The Layers Interface ---
struct Layers end
const layers = Layers()

function Base.getproperty(::Layers, s::Symbol)
    if s === :Dense
        # Now accepts (in => out) or just (out) if you handle the 'in' manually
        return (in_out; activation=identity) -> Flux.Dense(in_out, activation)
        elseif s === :Conv2D
        return (filters::Int, kernel_size::Tuple{Int, Int}; strides=(1, 1), padding="same", activation=identity) -> begin
            pad = padding == "same" ? Flux.SamePad() : 0
            (in_ch) -> Flux.Conv(kernel_size, in_ch => filters, activation; stride=strides, pad=pad)
        end
        elseif s === :Flatten
        return () -> Flux.flatten
    else
        return getfield(JTensorFlow, s)
    end
end

# --- 2. The Model Interface ---
mutable struct Model
    chain::Chain
    optimizer_state
    loss_fn

    Model(args...) = new(Chain(args...), nothing, nothing)
end

# Helper to find the output shape of a partial network
function get_flatten_size(layers_list, input_shape)
    # input_shape should be (W, H, C, Batch)
    dummy_input = rand(Float32, input_shape...)
    temp_chain = Chain(layers_list...)
    return size(temp_chain(dummy_input), 1)
end

function compile!(m::Model; optimizer="adam", loss="mse")
    opt_map = Dict(
        "adam" => Optimisers.Adam(0.001),
        "rmsprop" => Optimisers.RMSProp(),
        "sgd" => Optimisers.Descent(0.01)
        )
    loss_map = Dict(
        "mse" => Flux.mse,
        "categorical_crossentropy" => Flux.logitcrossentropy
        )

    m.loss_fn = loss_map[loss]
    m.optimizer_state = Optimisers.setup(opt_map[optimizer], m.chain)
end

function fit!(m::Model, x, y; epochs=1)
    for epoch in 1:epochs
        val, grads = Flux.withgradient(m.chain) do model
            m.loss_fn(model(x), y)
        end
        m.optimizer_state, m.chain = Optimisers.update!(m.optimizer_state, m.chain, grads[1])
        println("Epoch $epoch: Loss = $(round(val, digits=4))")
    end
end

(m::Model)(x) = m.chain(x)

export layers, Model, compile!, fit!, get_flatten_size
end

# --- 3. RUNNABLE EXECUTION ---
using .JTensorFlow
using Flux: relu, onehotbatch

# Define the input resolution
input_dim = (28, 28, 1, 1)

# Step A: Define the convolutional part
conv_part = [
    JTensorFlow.layers.Conv2D(16, (3, 3), activation=relu)(1),
    JTensorFlow.layers.Flatten()
    ]

# Step B: Automatically calculate the input size for the Dense layer
# This prevents the DimensionMismatch error!
flat_size = JTensorFlow.get_flatten_size(conv_part, input_dim)
println("Detected Flatten size: $flat_size")

# Step C: Build the final model
model = JTensorFlow.Model(
    conv_part...,
    JTensorFlow.layers.Dense(flat_size => 10)
    )

# Step D: Standard Workflow
JTensorFlow.compile!(model, optimizer="adam", loss="categorical_crossentropy")

x_train = rand(Float32, 28, 28, 1, 10)
y_train = onehotbatch(rand(1:10, 10), 1:10)

JTensorFlow.fit!(model, x_train, y_train, epochs=3)


