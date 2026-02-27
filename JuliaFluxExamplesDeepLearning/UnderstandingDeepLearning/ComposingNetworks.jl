using Flux
using Plots
using Statistics

# ---------------------------------------------------------
# 1. Define the Shallow Network Structure
# ---------------------------------------------------------
# In Julia, we can define a function that builds a 1-3-1 network
# using the specific parameters provided in the notebook.

function build_shallow_net(θ_bias, θ_weights, ϕ_bias, ϕ_weights)
    return Chain(
        Dense(θ_weights, θ_bias, relu), # Hidden layer (3 units)
        Dense(ϕ_weights, ϕ_bias)       # Output layer (1 unit)
        )
end

# ---------------------------------------------------------
# 2. Define Parameters for Net 1 and Net 2
# ---------------------------------------------------------
# Data from the notebook:
# Net 1 Parameters
n1_θ_bias = Float32[0.0, 0.0, -0.67]
n1_θ_weights = Float32[-1.0; 1.0; 1.0;;] # 3x1 matrix
n1_ϕ_bias = Float32[1.0]
n1_ϕ_weights = Float32[-2.0 -3.0 9.3]    # 1x3 matrix

# Net 2 Parameters
n2_θ_bias = Float32[-0.6, 0.2, -0.5]
n2_θ_weights = Float32[-1.0; 1.0; 1.0;;]
n2_ϕ_bias = Float32[0.5]
n2_ϕ_weights = Float32[-1.0 -1.5 2.0]

# Instantiate the models
net1 = build_shallow_net(n1_θ_bias, n1_θ_weights, n1_ϕ_bias, n1_ϕ_weights)
net2 = build_shallow_net(n2_θ_bias, n2_θ_weights, n2_ϕ_bias, n2_ϕ_weights)

# ---------------------------------------------------------
# 3. Composing the Networks
# ---------------------------------------------------------
# Define input range [-1, 1]
x_range = collect(-1:0.001:1)' # 1x2001 matrix

# Run Net 1
net1_out = net1(x_range)

# Run Net 2
net2_out = net2(x_range)

# Run the Composition (Net 1 -> Net 2)
# This is equivalent to Net2(Net1(x))
net12_out = net2(net1_out)

# ---------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------
# Plot Net 1 and Net 2 separately
p1 = plot(x_range', net1_out', title="Net 1 Output", color=:red, legend=false)
p2 = plot(x_range', net2_out', title="Net 2 Output", color=:blue, legend=false)

# Plot the Composition
p3 = plot(x_range', net12_out', title="Composition (Net 1 -> Net 2)",
          color=:green, lw=2, legend=false)
xlabel!(p3, "Net 1 Input")
ylabel!(p3, "Net 2 Output")

p4 = plot(p1, p2, p3, layout=(2, 2), size=(800, 800))
display(p4)


