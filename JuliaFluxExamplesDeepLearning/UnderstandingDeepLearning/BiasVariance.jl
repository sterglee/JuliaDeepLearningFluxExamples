using Flux
using Statistics
using Plots

# ---------------------------------------------------------
# 1. Define a Deep Network
# ---------------------------------------------------------
# A deep network (e.g., 50 layers) to see the effects of initialization
n_layers = 50
D_h = 100
# Define a Chain with many layers
layers = [Dense(D_h => D_h, relu) for _ in 1:n_layers]
    model = Chain(layers...)

    # ---------------------------------------------------------
    # 2. Custom Initialization Functions
    # ---------------------------------------------------------
    # He Initialization (Standard for ReLU)
    # Variance = 2 / fan_in
    init_he(out, in) = Float32.(randn(out, in) .* sqrt(2.0 / in))

    # Small Initialization (Will lead to vanishing activations)
    init_small(out, in) = Float32.(randn(out, in) .* 0.01)

    # Large Initialization (Will lead to exploding activations)
    init_large(out, in) = Float32.(randn(out, in) .* 1.0)

    # ---------------------------------------------------------
    # 3. Analyze Activation Flow (Forward Pass)
    # ---------------------------------------------------------
    function check_activations(init_func)
        # Create model with specific init
        m = Chain([Dense(D_h => D_h, relu, init=init_func) for _ in 1:n_layers]...)

            x = randn(Float32, D_h, 1)
            stds = []

            current_h = x
            for layer in m
                current_h = layer(current_h)
                push!(stds, std(current_h))
            end
            return stds
        end

        stds_he = check_activations(init_he)
        stds_small = check_activations(init_small)
        stds_large = check_activations(init_large)

        # ---------------------------------------------------------
        # 4. Analyze Gradient Flow (Backward Pass)
        # ---------------------------------------------------------
        # Flux computes these automatically, but we can inspect their magnitude
        m_grad = Chain([Dense(D_h => D_h, relu, init=init_he) for _ in 1:n_layers]...)
            x_in = randn(Float32, D_h, 1)

            val, grads = Flux.withgradient(m_grad) do m
                sum(m(x_in)) # Dummy loss
            end

            grad_stds = [std(grads[1].layers[i].weight) for i in 1:n_layers]

                # ---------------------------------------------------------
                # 5. Visualization
                # ---------------------------------------------------------


                p1 = plot(stds_small, label="Small Init (0.01)", title="Activation Std Dev", yaxis=:log)
                plot!(p1, stds_large, label="Large Init (1.0)")
                plot!(p1, stds_he, label="He Init", lw=2, color=:black)
                ylabel!(p1, "Std Dev of Activations")
                xlabel!(p1, "Layer Depth")

                p2 = plot(grad_stds, title="Gradient Magnitude", label="Weight Gradients", color=:purple)
                ylabel!(p2, "Std Dev of Gradients")
                xlabel!(p2, "Layer Depth")

                display(plot(p1, p2, layout=(1, 2), size=(900, 400)))



