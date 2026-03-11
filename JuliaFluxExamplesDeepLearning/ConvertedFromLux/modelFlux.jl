# time Flux: 2.8 sec
# time  TensorFlow: 15.23 sec

using Flux
using Random
using Statistics
# Mocking utils.jl functions since they aren't provided,
# assuming they return standard arrays/vectors.
# include("utils.jl")

@Base.kwdef mutable struct Args
    seed::Int            = 72
    ϕ::Vector{Float32}   = [0.3f0, 0.2f0, -0.5f0]
    proclen::Int         = 750
    dev                  = cpu # or gpu
    opt                  = Flux.Adam # Updated to modern Flux naming
    η::Float64           = 2e-3
    hidden_nodes::Int    = 64
    hidden_layers::Int   = 2
    layer                = LSTM
    epochs::Int          = 10
    seqlen::Int          = 1000
    seqshift::Int        = 10
    train_ratio::Float64 = 0.7
    verbose::Bool        = true
end

function build_model(args)
    # Using a list of layers inside Chain
    layers = []
    push!(layers, args.layer(1 => args.hidden_nodes))
    for _ in 1:args.hidden_layers-1
        push!(layers, args.layer(args.hidden_nodes => args.hidden_nodes))
    end
    push!(layers, Dense(args.hidden_nodes => 1))

    return Chain(layers...) |> args.dev
end

# Enhanced loss function for sequence data
function compute_loss(model, x_seq, y_seq)
    Flux.reset!(model)
    # 1. Generate predictions for the whole sequence
    # 2. Use the '.' operator to ensure MSE is calculated element-wise across the collection
    preds = [model(xᵢ) for xᵢ in x_seq]

        # Use Flux.Losses.mse on the combined vectors or broadcast it
        return mean(Flux.Losses.mse.(preds, y_seq))

    end

    function train_model(args)
        Random.seed!(args.seed)
        model = build_model(args)

        # Placeholder for generate_train_test_data logic
        # Replace with your actual data loading from utils.jl
        Xtrain = [randn(Float32, 1, 1) for _ in 1:args.seqlen]
            ytrain = [randn(Float32, 1, 1) for _ in 1:args.seqlen]
                Xtest, ytest = Xtrain, ytrain

                # Modern Flux setup
                optim = Flux.setup(args.opt(args.η), model)

                println("Starting training for $(args.epochs) epochs...")

                    # Timing the loop
                    @time begin
                        for i in 1:args.epochs
                            # Gradient calculation
                            val, grads = Flux.withgradient(model) do m
                                compute_loss(m, Xtrain, ytrain)
                            end

                            Flux.update!(optim, model, grads[1])

                            if args.verbose 
                                @info "Epoch $i, Loss: $(round(val, digits=5))"
                            end
                        end
                    end
                    return model
                end

                # Execution
                args = Args()

                # First call: Includes compilation time
                println("--- First Run (Compilation + Training) ---")
                m = train_model(args)

                # Second call: Pure execution time
                println("\n--- Second Run (Warmed up) ---")
                m_final = train_model(args)

