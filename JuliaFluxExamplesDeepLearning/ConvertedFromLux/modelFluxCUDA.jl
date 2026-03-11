using Flux
using Random
using Statistics
using CUDA # Ensure this is installed: Pkg.add("CUDA")

@Base.kwdef mutable struct Args
    seed::Int            = 72
    ϕ::Vector{Float32}   = [0.3f0, 0.2f0, -0.5f0]
    proclen::Int         = 750
    # Use 'gpu' if available, otherwise fallback to 'cpu'
    dev                  = CUDA.functional() ? gpu : cpu 
    opt                  = Flux.Adam 
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
    layers = []
    push!(layers, args.layer(1 => args.hidden_nodes))
    for _ in 1:args.hidden_layers-1
        push!(layers, args.layer(args.hidden_nodes => args.hidden_nodes))
    end
    push!(layers, Dense(args.hidden_nodes => 1))

    # Move model parameters to device (GPU/CPU)
    return Chain(layers...) |> args.dev
end

function compute_loss(model, x_seq, y_seq)
    Flux.reset!(model)
    # Mapping the model over the sequence
    preds = [model(xᵢ) for xᵢ in x_seq]
    
    # Vectorizing the loss prevents the "abs2(Matrix)" error 
    # and is significantly faster on GPU.
    return Flux.Losses.mse(flatten_sequence(preds), flatten_sequence(y_seq))
end

# Helper to concatenate sequence into a single matrix for the loss function
flatten_sequence(seq) = reduce(hcat, seq)

function train_model(args)
    Random.seed!(args.seed)
    model = build_model(args)

    # Generate and move data to the device
    # Note: data must be on the same device as the model!
    Xtrain = [randn(Float32, 1, 1) for _ in 1:args.seqlen] |> args.dev
    ytrain = [randn(Float32, 1, 1) for _ in 1:args.seqlen] |> args.dev

    optim = Flux.setup(args.opt(args.η), model)

    device_name = CUDA.functional() ? "GPU" : "CPU"
    println("Starting training on $device_name for $(args.epochs) epochs...")

    @time begin
        for i in 1:args.epochs
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
println("--- First Run (Compilation + Training) ---")
m = train_model(args)

println("\n--- Second Run (Warmed up) ---")
m_final = train_model(args)

