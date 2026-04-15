using Flux, Statistics, Random, LinearAlgebra, Printf
using Flux: onehotbatch, DataLoader

# --- CONSTANTS ---
const ALPHABET = ['a', 'g', 'c', 't']
const SEQ_LEN = 81 
const TIMESTEPS = 100
const β = Float32.(range(1e-4, 0.02, length=TIMESTEPS))
const α = 1 .- β
const α_bar = cumprod(α)

# --- 1. DATA GENERATOR ---
function generate_data(n_samples=2000)
    X = []
    for _ in 1:n_samples
        seq = [rand(ALPHABET) for _ in 1:SEQ_LEN]
        # Embed Consensus Motifs
        seq[20:25] .= collect("ttgaca") # -35 box
        seq[43:48] .= collect("tataat") # -10 box
        push!(X, Float32.(onehotbatch(seq, ALPHABET))') # Shape: (81, 4)
    end
    return cat(X..., dims=3) # Shape: (81, 4, Batch)
end

# --- 2. GRU DIFFUSION MODEL ---
# We use a Bidirectional approach or a deep GRU to see the whole sequence
struct GRUNet
    rnn::Flux.Recur{GRUCell{Matrix{Float32}, Vector{Float32}}}
    head::Chain
end
Flux.@functor GRUNet

function build_gru_model()
    # Input size: 5 (4 bases + 1 time), Hidden size: 128
    return Chain(
        Dense(5 => 128, relu),
        # Flux GRU expects input as a vector/sequence, 
        # but for Diffusion we often wrap it in a layer that handles the spatial dimension
        GRU(128 => 128), 
        Dense(128 => 4)
    )
end

# Note: Flux.GRU is designed for sequences. 
# For a standard 1D array (81, 5, Batch), we map the GRU across the sequence length.
function model_pass(m, x)
    # x is (81, 5, Batch)
    # We reorder to (5, 81, Batch) then treat as sequence of 81 steps
    x_steps = [x[i, :, :] for i in 1:SEQ_LEN] 
    Flux.reset!(m)
    h = [m(step) for step in x_steps] # Output list of (4, Batch)
    return cat(h..., dims=1) # Returns (324, Batch), needs reshape to (81, 4, Batch)
end

# --- 3. TRAINING & SAMPLING ---
function add_noise(x_0, t)
    ϵ = randn(Float32, size(x_0))
    return sqrt(α_bar[t]) .* x_0 .+ sqrt(1 - α_bar[t]) .* ϵ, ϵ
end

function main()
    Random.seed!(42)
    X_raw = generate_data(1000)
    model = build_gru_model()
    opt = Flux.setup(Adam(0.001), model)
    loader = DataLoader(X_raw, batchsize=32, shuffle=true)

    println("Training GRU Diffusion...")
    for epoch in 1:50
        for x_0 in loader
            t = rand(1:TIMESTEPS)
            x_t, ϵ_target = add_noise(x_0, t)
            t_chan = fill(Float32(t/TIMESTEPS), SEQ_LEN, 1, size(x_0, 3))
            input = cat(x_t, t_chan, dims=2) # (81, 5, Batch)

            grads = Flux.gradient(model) do m
                # Simple sequence processing
                Flux.reset!(m)
                # Unroll GRU over the 81 positions
                preds = [m(input[i, :, :]) for i in 1:SEQ_LEN]
                ϵ_pred = cat(preds..., dims=1) # (4*81, Batch)
                
                # Flatten target to match
                target_flat = reshape(ϵ_target, :, size(x_0, 3))
                mean((target_flat .- ϵ_pred).^2)
            end
            Flux.update!(opt, model, grads[1])
        end
        epoch % 10 == 0 && println("Epoch $epoch complete")
    end
end

main()


