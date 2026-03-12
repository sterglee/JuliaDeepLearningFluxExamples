
#Real Mean GC: 42.14%
#Synth Mean GC: 41.05%
# 29.683883 seconds (62.13 M allocations: 35.484 GiB, 6.25% gc time, 38.10% compilation time: 9% of which was recompilation)

 using HTTP, Flux, Statistics, Random, LinearAlgebra
using Flux: onehotbatch, DataLoader, onecold

# ------------------------------------------------------------
# 1. CONSTANTS & NOISE SCHEDULE
# ------------------------------------------------------------
const ALPHABET = ['a', 'g', 'c', 't']
const SEQ_LEN = 57
const TIMESTEPS = 100
# Linear noise schedule
const β = Float32.(range(1e-4, 0.02, length=TIMESTEPS))
const α = 1 .- β
const α_bar = cumprod(α)

# ------------------------------------------------------------
# 2. DATA LOADING
# ------------------------------------------------------------
function download_and_preprocess()
    println("Fetching dataset from UCI...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data"
    response = HTTP.get(url)
    raw_lines = split(strip(String(response.body)), "\n")

    X = []
    real_strings = String[]

    for line in raw_lines
        parts = split(line, ",")
        if length(parts) < 3 || strip(parts[1]) != "+" continue end
        seq = lowercase(replace(strip(parts[3]), r"\s+" => ""))
        if length(seq) != 57 continue end

        # Reshape to (Width=57, Channels=4) for Flux 1D Conv
        encoded = Float32.(onehotbatch(collect(seq), ALPHABET))
        push!(X, collect(encoded')) 
        push!(real_strings, seq)
    end
    return cat(X..., dims=3), real_strings
end

# ------------------------------------------------------------
# 3. MODEL ARCHITECTURE
# ------------------------------------------------------------
function build_diffusion_model()
    return Chain(
        # Input: (57, 5, Batch) -> 4 bases + 1 time channel
        Conv((7,), 5 => 32, relu, pad=SamePad()),
        Conv((3,), 32 => 64, relu, pad=SamePad()),
        Conv((3,), 64 => 32, relu, pad=SamePad()),
        # Output: (57, 4, Batch) -> predicting the noise for the 4 bases
        Conv((3,), 32 => 4, pad=SamePad()) 
    )
end

# ------------------------------------------------------------
# 4. DIFFUSION LOGIC
# ------------------------------------------------------------
function add_noise(x_0, t)
    ϵ = randn(Float32, size(x_0))
    x_t = sqrt(α_bar[t]) .* x_0 .+ sqrt(1 - α_bar[t]) .* ϵ
    return x_t, ϵ
end

function sample_dna(model)
    # Start with pure Gaussian noise: (57, 4, 1)
    x_t = randn(Float32, 57, 4, 1)
    
    for t in TIMESTEPS:-1:1
        # Inject time information
        t_chan = fill(Float32(t/TIMESTEPS), 57, 1, 1)
        input = cat(x_t, t_chan, dims=2)
        
        ϵ_pred = model(input)
        
        # Reverse Step math
        c1 = 1 / sqrt(α[t])
        c2 = β[t] / sqrt(1 - α_bar[t])
        x_t = c1 .* (x_t .- c2 .* ϵ_pred)
        
        # Add posterior noise back for sampling variety
        if t > 1
            x_t .+= sqrt(β[t]) .* randn(Float32, size(x_t))
        end
    end
    
    # Argmax back to DNA string
    indices = [argmax(x_t[i, :, 1]) for i in 1:SEQ_LEN]
    return join(ALPHABET[indices])
end

# ------------------------------------------------------------
# 5. EVALUATION UTILS
# ------------------------------------------------------------
function calculate_mse(model, x_data)
    t = rand(1:TIMESTEPS)
    x_t, ϵ_target = add_noise(x_data, t)
    t_chan = fill(Float32(t/TIMESTEPS), 57, 1, size(x_data, 3))
    ϵ_pred = model(cat(x_t, t_chan, dims=2))
    return mean((ϵ_target .- ϵ_pred).^2)
end

# ------------------------------------------------------------
# 6. MAIN EXECUTION
# ------------------------------------------------------------
function main()
    Random.seed!(42)
    X_raw, real_seqs = download_and_preprocess()
    
    model = build_diffusion_model()
    opt_state = Flux.setup(Adam(0.001), model)
    loader = DataLoader(X_raw, batchsize=16, shuffle=true)

    println("\n--- Training Generative Diffusion Model ---")
    for epoch in 1:1000
        for x_0 in loader
            batch_size = size(x_0, 3)
            t = rand(1:TIMESTEPS)
            x_t, ϵ_target = add_noise(x_0, t)
            
            t_chan = fill(Float32(t/TIMESTEPS), 57, 1, batch_size)
            
            grads = Flux.gradient(model) do m
                ϵ_pred = m(cat(x_t, t_chan, dims=2))
                mean((ϵ_target .- ϵ_pred).^2)
            end
            Flux.update!(opt_state, model, grads[1])
        end
        
        if epoch % 100 == 0
            loss = calculate_mse(model, X_raw)
            println("Epoch $epoch | Denoising MSE: $(round(loss, digits=5))")
        end
    end

    println("\n--- Evaluation & Generation ---")
    println("Generating 3 synthetic promoters:")
    for i in 1:3
        println("Gen $i: ", sample_dna(model))
    end
    
    # Quick GC check
    real_gc = mean([count(c->c=='g'||c=='c', s)/57 for s in real_seqs])
    synth_seqs = [sample_dna(model) for _ in 1:10]
    synth_gc = mean([count(c->c=='g'||c=='c', s)/57 for s in synth_seqs])
    
    println("\nReal Mean GC: $(round(real_gc*100, digits=2))%")
    println("Synth Mean GC: $(round(synth_gc*100, digits=2))%")
end

@time main()

