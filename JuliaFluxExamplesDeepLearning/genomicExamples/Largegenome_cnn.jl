using HTTP, Flux, Statistics, Random, LinearAlgebra, GLMakie, Printf
using Flux: onehotbatch, DataLoader, SamePad

# --- CONSTANTS ---
const ALPHABET = ['a', 'g', 'c', 't']
const SEQ_LEN = 81 # RegulonDB sequences are typically longer (set to 81 for -60 to +20)
const TIMESTEPS = 100
const β = Float32.(range(1e-4, 0.02, length=TIMESTEPS))
const α = 1 .- β
const α_bar = cumprod(α)

# --- 1. DATA PIPELINE (LARGE DATASET) ---
function download_large_promoters()
    println("Fetching RegulonDB large promoter set...")
    # This URL points to a larger curated set of E. coli sigma-70 promoters
    url = "https://raw.githubusercontent.com/mdunne/promoter-sequences/master/ecoli_promoters_60_20.fasta"
    
    response = HTTP.get(url)
    lines = split(strip(String(response.body)), "\n")
    
    X, real_strings = [], String[]
    current_seq = ""
    
    for line in lines
        if startswith(line, ">")
            if length(current_seq) == SEQ_LEN
                push!(X, Float32.(onehotbatch(collect(current_seq), ALPHABET)))
                push!(real_strings, current_seq)
            end
            current_seq = ""
        else
            current_seq *= lowercase(strip(line))
        end
    end
    
    data = cat(X..., dims=3)
    println("Dataset loaded: $(size(data, 3)) sequences.")
    return data, real_strings
end

# --- 2. RESIDUAL ARCHITECTURE ---
struct ResBlock
    layers::Chain
end
Flux.@functor ResBlock
(m::ResBlock)(x) = x .+ m.layers(x)

function build_diffusion_model()
    return Chain(
        # Entry flow: 4 bases + 1 time channel = 5
        Conv((7,), 5 => 64, relu, pad=SamePad()),
        
        # Residual Tower (Allows learning deeper features of larger datasets)
        ResBlock(Chain(Conv((3,), 64 => 64, relu, pad=SamePad()), Conv((3,), 64 => 64, pad=SamePad()))),
        ResBlock(Chain(Conv((3,), 64 => 64, relu, pad=SamePad()), Conv((3,), 64 => 64, pad=SamePad()))),
        ResBlock(Chain(Conv((3,), 64 => 64, relu, pad=SamePad()), Conv((3,), 64 => 64, pad=SamePad()))),
        
        # Exit flow
        Conv((3,), 64 => 32, relu, pad=SamePad()),
        Conv((1,), 32 => 4, pad=SamePad())
    )
end

# --- 3. DIFFUSION CORE ---
function add_noise(x_0, t)
    ϵ = randn(Float32, size(x_0))
    return sqrt(α_bar[t]) .* x_0 .+ sqrt(1 - α_bar[t]) .* ϵ, ϵ
end

function sample_dna(model)
    x_t = randn(Float32, SEQ_LEN, 4, 1)
    for t in TIMESTEPS:-1:1
        t_chan = fill(Float32(t/TIMESTEPS), SEQ_LEN, 1, 1)
        ϵ_pred = model(cat(x_t, t_chan, dims=2))
        
        c1, c2 = 1/sqrt(α[t]), β[t]/sqrt(1-α_bar[t])
        x_t = c1 .* (x_t .- c2 .* ϵ_pred)
        if t > 1 x_t .+= sqrt(β[t]) .* randn(Float32, size(x_t)) end
    end
    return join(ALPHABET[[argmax(x_t[i, :, 1]) for i in 1:SEQ_LEN]])
end

# --- 4. EVALUATION ---
function get_kmer_freq(seqs)
    counts = Dict{String, Float32}()
    for s in seqs, i in 1:(length(s)-1)
        k = s[i:i+1]; counts[k] = get(counts, k, 0f0) + 1f0
    end
    total = sum(values(counts))
    return Dict(k => v/total for (k,v) in counts)
end

function run_diagnostics(real_seqs, model, X_raw)
    println("\n" * "═"^50)
    println("             LARGE DATASET DIAGNOSTICS")
    println("═"^50)
    
    # K-mer Similarity
    synth = [sample_dna(model) for _ in 1:100]
    r_km, s_km = get_kmer_freq(real_seqs), get_kmer_freq(synth)
    all_k = unique(vcat(keys(r_km)..., keys(s_km)...))
    v_r, v_s = [get(r_km, k, 0f0) for k in all_k], [get(s_km, k, 0f0) for k in all_k]
    k_sim = dot(v_r, v_s) / (norm(v_r) * norm(v_s))

    @printf("K-mer Cosine Similarity: %.4f\n", k_sim)
    @printf("Unique Sequences Rate:   %.1f%%\n", (count(s -> !(s in real_seqs), synth)/100)*100)
    
    # Print a few samples
    println("\nSample Generated Sequences:")
    for i in 1:3 println("  >Synth_$i: $(synth[i][1:40])...") end
    println("═"^50)
end

# --- 5. MAIN ---
function main()
    Random.seed!(42)
    X_raw, real_seqs = download_large_promoters()
    model = build_diffusion_model()
    opt = Flux.setup(Adam(0.0005), model) # Lower LR for larger data
    loader = DataLoader(X_raw, batchsize=32, shuffle=true)

    println("Starting Training (200 Epochs)...")
    for epoch in 1:200
        avg_loss = 0.0
        for x_0 in loader
            t = rand(1:TIMESTEPS)
            x_t, ϵ_target = add_noise(x_0, t)
            t_chan = fill(Float32(t/TIMESTEPS), SEQ_LEN, 1, size(x_0, 3))
            
            grads = Flux.gradient(model) do m
                ϵ_pred = m(cat(x_t, t_chan, dims=2))
                sum((ϵ_target .- ϵ_pred).^2) / size(x_0, 3)
            end
            Flux.update!(opt, model, grads[1])
        end
        if epoch % 20 == 0 println("Epoch $epoch/200 complete") end
    end

    run_diagnostics(real_seqs, model, X_raw)
end

main()

