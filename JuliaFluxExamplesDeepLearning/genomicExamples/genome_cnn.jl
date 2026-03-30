using HTTP, Flux, Statistics, Random, LinearAlgebra, GLMakie, Printf
using Flux: onehotbatch, DataLoader, onecold

# --- CONSTANTS ---
const ALPHABET = ['a', 'g', 'c', 't']
const SEQ_LEN = 57
const TIMESTEPS = 100
const β = Float32.(range(1e-4, 0.02, length=TIMESTEPS))
const α = 1 .- β
const α_bar = cumprod(α)

# --- 1. DATA & MODEL ---
function download_and_preprocess()
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data"
    raw_lines = split(strip(String(HTTP.get(url).body)), "\n")
    X, real_strings = [], String[]
    for line in raw_lines
        parts = split(line, ",")
        if length(parts) < 3 || strip(parts[1]) != "+" continue end
        seq = lowercase(replace(strip(parts[3]), r"\s+" => ""))
        if length(seq) == 57
            push!(X, collect(Float32.(onehotbatch(collect(seq), ALPHABET))'))
            push!(real_strings, seq)
        end
    end
    return cat(X..., dims=3), real_strings
end

function build_diffusion_model()
    return Chain(
        Conv((7,), 5 => 32, relu, pad=SamePad()),
        Conv((3,), 32 => 64, relu, pad=SamePad()),
        Conv((3,), 64 => 32, relu, pad=SamePad()),
        Conv((3,), 32 => 4, pad=SamePad())
        )
end

# --- 2. EVALUATION & SALIENCY ---
function get_saliency_map(model, x_0)
    t = rand(1:TIMESTEPS)
    x_t, ϵ_target = add_noise(x_0, t)
    t_chan = fill(Float32(t/TIMESTEPS), 57, 1, 1)

    # Backpropagate to the input x_t to see importance
    grads = Flux.gradient(x_t) do xt
        ϵ_pred = model(cat(xt, t_chan, dims=2))
        sum((ϵ_target .- ϵ_pred).^2)
    end
    return abs.(grads[1][:, :, 1])
end

function get_kmer_freq(seqs)
    counts = Dict{String, Float32}()
    for s in seqs, i in 1:(length(s)-1)
        k = s[i:i+1]; counts[k] = get(counts, k, 0f0) + 1f0
    end
    total = sum(values(counts))
    return Dict(k => v/total for (k,v) in counts)
    end

    function print_performance_report(real_seqs, model, X_raw)
        println("\n" * "="^50)
        println("          DIFFUSION PERFORMANCE METRICS")
        println("="^50)

        # Denoising Quality (MSE)
        t_test = rand(1:TIMESTEPS)
        x_t, ϵ_target = add_noise(X_raw, t_test)
        t_chan = fill(Float32(t_test/TIMESTEPS), 57, 1, size(X_raw, 3))
        ϵ_pred = model(cat(x_t, t_chan, dims=2))
        mse = mean((ϵ_target .- ϵ_pred).^2)

        # Generation Quality
        synth_seqs = [sample_dna(model) for _ in 1:50]
            calc_gc(s) = count(c -> c == 'g' || c == 'c', s) / length(s)
            real_gc = mean(calc_gc.(real_seqs))
            synth_gc = mean(calc_gc.(synth_seqs))

            # K-mer Similarity (Cosine)
            r_kmers = get_kmer_freq(real_seqs)
            s_kmers = get_kmer_freq(synth_seqs)
            all_k = unique(vcat(collect(keys(r_kmers)), collect(keys(s_kmers))))
            v_r, v_s = [get(r_kmers, k, 0f0) for k in all_k], [get(s_kmers, k, 0f0) for k in all_k]
                k_sim = dot(v_r, v_s) / (norm(v_r) * norm(v_s))

                @printf("1. Denoising MSE:           %.6f\n", mse)
                @printf("2. Real Mean GC:            %.2f%%\n", real_gc * 100)
                @printf("3. Synth Mean GC:           %.2f%%\n", synth_gc * 100)
                @printf("4. K-mer Similarity:        %.4f (Target: 1.0)\n", k_sim)
                @printf("5. Novelty Rate:            %.1f%% unique\n",
                        (count(s -> !(s in real_seqs), synth_seqs)/50)*100)
                println("="^50)
            end

            # --- 3. VISUALIZATION ---
            function create_visualizations(model, real_seqs, X_raw)
                println("Generating Visualizations...")

                # Heatmap: Saliency
                sample_idx = rand(1:size(X_raw, 3))
                saliency = get_saliency_map(model, X_raw[:, :, sample_idx:sample_idx])
                fig1 = Figure(resolution = (1000, 450))
                ax1 = Axis(fig1[1, 1], title="Saliency Map (Base Importance)",
                           xlabel="Sequence Position", ylabel="Base",
                           xticks=(1:5:57), yticks=(1:4, ["A","G","C","T"]))
                hm1 = heatmap!(ax1, 1:57, 1:4, saliency', colormap=:viridis)
                Colorbar(fig1[1, 2], hm1)
                save("saliency_mapGAN.png", fig1)

                # Barplot: K-mer Distribution
                synth_seqs = [sample_dna(model) for _ in 1:50]
                    r_km = get_kmer_freq(real_seqs); s_km = get_kmer_freq(synth_seqs)
                    keys_sorted = sort(collect(keys(r_km)))

                    fig2 = Figure(resolution = (800, 400))
                    ax2 = Axis(fig2[1, 1], title="K-mer Frequency Comparison", xticks=(1:16, keys_sorted))
                    barplot!(ax2, 1:16, [get(r_km, k, 0f0) for k in keys_sorted], color=(:blue, 0.4), label="Real")
                        barplot!(ax2, 1:16, [get(s_km, k, 0f0) for k in keys_sorted], color=(:red, 0.4), label="Synth")
                            axislegend()
                            save("kmer_comparisonGAN.png", fig2)

                            println("Files saved: saliency_map.png, kmer_comparison.png")
                        end

                        # --- 4. DIFFUSION LOGIC ---
                        function add_noise(x_0, t)
                            ϵ = randn(Float32, size(x_0))
                            return sqrt(α_bar[t]) .* x_0 .+ sqrt(1 - α_bar[t]) .* ϵ, ϵ
                        end

                        function sample_dna(model)
                            x_t = randn(Float32, 57, 4, 1)
                            for t in TIMESTEPS:-1:1
                                t_chan = fill(Float32(t/TIMESTEPS), 57, 1, 1)
                                ϵ_pred = model(cat(x_t, t_chan, dims=2))
                                c1, c2 = 1/sqrt(α[t]), β[t]/sqrt(1-α_bar[t])
                                x_t = c1 .* (x_t .- c2 .* ϵ_pred)
                                if t > 1 x_t .+= sqrt(β[t]) .* randn(Float32, size(x_t)) end
                            end
                            return join(ALPHABET[[argmax(x_t[i, :, 1]) for i in 1:SEQ_LEN]])
                            end

                            # --- 5. MAIN ---
                            function main()
                                Random.seed!(42)
                                X_raw, real_seqs = download_and_preprocess()
                                model = build_diffusion_model()
                                opt = Flux.setup(Adam(0.001), model)
                                loader = DataLoader(X_raw, batchsize=16, shuffle=true)

                                println("Training...")
                                for epoch in 1:1000
                                    for x_0 in loader
                                        t = rand(1:TIMESTEPS)
                                        x_t, ϵ_target = add_noise(x_0, t)
                                        t_chan = fill(Float32(t/TIMESTEPS), 57, 1, size(x_0, 3))
                                        grads = Flux.gradient(model) do m
                                            ϵ_pred = m(cat(x_t, t_chan, dims=2))
                                            mean((ϵ_target .- ϵ_pred).^2)
                                        end
                                        Flux.update!(opt, model, grads[1])
                                    end
                                    if epoch % 200 == 0 println("Epoch $epoch/1000") end
                                end

                                print_performance_report(real_seqs, model, X_raw)
                                create_visualizations(model, real_seqs, X_raw)
                            end

                            main()

