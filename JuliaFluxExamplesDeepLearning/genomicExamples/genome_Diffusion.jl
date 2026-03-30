using HTTP, Flux, Statistics, MLUtils, Random, GLMakie, Printf
using Flux: onehotbatch, DataLoader, onecold, logitcrossentropy, reset!, crossentropy
using ChainRulesCore: @ignore_derivatives

# --- CONSTANTS ---
const ALPHABET = ['a','g','c','t']
const SEQ_LEN = 57
const DIFFUSION_STEPS = 100

# ------------------------------------------------------------
# 1. DATA LOADING & ORACLE PREPARATION
# ------------------------------------------------------------
function load_data()
    println("Fetching UCI Promoter Dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data"
    raw_lines = split(strip(String(HTTP.get(url).body)), "\n")

    X_prom, X_all, Y_all = [], [], []
    for line in raw_lines
        parts = split(line, ",")
        if length(parts) < 3 continue end
        label = strip(parts[1]) == "+" ? 1 : 2
        seq = lowercase(replace(strip(parts[3]), r"\s+" => ""))
        if length(seq) != 57 continue end

        encoded = Float32.(onehotbatch(collect(seq), ALPHABET))
        push!(X_all, encoded)
        push!(Y_all, Flux.onehot(label, 1:2))
        if label == 1 push!(X_prom, encoded) end
    end
    return cat(X_prom..., dims=3), cat(X_all..., dims=3), cat(Y_all..., dims=2)
end

# ------------------------------------------------------------
# 2. MODELS (Oracle Classifier & Denoiser)
# ------------------------------------------------------------
# Oracle to judge synthetic quality
function build_oracle()
    return Chain(LSTM(4 => 64), x -> x[:, end, :], Dense(64, 2), softmax)
end

# Diffusion Denoiser
struct BidirDenoiser; gru_f; gru_b; head; end
Flux.@functor BidirDenoiser
function (m::BidirDenoiser)(x, t)
    t_emb = fill(Float32(t/DIFFUSION_STEPS), 1, size(x, 2), size(x, 3))
    x_i = vcat(x, t_emb)
    f, b = m.gru_f(x_i), m.gru_b(reverse(x_i, dims=2))
    return m.head(vcat(f, reverse(b, dims=2)))
end

function build_denoiser()
    h = 64
    BidirDenoiser(GRU(5 => h), GRU(5 => h),
                  Chain(Dense(2h, 32, relu), Dense(32, 4), x -> softmax(x, dims=1)))
end

# ------------------------------------------------------------
# 3. PERFORMANCE METRICS
# ------------------------------------------------------------
function calculate_motif_score(seq::String)
    # Simplified search for -10 (TATAAT) and -35 (TTGACA) with 1 mismatch allowed
    tata = occursin(r"tataat|tataaa|tataac", seq) ? 0.5 : 0.0
    ttgaca = occursin(r"ttgaca|ttgact|ttgata", seq) ? 0.5 : 0.0
    return tata + ttgaca
end

# ------------------------------------------------------------
# 4. MAIN EXECUTION
# ------------------------------------------------------------
function main()
    Random.seed!(42)
    X_prom, X_all, Y_all = load_data()

    # Step A: Train Oracle
    oracle = build_oracle()
    opt_o = Flux.setup(Adam(0.001), oracle)
    println("Training Oracle Judge...")
    for _ in 1:100, (x, y) in DataLoader((X_all, Y_all), batchsize=16, shuffle=true)
        reset!(oracle)
        Flux.update!(opt_o, oracle, Flux.gradient(m -> crossentropy(m(x), y), oracle)[1])
    end

    # Step B: Train Diffusion
    denoiser = build_denoiser()
    opt_d = Flux.setup(Adam(0.001), denoiser)
    loader = DataLoader(X_prom, batchsize=16, shuffle=true)

    loss_hist, prec_hist = Float32[], Float32[]

    println("\n--- Training DNA Diffusion ---")
    for epoch in 1:600
        e_loss = 0.0f0
        for x in loader
            t = rand(1:DIFFUSION_STEPS)
            x_noisy = x .* (1f0 - t/DIFFUSION_STEPS) .+ (0.25f0 * t/DIFFUSION_STEPS)
            grads = Flux.gradient(denoiser) do m
                reset!(m); pred = m(x_noisy, t)
                l = mean((pred .- x).^2); @ignore_derivatives e_loss += l; l
            end
            Flux.update!(opt_d, denoiser, grads[1])
        end

        if epoch % 50 == 0
            # Evaluate: Generate 20 sequences and check with Oracle
            reset!(denoiser); sample = fill(0.25f32, 4, 57, 20)
            for t in reverse(1:DIFFUSION_STEPS); sample = denoiser(sample, t); end

            reset!(oracle)
            preds = onecold(oracle(sample))
            precision = mean(preds .== 1) # Label 1 = Promoter

            push!(loss_hist, e_loss/length(loader))
            push!(prec_hist, precision * 100)
            @printf("Epoch %d | Loss: %.6f | Oracle Precision: %.1f%%\n", epoch, loss_hist[end], precision*100)
        end
    end

    # Final Generation & Motif Check
    reset!(denoiser); final_probs = fill(0.25f32, 4, 57, 1)
    for t in reverse(1:DIFFUSION_STEPS); final_probs = denoiser(final_probs, t); end
    final_seq = join(ALPHABET[[argmax(final_probs[:, i, 1]) for i in 1:57]])

        m_score = calculate_motif_score(final_seq)
        println("\nGenerated: ", final_seq)
        println("Motif Score: ", m_score * 100, "% Match")

        # ------------------------------------------------------------
        # 5. VISUALIZATION
        # ------------------------------------------------------------
        fig = Figure(size = (1200, 800))
        ax1 = Axis(fig[1, 1], title="MSE Training Loss", xlabel="Step", ylabel="Loss")
        lines!(ax1, loss_hist, color=:red, linewidth=2)

        ax2 = Axis(fig[1, 2], title="Oracle Prediction Accuracy (%)", xlabel="Step", ylabel="Precision")
        lines!(ax2, prec_hist, color=:blue, linewidth=2)

        ax3 = Axis(fig[2, 1:2], title="Synthetic Sequence Probability Heatmap",
                   xticks=(1:57, string.(1:57)), yticks=(1:4, ["A","G","C","T"]))
        heatmap!(ax3, 1:57, 1:4, final_probs[:, :, 1]', colormap=:magma)

        save("diffusion_performance.png", fig)
        display(fig)
    end

    @time main()

