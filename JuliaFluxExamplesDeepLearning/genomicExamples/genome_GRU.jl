using HTTP, Flux, Statistics, MLUtils, Random, GLMakie, Printf
using Flux: onehotbatch, onehot, DataLoader, reset!, onecold, crossentropy
using ChainRulesCore: @ignore_derivatives

# --- CONSTANTS ---
const ALPHABET = ['a','g','c','t']
const SEQ_LEN = 57

# 1. DATA LOADING
function download_and_preprocess()
    println("Fetching dataset from UCI...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data"
    response = HTTP.get(url)
    raw_lines = split(strip(String(response.body)), "\n")

    X, Y = Vector{Array{Float32,2}}(), []

    for line in raw_lines
        parts = split(line, ",")
        if length(parts) < 3 continue end
        label = strip(parts[1]) == "+" ? 1 : 2
        seq = lowercase(replace(strip(parts[3]), r"\s+" => ""))
        if length(seq) != 57 continue end

        push!(X, Float32.(onehotbatch(collect(seq), ALPHABET)))
        push!(Y, onehot(label, 1:2))
    end
    return cat(X..., dims=3), cat(Y..., dims=2)
end

# 2. CUSTOM BIDIRECTIONAL WRAPPER
# This runs the sequence forward and backward and joins the results.
struct Bidir{F, B}
    forward::F
    backward::B
end

Flux.@functor Bidir

function (m::Bidir)(x)
    # Forward pass
    f = m.forward(x)
    # Backward pass: flip sequence along the time dimension (dim 2)
    b = m.backward(reverse(x, dims=2))
    # Concatenate the final states
    return vcat(f[:, end, :], b[:, end, :])
end

Flux.reset!(m::Bidir) = (reset!(m.forward); reset!(m.backward))

# 3. IMPROVED MODEL DEFINITION
function build_improved_gru()
    hidden_dim = 64

    return Chain(
        Bidir(GRU(4 => hidden_dim), GRU(4 => hidden_dim)),
        Dropout(0.4), # Increased dropout for better generalization
        Dense(hidden_dim * 2, 32, relu),
        Dense(32, 2),
        softmax
        )
end

# 4. VISUALIZATION
function plot_results(epochs, losses, accs, y_true, y_pred)
    fig = Figure(size = (1200, 500))

    ax1 = Axis(fig[1, 1], title="Training Progress (Bi-GRU)", xlabel="Epoch", ylabel="Value")
    lines!(ax1, epochs, losses, label="Loss", color=:crimson, linewidth=2)
    lines!(ax1, epochs, accs, label="Test Acc", color=:royalblue, linewidth=2)
    axislegend(ax1)

    conf_mat = zeros(Int, 2, 2)
    for (t, p) in zip(y_true, y_pred)
        conf_mat[t, p] += 1
    end

    ax2 = Axis(fig[1, 2], title="Confusion Matrix",
               xticks=(1:2, ["Promoter", "Non-Prom"]),
               yticks=(1:2, ["Promoter", "Non-Prom"]))
    heatmap!(ax2, 1:2, 1:2, conf_mat', colormap=:Greens)

    for i in 1:2, j in 1:2
        text!(ax2, string(conf_mat[i,j]), position=(i,j), align=(:center, :center),
              color=conf_mat[i,j] > maximum(conf_mat)/2 ? :white : :black)
    end

    save("bigru_results.png", fig)
    println("\n[Visuals] Report saved to bigru_results.png")
    display(fig)
end

# 5. MAIN
function main()
    Random.seed!(123)
    X_data, Y_data = download_and_preprocess()

    (x_train, y_train), (x_test, y_test) = splitobs((X_data, Y_data), at = 0.8)
    train_loader = DataLoader((x_train, y_train), batchsize = 16, shuffle = true)

    model = build_improved_gru()
    opt_state = Flux.setup(Adam(0.0005), model)

    e_h, l_h, a_h = Int[], Float32[], Float32[]

    println("\n--- Training Bidirectional GRU ---")
    for epoch in 1:800
        epoch_l = 0.0f0
        Flux.trainmode!(model)

        for (x, y) in train_loader
            grads = Flux.gradient(model) do m
                reset!(m)
                l = crossentropy(m(x), y)
                @ignore_derivatives epoch_l += l
                l
            end
            Flux.update!(opt_state, model, grads[1])
        end

        if epoch % 20 == 0
            Flux.testmode!(model)
            reset!(model)
            acc = mean(onecold(model(x_test)) .== onecold(y_test))

            push!(e_h, epoch)
            push!(l_h, epoch_l / length(train_loader))
            push!(a_h, acc)

            @printf("Epoch %d | Loss: %.4f | Accuracy: %.2f%%\n", epoch, l_h[end], acc * 100)
            if acc >= 0.98 break end
        end
    end

    Flux.testmode!(model)
    reset!(model)
    y_p, y_t = onecold(model(x_test)), onecold(y_test)
    plot_results(e_h, l_h, a_h, y_t, y_p)
end

@time main()

