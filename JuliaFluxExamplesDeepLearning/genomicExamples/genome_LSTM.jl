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

    X = Vector{Array{Float32,2}}()
    Y = []

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

# 2. MODEL (LSTM)
function build_lstm_model()
    hidden_dim = 128
    return Chain(
        LSTM(4 => hidden_dim),
        x -> x[:, end, :],   # Take the final hidden state of the sequence
        Dense(hidden_dim, 16, relu),
        Dense(16, 2),
        softmax
        )
end

# 3. VISUALIZATION
function plot_results(epochs, losses, accs, y_true, y_pred)
    fig = Figure(size = (1200, 500))

    # Left: Training Curves
    ax1 = Axis(fig[1, 1], title="LSTM Training Progress", xlabel="Epoch", ylabel="Value")
    lines!(ax1, epochs, losses, label="Loss", color=:red, linewidth=2)
    lines!(ax1, epochs, accs, label="Accuracy", color=:blue, linewidth=2)
    axislegend(ax1)

    # Right: Confusion Matrix
    conf_mat = zeros(Int, 2, 2)
    for (t, p) in zip(y_true, y_pred)
        conf_mat[t, p] += 1
    end

    ax2 = Axis(fig[1, 2], title="Confusion Matrix",
               xticks=(1:2, ["Promoter", "Non-Prom"]),
               yticks=(1:2, ["Promoter", "Non-Prom"]))
    heatmap!(ax2, 1:2, 1:2, conf_mat', colormap=:Purples)

    for i in 1:2, j in 1:2
        text!(ax2, string(conf_mat[i,j]), position=(i,j), align=(:center, :center),
              color=conf_mat[i,j] > maximum(conf_mat)/2 ? :white : :black)
    end

    save("lstm_performance.png", fig)
    println("\n[Visuals] Performance plots saved to: lstm_performance.png")
    display(fig)
end

# 4. MAIN
function main()
    Random.seed!(42)
    X_data, Y_data = download_and_preprocess()

    (x_train, y_train), (x_test, y_test) = splitobs((X_data, Y_data), at = 0.8)
    train_loader = DataLoader((x_train, y_train), batchsize = 16, shuffle = true)

    model = build_lstm_model()
    opt_state = Flux.setup(Adam(0.001), model)

    epochs_hist, loss_hist, acc_hist = Int[], Float32[], Float32[]

    println("\n--- Training LSTM ---")
    for epoch in 1:500
        total_loss = 0.0f0
        for (x, y) in train_loader
            grads = Flux.gradient(model) do m
                Flux.reset!(m)
                l = crossentropy(m(x), y)
                @ignore_derivatives total_loss += l
                l
            end
            Flux.update!(opt_state, model, grads[1])
        end

        if epoch % 20 == 0
            Flux.reset!(model)
            preds = onecold(model(x_test))
            actuals = onecold(y_test)
            acc = mean(preds .== actuals)

            push!(epochs_hist, epoch)
            push!(loss_hist, total_loss / length(train_loader))
            push!(acc_hist, acc)

            @printf("Epoch %d | Loss: %.4f | Test Accuracy: %.2f%%\n",
                    epoch, loss_hist[end], acc * 100)

            if acc >= 0.98
                println("Target accuracy reached early.")
                break
            end
        end
    end

    # Final Evaluation for Plotting
    Flux.reset!(model)
    y_pred = onecold(model(x_test))
    y_true = onecold(y_test)

    plot_results(epochs_hist, loss_hist, acc_hist, y_true, y_pred)
end

@time main()

