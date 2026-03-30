using HTTP, Flux, Statistics, MLUtils, GLMakie, Printf, Random
using Flux: onehotbatch, DataLoader, onecold, crossentropy
using ChainRulesCore: @ignore_derivatives

# --- CONSTANTS ---
const ALPHABET = ['a', 'g', 'c', 't']
const SEQ_LEN = 57

# ------------------------------------------------------------
# 1. DATA LOADING
# ------------------------------------------------------------
function download_and_preprocess()
    println("Fetching dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-length-databases/molecular-biology/promoter-gene-sequences/promoters.data"
    # Note: Using a robust string conversion for the HTTP body
    response = HTTP.get("https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data")
    raw_lines = split(strip(String(response.body)), "\n")

    X, Y = [], []
    for line in raw_lines
        parts = split(line, ",")
        if length(parts) < 3 continue end
        label = strip(parts[1]) == "+" ? 1 : 2
        seq = replace(strip(parts[3]), r"\s+" => "")
        if length(seq) != 57 continue end

        encoded = Float32.(onehotbatch(collect(lowercase(seq)), ALPHABET))
        push!(X, encoded)
        push!(Y, onehotbatch(label, 1:2))
    end
    return cat(X..., dims=3), cat(Y..., dims=2)
end

# ------------------------------------------------------------
# 2. TRANSFORMER COMPONENTS
# ------------------------------------------------------------
function get_sinusoidal_embeddings(dim, len)
    pos = reshape(0:len-1, 1, :)
    div_term = exp.((0:2:dim-1) .* -(log(10000.0) / dim))
    pe = zeros(Float32, dim, len)
    pe[1:2:end, :] .= sin.(pos .* div_term)
    pe[2:2:end, :] .= cos.(pos .* div_term)
    return pe
end

function build_transformer_model()
    hidden_dim = 64
    pe = get_sinusoidal_embeddings(hidden_dim, SEQ_LEN)

    return Chain(
        Dense(4, hidden_dim),
        x -> x .+ pe,
        LayerNorm(hidden_dim),
        x -> mean(x, dims=2),               # Global Average Pooling
        Flux.flatten,
        Dense(hidden_dim, 32, relu),
        Dense(32, 2),
        softmax
        )
end

# ------------------------------------------------------------
# 3. VISUALIZATION
# ------------------------------------------------------------
function plot_performance(epochs, train_losses, test_accs, y_true, y_pred)
    fig = Figure(size = (1200, 500))

    # Axis 1: Training Curves
    ax1 = Axis(fig[1, 1], title="Training Progress", xlabel="Epoch", ylabel="Value")
    lines!(ax1, epochs, train_losses, label="Loss", color=:red, linewidth=2)
    lines!(ax1, epochs, test_accs, label="Test Accuracy", color=:blue, linewidth=2)
    axislegend(ax1)

    # Axis 2: Confusion Matrix
    conf_mat = zeros(Int, 2, 2)
    for (t, p) in zip(y_true, y_pred)
        conf_mat[t, p] += 1
    end

    ax2 = Axis(fig[1, 2], title="Confusion Matrix",
               xticks=(1:2, ["Promoter (+)", "Non-Prom (-)"]),
               yticks=(1:2, ["Promoter (+)", "Non-Prom (-)"]))
    heatmap!(ax2, 1:2, 1:2, conf_mat', colormap=:Blues)

    for i in 1:2, j in 1:2
        text!(ax2, string(conf_mat[i,j]), position=(i,j), align=(:center, :center),
              color=conf_mat[i,j] > maximum(conf_mat)/2 ? :white : :black)
    end

    save("transformer_results.png", fig)
    println("\n[Visuals] Plots saved to transformer_results.png")
    display(fig)
end

# ------------------------------------------------------------
# 4. MAIN EXECUTION
# ------------------------------------------------------------
function main()
    Random.seed!(42)

    # Step 1: Load Data
    X_raw, Y_raw = download_and_preprocess()

    # Step 2: Split and Load
    (x_train, y_train), (x_test, y_test) = splitobs((X_raw, Y_raw), at=0.8)
    train_loader = DataLoader((x_train, y_train), batchsize=16, shuffle=true)

    # Step 3: Build Model
    model = build_transformer_model()
    opt_state = Flux.setup(Adam(0.0005), model)

    epochs_log, loss_log, acc_log = Int[], Float32[], Float32[]

    println("\n--- Training Transformer ---")
    for epoch in 1:1500
        current_loss = 0.0f0
        for (x, y) in train_loader
            grads = Flux.gradient(model) do m
                l = crossentropy(m(x), y)
                @ignore_derivatives current_loss += l
                l
            end
            Flux.update!(opt_state, model, grads[1])
        end

        # Periodic Evaluation
        if epoch % 20 == 0
            preds = onecold(model(x_test))
            actuals = onecold(y_test)
            acc = mean(preds .== actuals)

            push!(epochs_log, epoch)
            push!(loss_log, current_loss / length(train_loader))
            push!(acc_log, acc)

            @printf("Epoch %d | Loss: %.4f | Accuracy: %.2f%%\n",
                    epoch, loss_log[end], acc * 100)
        end
    end

    # Step 4: Final Evaluation and Visuals
    y_final_preds = onecold(model(x_test))
    y_final_true = onecold(y_test)

    plot_performance(epochs_log, loss_log, acc_log, y_final_true, y_final_preds)
end

# Run the script
@time main()


