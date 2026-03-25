using Flux, Images, CSV, DataFrames, Statistics, CUDA, ProgressMeter, Dates, Plots
using Flux: train!, onehotbatch, onecold, DataLoader

# --- 1. Helper Functions ---
function accuracy(loader, model, device)
    acc = 0.0
    num_samples = 0
    for (x, y) in loader
        x, y = x |> device, y |> device
        # Compare predicted indices to actual indices
        acc += sum(onecold(model(x)) .== onecold(y))
        num_samples += size(y, 2)
    end
    return acc / num_samples
end

# --- 2. Training Loop with Metrics ---
function train_model()
    # ... [Data Loading & Model Setup same as your script] ...

    # Initialize History
    history = Dict(
        "train_loss" => Float32[],
        "val_loss"   => Float32[],
        "train_acc"  => Float32[],
        "val_acc"    => Float32[]
        )

    @info "Starting Training..."
    for epoch in 1:10
        train_l = 0.0f0
        @showprogress "Epoch $epoch: " for (x, y) in train_loader
            # Standard Flux gradient pattern
            val, grads = Flux.withgradient(ps) do
                loss(x, y)
            end
            Flux.update!(opt, ps, grads)
            train_l += val
        end

        # Calculate Epoch Metrics
        avg_train_loss = train_l / length(train_loader)
        avg_val_loss = mean(loss(x, y) for (x, y) in val_loader) # Simple mean for val

            t_acc = accuracy(train_loader, model, device)
            v_acc = accuracy(val_loader, model, device)

            # Store in history
            push!(history["train_loss"], avg_train_loss)
            push!(history["val_loss"], avg_val_loss)
            push!(history["train_acc"], t_acc)
            push!(history["val_acc"], v_acc)

            println("Epoch $epoch | Train Loss: $(round(avg_train_loss, digits=4)) | Val Acc: $(round(v_acc, digits=4))")
        end

        return history
    end

    # --- 3. Visualization ---
    function plot_metrics(history)
        epochs = 1:length(history["train_loss"])

        # Plot Loss
        p1 = plot(epochs, [history["train_loss"] history["val_loss"]],
                  label=["Train" "Val"], title="Model Loss",
                  xlabel="Epochs", ylabel="Loss", lw=2)

        # Plot Accuracy
        p2 = plot(epochs, [history["train_acc"] history["val_acc"]],
                  label=["Train" "Val"], title="Model Accuracy",
                  xlabel="Epochs", ylabel="Accuracy", lw=2, ylims=(0, 1))

        plot(p1, p2, layout=(1, 2), size=(900, 400))
        savefig("training_results.png")
    end

    # Run execution
    history = train_model()
    plot_metrics(history)

