using Flux
using Flux: onehotbatch, onecold, logitcrossentropy
using Metalhead
using Images, ImageTransformations
using DataFrames, CSV
using CUDA
using Random
using Statistics
using Plots
using Dates

# -------------------------------
# 1. SAFE SPLIT
# -------------------------------
function split_indices(idx)
    n = length(idx)

    n_train = Int(floor(0.7 * n))
    n_temp  = n - n_train

    train_idx = idx[1:n_train]
    temp_idx  = idx[n_train+1:end]

    n_val = Int(floor(0.5 * n_temp))

    val_idx  = temp_idx[1:n_val]
    test_idx = temp_idx[n_val+1:end]

    return train_idx, val_idx, test_idx
end

# -------------------------------
# 2. LOAD IMAGE
# -------------------------------
function load_image(img_id, image_dirs)
    img_name = img_id * ".jpg"

    img_path = nothing
    for dir in image_dirs
        p = joinpath(dir, img_name)
        if isfile(p)
            img_path = p
            break
        end
    end

    img_path === nothing && error("Image not found: $img_id")

    img = load(img_path)
    img = imresize(img, (224, 224))

    x = Float32.(channelview(RGB.(img)))
    x = permutedims(x, (3, 2, 1))  # (W,H,C)

    return (x .- 0.5f0) ./ 0.5f0
end

# -------------------------------
# 3. MODEL (CORRECT & STABLE)
# -------------------------------
function create_model(num_classes)
    return Chain(
        Metalhead.ResNet(18),   # backbone
        Flux.flatten,
        Dense(1000, num_classes)
        )
end

# -------------------------------
# 4. TRAIN STEP
# -------------------------------
function train_epoch!(model, idx, df, dirs, opt_state, num_classes, device, batch_size)
    Flux.trainmode!(model)

    total_loss = 0.0f0
    correct = 0
    total = 0

    for i in 1:batch_size:length(idx)
        batch_idx = idx[i:min(i+batch_size-1, end)]
        rows = df[batch_idx, :]

        x = cat([load_image(r.image_id, dirs) for r in eachrow(rows)]..., dims=4) |> device
            y = onehotbatch(rows.label, 1:num_classes) |> device

            loss, grads = Flux.withgradient(model) do m
                logitcrossentropy(m(x), y)
            end

            Flux.update!(opt_state, model, grads[1])

            preds = onecold(cpu(model(x)))

            correct += sum(preds .== rows.label)
            total_loss += loss * size(y,2)
            total += size(y,2)
        end

        return total_loss/total, correct/total
    end

    # -------------------------------
    # 5. EVALUATION
    # -------------------------------
    function evaluate(model, idx, df, dirs, num_classes, device, batch_size)
        Flux.testmode!(model)

        y_true = Int[]
        y_pred = Int[]

        total_loss = 0.0f0
        total = 0

        for i in 1:batch_size:length(idx)
            batch_idx = idx[i:min(i+batch_size-1, end)]
            rows = df[batch_idx, :]

            x = cat([load_image(r.image_id, dirs) for r in eachrow(rows)]..., dims=4) |> device
                y = onehotbatch(rows.label, 1:num_classes) |> device

                logits = model(x)
                loss = logitcrossentropy(logits, y)

                preds = onecold(cpu(logits))

                append!(y_true, rows.label)
                append!(y_pred, preds)

                total_loss += loss * size(y,2)
                total += size(y,2)
            end

            acc = sum(y_true .== y_pred) / length(y_true)

            return total_loss/total, acc, y_true, y_pred
        end

        # -------------------------------
        # 6. MAIN
        # -------------------------------
        function main()
            println("Starting training...")
            start_time = now()

            image_dirs = ["HAM10000_images_part_1", "HAM10000_images_part_2"]
            df = CSV.read("HAM10000_metadata.csv", DataFrame)

            # Labels
            classes = unique(df.dx)
            label_map = Dict(l => i for (i,l) in enumerate(classes))
                df.label = [label_map[x] for x in df.dx]

                    num_classes = length(classes)

                    # Split
                    Random.seed!(42)
                    idx = shuffle(1:nrow(df))
                    train_idx, val_idx, test_idx = split_indices(idx)

                    # Device
                    device = CUDA.functional() ? gpu : cpu
                    println("Using device: $device")

                    # Model + optimizer
                    model = create_model(num_classes) |> device
                    opt = Adam(1e-4)
                    opt_state = Flux.setup(opt, model)

                    train_loss_hist = Float32[]
                    val_acc_hist = Float32[]

                    # Training loop
                    for epoch in 1:5
                        train_loss, train_acc = train_epoch!(model, train_idx, df, image_dirs, opt_state, num_classes, device, 16)
                        val_loss, val_acc, _, _ = evaluate(model, val_idx, df, image_dirs, num_classes, device, 16)

                        push!(train_loss_hist, train_loss)
                        push!(val_acc_hist, val_acc)

                        println("Epoch $epoch")
                        println("Train Loss=$(round(train_loss,digits=4)) Acc=$(round(train_acc,digits=4))")
                        println("Val   Loss=$(round(val_loss,digits=4)) Acc=$(round(val_acc,digits=4))")
                    end

                    # Test
                    test_loss, test_acc, y_true, y_pred = evaluate(model, test_idx, df, image_dirs, num_classes, device, 16)

                    println("\nTest Accuracy: $(round(test_acc*100,digits=2))%")

                    # Confusion Matrix
                    cm = zeros(Int, num_classes, num_classes)
                    for (t,p) in zip(y_true, y_pred)
                        cm[t,p] += 1
                    end

                    # Plot
                    p1 = plot(train_loss_hist, title="Training Loss")
                    p2 = plot(val_acc_hist, title="Validation Accuracy")
                    p3 = heatmap(cm, title="Confusion Matrix")

                    display(plot(p1, p2, p3, layout=(3,1)))

                    println("\nExecution Time: $(now() - start_time)")
                end

                main()

