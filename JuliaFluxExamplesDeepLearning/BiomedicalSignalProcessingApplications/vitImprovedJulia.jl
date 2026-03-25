using Flux
using Flux: onehotbatch, onecold, logitcrossentropy
using Images, ImageTransformations
using DataFrames, CSV
using CUDA
using Random
using Statistics
using Plots
using Dates

# -------------------------------
# 1. Custom Batching
# -------------------------------
function get_batches(idx, batch_size)
    return [idx[i:min(i+batch_size-1, end)] for i in 1:batch_size:length(idx)]
    end

    # -------------------------------
    # 2. Custom Train/Val/Test Split
    # -------------------------------
    function split_indices(idx)
        n = length(idx)

        n_train = Int(floor(0.7 * n))
        n_temp  = n - n_train

        train_idx = idx[1:n_train]
        temp_idx  = idx[n_train+1:end]

        n_val = Int(floor(0.5 * length(temp_idx)))

        val_idx  = temp_idx[1:n_val]
        test_idx = temp_idx[n_val+1:end]

        return train_idx, val_idx, test_idx
    end

    # -------------------------------
    # 3. Load & Preprocess Image
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

        x = Float32.(channelview(RGB.(img)))  # (C,H,W)
        x = permutedims(x, (2,3,1))           # (H,W,C)

        return (x .- 0.5f0) ./ 0.5f0
    end

    # -------------------------------
    # 4. Simple CNN Model (Fully Stable)
    # -------------------------------
    function create_model(num_classes)
        return Chain(
            Conv((3,3), 3 => 16, relu, pad=1),
            MaxPool((2,2)),

            Conv((3,3), 16 => 32, relu, pad=1),
            MaxPool((2,2)),

            Conv((3,3), 32 => 64, relu, pad=1),
            MaxPool((2,2)),

            Conv((3,3), 64 => 128, relu, pad=1),

            x -> mean(x, dims=(1,2)),  # Global Average Pooling

            Flux.flatten,
            Dense(128, num_classes)
            )
    end

    # -------------------------------
    # 5. Training Step
    # -------------------------------
    function train_epoch!(model, idx, df, dirs, opt_state, num_classes, device, batch_size)
        Flux.trainmode!(model)

        total_loss = 0.0f0
        total = 0

        for batch in get_batches(idx, batch_size)
            rows = df[batch, :]

            x = cat([load_image(r.image_id, dirs) for r in eachrow(rows)]..., dims=4) |> device
                y = onehotbatch(rows.label, 1:num_classes) |> device

                loss, grads = Flux.withgradient(model) do m
                    logitcrossentropy(m(x), y)
                end

                Flux.update!(opt_state, model, grads)

                total_loss += loss * size(y,2)
                total += size(y,2)
            end

            return total_loss / total
        end

        # -------------------------------
        # 6. Evaluation
        # -------------------------------
        function evaluate(model, idx, df, dirs, num_classes, device, batch_size)
            Flux.testmode!(model)

            y_true = Int[]
            y_pred = Int[]

            total_loss = 0.0f0
            total = 0

            for batch in get_batches(idx, batch_size)
                rows = df[batch, :]

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
            # 7. Metrics
            # -------------------------------
            function compute_metrics(y_true, y_pred, num_classes)
                cm = zeros(Int, num_classes, num_classes)

                for (t,p) in zip(y_true, y_pred)
                    cm[t,p] += 1
                end

                precision = zeros(Float64, num_classes)
                recall    = zeros(Float64, num_classes)
                f1        = zeros(Float64, num_classes)

                for i in 1:num_classes
                    tp = cm[i,i]
                    fp = sum(cm[:,i]) - tp
                    fn = sum(cm[i,:]) - tp

                    precision[i] = tp + fp > 0 ? tp/(tp+fp) : 0.0
                    recall[i]    = tp + fn > 0 ? tp/(tp+fn) : 0.0
                    f1[i]        = (precision[i]+recall[i]) > 0 ? 2*precision[i]*recall[i]/(precision[i]+recall[i]) : 0.0
                end

                return cm, precision, recall, f1
            end

            # -------------------------------
            # 8. Main
            # -------------------------------
            function main()
                start_time = now()

                image_dirs = ["HAM10000_images_part_1", "HAM10000_images_part_2"]
                df = CSV.read("HAM10000_metadata.csv", DataFrame)

                Random.seed!(42)

                # Labels
                labels = unique(df.dx)
                label_map = Dict(l => i for (i,l) in enumerate(labels))
                    df.label = [label_map[x] for x in df.dx]
                        num_classes = length(labels)

                        # Split indices
                        idx = shuffle(1:nrow(df))
                        train_idx, val_idx, test_idx = split_indices(idx)

                        # Device
                        device = CUDA.functional() ? gpu : cpu
                        println("Using device: $device")

                        # Model
                        model = create_model(num_classes) |> device
                        opt_state = Flux.setup(Adam(1e-4), model)

                        loss_hist = Float32[]
                        acc_hist  = Float32[]

                        println("Training...")

                        for epoch in 1:5
                            train_loss = train_epoch!(model, train_idx, df, image_dirs, opt_state, num_classes, device, 32)
                            val_loss, val_acc, _, _ = evaluate(model, val_idx, df, image_dirs, num_classes, device, 32)

                            push!(loss_hist, train_loss)
                            push!(acc_hist, val_acc)

                            println("Epoch $epoch | Loss=$(round(train_loss,digits=4)) | Val Acc=$(round(val_acc*100,digits=2))%")
                        end

                        # Test
                        test_loss, test_acc, y_true, y_pred = evaluate(model, test_idx, df, image_dirs, num_classes, device, 32)

                        println("\nTest Accuracy: $(round(test_acc*100,digits=2))%")

                        # Metrics
                        cm, prec, rec, f1 = compute_metrics(y_true, y_pred, num_classes)

                        println("\nClassification Report:")
                        for i in 1:num_classes
                            println("Class $i → P=$(round(prec[i],digits=3)) R=$(round(rec[i],digits=3)) F1=$(round(f1[i],digits=3))")
                        end

                        # Plots
                        p1 = plot(loss_hist, title="Training Loss", label="Loss")
                        p2 = plot(acc_hist, title="Validation Accuracy", label="Accuracy")
                        p3 = heatmap(cm, title="Confusion Matrix", xlabel="Predicted", ylabel="Actual")

                        display(plot(p1, p2, p3, layout=(3,1)))

                        elapsed = now() - start_time
                        println("\nExecution time: $(Dates.value(elapsed)/1e9) seconds")
                    end

                    main()

