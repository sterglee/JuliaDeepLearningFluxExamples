using CSV
using DataFrames
using Images
using FileIO
using Flux
using Flux: onehotbatch, onecold, crossentropy, flatten, GlobalMeanPool
using Metalhead
using CUDA
using Random
using Statistics
using MLUtils

# Standardize device selection
const device = CUDA.functional() ? gpu : cpu

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

struct HAMDataset
    df::DataFrame
    image_dirs::Vector{String}
end

Base.length(d::HAMDataset) = nrow(d.df)

function load_image(path)
    img = load(path)
    img = imresize(img, (224, 224))
    
    # channelview returns (C, H, W). 
    # Flux/Metalhead expect (W, H, C, N) or (H, W, C, N).
    # We'll convert to Float32 and ensure RGB format.
    img_data = Float32.(channelview(RGB.(img)))
    
    # Permute from (C, H, W) to (H, W, C)
    return permutedims(img_data, (2, 3, 1))
end

function get_batch(dataset::HAMDataset, indices)
    imgs = []
    labels = Int[]

    for idx in indices
        row = dataset.df[idx, :]
        imgname = string(row.image_id, ".jpg")
        imgpath = nothing

        for dir in dataset.image_dirs
            p = joinpath(dir, imgname)
            if isfile(p)
                imgpath = p
                break
            end
        end

        if !isnothing(imgpath)
            push!(imgs, load_image(imgpath))
            push!(labels, row.label)
        end
    end

    if isempty(imgs)
        return nothing, nothing
    end

    # Concatenate along the 4th dimension to create (H, W, C, N)
    x = cat(imgs..., dims=4)
    return x, labels
end

# ------------------------------------------------------------
# Training & Evaluation
# ------------------------------------------------------------

function train_epoch!(model, loader, opt_state, dataset)
    loss_sum = 0.0
    correct = 0
    total = 0

    for idx_batch in loader
        x_cpu, y_cpu = get_batch(dataset, idx_batch)
        if isnothing(x_cpu) continue end

        x = device(x_cpu)
        y_hot = device(onehotbatch(y_cpu, 0:6))

        # Modern Flux gradient handling
        loss, back = Flux.withgradient(model) do m
            ŷ = m(x)
            crossentropy(ŷ, y_hot)
        end

        Flux.update!(opt_state, model, back[1])

        # Move to CPU for metrics to avoid GPU memory bloat
        ŷ_cpu = cpu(model(x))
        preds = onecold(ŷ_cpu, 0:6)

        loss_sum += loss * length(y_cpu)
        correct += sum(preds .== y_cpu)
        total += length(y_cpu)
    end

    return loss_sum / total, correct / total
end

function evaluate(model, loader, dataset)
    total = 0
    correct = 0
    preds_all = Int[]
    labels_all = Int[]

    for idx_batch in loader
        x_cpu, y_cpu = get_batch(dataset, idx_batch)
        if isnothing(x_cpu) continue end

        x = device(x_cpu)
        ŷ = cpu(model(x))
        preds = onecold(ŷ, 0:6)

        append!(preds_all, preds)
        append!(labels_all, y_cpu)
        correct += sum(preds .== y_cpu)
        total += length(y_cpu)
    end

    return correct / total, preds_all, labels_all
end

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

function main()
    image_dirs = ["HAM10000_images_part_1", "HAM10000_images_part_2"]
    
    if !isfile("HAM10000_metadata.csv")
        error("Metadata file not found. Ensure HAM10000_metadata.csv is in the directory.")
    end

    df = CSV.read("HAM10000_metadata.csv", DataFrame)

    # Label encoding
    labels_unique = unique(df.dx)
    labelmap = Dict(l => i-1 for (i, l) in enumerate(labels_unique))
    df.label = [labelmap[x] for x in df.dx]

    # Splitting
    Random.seed!(42)
    n = nrow(df)
    idx = shuffle(1:n)

    train_end = Int(floor(0.7 * n))
    val_end   = Int(floor(0.85 * n))

    train_ds = HAMDataset(df[idx[1:train_end], :], image_dirs)
    val_ds   = HAMDataset(df[idx[train_end+1:val_end], :], image_dirs)
    test_ds  = HAMDataset(df[idx[val_end+1:end], :], image_dirs)

    # DataLoaders - Fixed keyword 'batchsize'
    batch_size = 32
    train_loader = DataLoader(1:length(train_ds), batchsize=batch_size, shuffle=true)
    val_loader   = DataLoader(1:length(val_ds), batchsize=batch_size)
    test_loader  = DataLoader(1:length(test_ds), batchsize=batch_size)

    # ------------------------------------------------------------
    # Model Setup
    # ------------------------------------------------------------
    
    # Metalhead ResNet usually expects (224, 224, 3, N)
    # 
    base_resnet = ResNet(18; pretrain=true)
    
    # We strip the original classifier (the last 2 layers of the .layers Chain)
    backbone = base_resnet.layers[1:end-2]

    model = Chain(
        backbone,
        GlobalMeanPool(),
        flatten,
        Dense(512, 7), # ResNet18 outputs 512 channels
        softmax
    ) |> device

    opt_state = Flux.setup(Adam(1e-4), model)
    epochs = 5

    println("Starting Training on $(CUDA.functional() ? "GPU" : "CPU")...")

    for epoch in 1:epochs
        t_loss, t_acc = train_epoch!(model, train_loader, opt_state, train_ds)
        v_acc, _, _ = evaluate(model, val_loader, val_ds)

        println("Epoch $epoch | Loss $(round(t_loss, digits=4)) | Train Acc $(round(t_acc, digits=4)) | Val Acc $(round(v_acc, digits=4))")
    end

    test_acc, _, _ = evaluate(model, test_loader, test_ds)
    println("\nFinal Test Accuracy: ", round(test_acc, digits=4))
end

main()

