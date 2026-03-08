using Flux
using Flux: train!, onehotbatch, onecold, DataLoader
using Images
using CSV
using DataFrames
using Lathe.preprocess: TrainTestSplit
using Statistics
using CUDA
using ProgressMeter
using Dates

# 1. Dataset Handling
# Equivalent to the Python __getitem__ logic
function load_image(path, target_size=(224, 224))
    img = load(path)
    img_resized = imresize(img, target_size)
    # Convert to Float32 and normalize (equivalent to Normalize(0.5, 0.5))
    img_array = channelview(RGB.(img_resized))
    return (Float32.(img_array) .- 0.5f0) ./ 0.5f0
end

# 2. Model Definition
# SimpleCNN equivalent using Flux.Chain
function create_model(num_classes)
    return Chain(
        # Features
        Conv((3, 3), 3 => 32, relu, pad=1), BatchNorm(32), MaxPool((2, 2)),
        Conv((3, 3), 32 => 64, relu, pad=1), BatchNorm(64), MaxPool((2, 2)),
        Conv((3, 3), 64 => 128, relu, pad=1), BatchNorm(128), MaxPool((2, 2)),
        Conv((3, 3), 128 => 256, relu, pad=1), BatchNorm(256), MaxPool((2, 2)),

        # Classifier
        Flux.flatten,
        Dense(256 * 14 * 14, 512, relu),
        Dropout(0.5),
        Dense(512, num_classes)
    )
end

# 3. Training Loop
function main()
    total_start = now()
    device = CUDA.functional() ? gpu : cpu
    @info "Using device: $(device == gpu ? "CUDA GPU" : "CPU")"

    # --- Data Preparation ---
    df = CSV.read("HAM10000_metadata.csv", DataFrame)

    # Path logic
    df.image_path = map(id -> begin
        p1 = "HAM10000_images_part_1/$id.jpg"
        isfile(p1) ? p1 : "HAM10000_images_part_2/$id.jpg"
    end, df.image_id)

    # Label Mapping
    labels = unique(df.dx)
    label_map = Dict(label => i for (i, label) in enumerate(labels))
    df.label = [label_map[x] for x in df.dx]
    num_classes = length(labels)

    # Split (Simplified 70/15/15)
    train_df, tmp_df = TrainTestSplit(df, 0.7)
    val_df, test_df = TrainTestSplit(tmp_df, 0.5)

    # Create DataLoaders (In a real scenario, use lazy loading)
    # For brevity, this assumes images fit in RAM; otherwise, use a custom generator
    function get_data(dataframe)
        X = [load_image(p) for p in dataframe.image_path]
        Y = onehotbatch(dataframe.label, 1:num_classes)
        return DataLoader((cat(X..., dims=4), Y), batchsize=32, shuffle=true)
    end

    @info "Loading images into memory..."
    train_loader = get_data(train_df) |> device
    val_loader = get_data(val_df) |> device

    # --- Model Init ---
    model = create_model(num_classes) |> device
    ps = Flux.params(model)
    opt = Adam(1f-4)
    loss(x, y) = Flux.logitcrossentropy(model(x), y)

    # --- Training ---
    @info "Starting Training..."
    for epoch in 1:5
        epoch_start = now()

        train_loss = 0.0f0
        @showprogress "Epoch $epoch: " for (x, y) in train_loader
            gs = gradient(() -> loss(x, y), ps)
            Flux.update!(opt, ps, gs)
            train_loss += loss(x, y)
        end

        # Validation
        val_acc = mean(onecold(model(val_loader.data[1])) .== onecold(val_loader.data[2]))

        epoch_end = now()
        duration = canonicalize(Dates.CompoundPeriod(epoch_end - epoch_start))

        println("Epoch $epoch | Time: $duration")
        println("Avg Loss: $(train_loss/length(train_loader)) | Val Acc: $val_acc")
        println("-"^30)
    end

    total_end = now()
    @info "Total Execution Time: $(canonicalize(Dates.CompoundPeriod(total_end - total_start)))"
end

main()
