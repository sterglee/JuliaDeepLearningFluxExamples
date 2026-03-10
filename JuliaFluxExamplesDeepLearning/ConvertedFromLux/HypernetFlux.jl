#time Lux: 5.1 sec/epoch
#time Flux: 5.2 sec/epoch

using Flux, MLDatasets, MLUtils, OneHotArrays, Optimisers, Printf, Random

# ------------------------------------------------------------------------------
# 1. Loading Datasets (Same logic as Lux, simplified for Flux)
# ------------------------------------------------------------------------------
function load_dataset(::Type{dset}, batchsize::Int) where {dset}
    # Load training and test data
    train_data = dset(:train)
    test_data = dset(:test)
    
    x_train = reshape(train_data.features, 28, 28, 1, :)
    y_train = onehotbatch(train_data.targets, 0:9)
    
    x_test = reshape(test_data.features, 28, 28, 1, :)
    y_test = onehotbatch(test_data.targets, 0:9)

    return (
        DataLoader((x_train, y_train), batchsize=batchsize, shuffle=true),
        DataLoader((x_test, y_test), batchsize=batchsize, shuffle=false)
    )
end

function load_datasets(batchsize=32)
    return load_dataset.((MNIST, FashionMNIST), batchsize)
end

# ------------------------------------------------------------------------------
# 2. Implement the HyperNetwork Structure
# ------------------------------------------------------------------------------

# Flux doesn't have @compact, so we use a custom struct or a closure.
# We use destructure to handle the dynamic weight assignment.
struct HyperNet
    weight_generator
    core_reconstructor # The 're' function from Flux.destructure
    core_network       # Kept for reference/initialization
end

Flux.@functor HyperNet (weight_generator,)

function create_hypernet()
    # Define the core architecture (the network whose weights we will generate)
    core_network = Chain(
        Conv((3, 3), 1 => 16, relu; stride=2),
        Conv((3, 3), 16 => 32, relu; stride=2),
        Conv((3, 3), 32 => 64, relu; stride=2),
        GlobalMeanPool(),
        Flux.flatten,
        Dense(64, 10)
    )

    # Flatten the core network to find out how many parameters we need to generate
    flat_ps, re = Flux.destructure(core_network)
    num_params = length(flat_ps)

    # Define the generator (HyperNet)
    # It takes a task ID (embedding) and generates all weights for the core network
    weight_generator = Chain(
        Embedding(2 => 32),
        Dense(32, 64, relu),
        Dense(64, num_params)
    )

    return HyperNet(weight_generator, re, core_network)
end

# Forward pass: (task_id, image_data)
function (hn::HyperNet)(task_id, x)
    # 1. Generate core weights from the task ID
    # Task ID should be 1 or 2
    generated_weights = hn.weight_generator(task_id)
    
    # 2. Reconstruct the core network with these new weights
    # Note: re(vec(weights)) handles single-sample or batch generation
    # For simplicity, we assume one task ID per batch here
    core_net_instance = hn.core_reconstructor(generated_weights)
    
    # 3. Pass images through the generated network
    return core_net_instance(x)
end

# ------------------------------------------------------------------------------
# 3. Utility and Training
# ------------------------------------------------------------------------------

function accuracy(model, dataloader, task_idx)
    total_correct, total = 0, 0
    # In Flux, we use testmode! to handle Dropout/BatchNorm if they existed
    Flux.testmode!(model)
    for (x, y) in dataloader
        preds = model(task_idx, x)
        total_correct += sum(onecold(preds) .== onecold(y))
        total += size(x, 4)
    end
    Flux.trainmode!(model)
    return total_correct / total
end

function train()
    Random.seed!(1234)
    model = create_hypernet()
    dataloaders = load_datasets()
    
    # Optimizer setup (using Optimisers.jl as in original)
    opt_state = Flux.setup(Optimisers.Adam(0.0003f0), model)

    nepochs = 5
    for epoch in 1:nepochs
        for (task_idx, (train_loader, test_loader)) in enumerate(dataloaders)
            
            # Training loop for specific dataset (MNIST or FashionMNIST)
            for (x, y) in train_loader
                loss, grads = Flux.withgradient(model) do m
                    logits = m(task_idx, x)
                    Flux.logitcrossentropy(logits, y)
                end
                Flux.update!(opt_state, model, grads[1])
            end

            # Logging
            t_acc = accuracy(model, train_loader, task_idx) * 100
            v_acc = accuracy(model, test_loader, task_idx) * 100
            d_name = task_idx == 1 ? "MNIST" : "FashionMNIST"
            
            @printf "[%3d] %12s | Train Acc: %3.2f%% | Test Acc: %3.2f%%\n" epoch d_name t_acc v_acc
        end
    end
end

@time train() 



