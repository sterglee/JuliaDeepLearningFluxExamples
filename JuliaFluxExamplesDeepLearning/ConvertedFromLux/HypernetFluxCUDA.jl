using Flux, MLDatasets, MLUtils, OneHotArrays, Optimisers, Printf, Random, CUDA

# Ensure CUDA is functional
if !CUDA.functional()
    @warn "CUDA is not functional. Falling back to CPU."
end

# ------------------------------------------------------------------------------
# 1. Loading Datasets (Pre-reshape and batch)
# ------------------------------------------------------------------------------
function load_dataset(::Type{dset}, batchsize::Int) where {dset}
    train_data = dset(:train)
    test_data = dset(:test)

    # Use Float32 for GPU performance
    x_train = Float32.(reshape(train_data.features, 28, 28, 1, :))
    y_train = Float32.(onehotbatch(train_data.targets, 0:9))

    x_test = Float32.(reshape(test_data.features, 28, 28, 1, :)) 
    y_test = Float32.(onehotbatch(test_data.targets, 0:9))

    return (
        DataLoader((x_train, y_train), batchsize=batchsize, shuffle=true),
        DataLoader((x_test, y_test), batchsize=batchsize, shuffle=false)
        )
end

function load_datasets(batchsize=128) # Increased batchsize for GPU utilization
    return load_dataset.((MNIST, FashionMNIST), batchsize)
end

# ------------------------------------------------------------------------------
# 2. HyperNetwork Structure
# ------------------------------------------------------------------------------
struct HyperNet
    weight_generator
    core_reconstructor
end

# Register for gradient tracking and GPU movement
Flux.@functor HyperNet (weight_generator,)

function create_hypernet()
    core_network = Chain(
        Conv((3, 3), 1 => 16, relu; stride=2),
        Conv((3, 3), 16 => 32, relu; stride=2),
        Conv((3, 3), 32 => 64, relu; stride=2),
        GlobalMeanPool(),
        Flux.flatten,
        Dense(64, 10)
        )

    flat_ps, re = Flux.destructure(core_network)

    weight_generator = Chain(
        Embedding(2 => 32),
        Dense(32, 64, relu),
        Dense(64, length(flat_ps))
        )

    return HyperNet(weight_generator, re)
end

function (hn::HyperNet)(task_id, x)
    # Task_id must be on GPU (usually a CuArray or a scalar moved to GPU)
    generated_weights = hn.weight_generator(task_id)
    # Reconstruct the core network
    core_net_instance = hn.core_reconstructor(generated_weights)
    return core_net_instance(x)
end

# ------------------------------------------------------------------------------
# 3. Optimized Utility and Training
# ------------------------------------------------------------------------------

function accuracy(model, dataloader, task_idx)
    total_correct, total = 0, 0
    Flux.testmode!(model)
    # Move task_idx to GPU once
    gpu_task_idx = task_idx |> gpu
    for (x, y) in dataloader
        x_g, y_g = x |> gpu, y |> gpu
        preds = model(gpu_task_idx, x_g)
        total_correct += sum(onecold(preds) .== onecold(y_g))
        total += size(x, 4)
    end
    Flux.trainmode!(model)
    return total_correct / total
end

function train()
    Random.seed!(1234)
    # 1. Create and move model to GPU
    model = create_hypernet() |> gpu
    dataloaders = load_datasets(256) # Larger batches exploit CUDA parallelism

    opt_state = Flux.setup(Optimisers.Adam(0.001f0), model)

    nepochs = 5
    for epoch in 1:nepochs
        for (task_idx, (train_loader, test_loader)) in enumerate(dataloaders)

            # Prepare task index as a GPU-compatible input
            gpu_task_idx = task_idx |> gpu

            for (x, y) in train_loader
                # 2. Move batch to GPU
                x_g, y_g = x |> gpu, y |> gpu

                loss, grads = Flux.withgradient(model) do m
                    logits = m(gpu_task_idx, x_g)
                    Flux.logitcrossentropy(logits, y_g)
                end
                Flux.update!(opt_state, model, grads[1])
            end

            v_acc = accuracy(model, test_loader, task_idx) * 100
            d_name = task_idx == 1 ? "MNIST" : "FashionMNIST"
            @printf "[%3d] %12s | Test Acc: %3.2f%%\n" epoch d_name v_acc
        end
    end
end

@time train()


