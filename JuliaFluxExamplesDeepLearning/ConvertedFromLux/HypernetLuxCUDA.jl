using Lux,
    ComponentArrays,
    MLDatasets,
    MLUtils,
    OneHotArrays,
    Optimisers,
    Printf,
    Random,
    CUDA

# ------------------------------------------------------------------------------
# 1. Loading Datasets (Pre-transfer to GPU or Batch-wise)
# ------------------------------------------------------------------------------
function load_dataset(::Type{dset}, n_train, n_eval, batchsize) where {dset}
    # Standard loading logic
    train_data = dset(:train)
    x_train = Float32.(reshape(train_data.features[:, :, 1:n_train], 28, 28, 1, :))
    y_train = onehotbatch(train_data.targets[1:n_train], 0:9)

    test_data = dset(:test)
    x_test = Float32.(reshape(test_data.features[:, :, 1:n_eval], 28, 28, 1, :))
    y_test = onehotbatch(test_data.targets[1:n_eval], 0:9)

    return (
        DataLoader((x_train, y_train); batchsize=batchsize, shuffle=true),
        DataLoader((x_test, y_test); batchsize=batchsize, shuffle=false),
        )
end

# ------------------------------------------------------------------------------
# 2. HyperNet Structure
# ------------------------------------------------------------------------------
# We keep the HyperNet logic but use gpu_device() for transfers
function HyperNet(weight_generator::AbstractLuxLayer, core_network::AbstractLuxLayer)
    ca_axes = getaxes(
        ComponentArray(Lux.initialparameters(Random.default_rng(), core_network))
        )
    return @compact(; ca_axes, weight_generator, core_network, dispatch=:HyperNet) do (x, y)
        ps_new = ComponentArray(vec(weight_generator(x)), ca_axes)
        @return core_network(y, ps_new)
    end
end

function Lux.initialparameters(rng::AbstractRNG, hn::CompactLuxLayer{:HyperNet})
    return (; weight_generator=Lux.initialparameters(rng, hn.layers.weight_generator))
end

# ------------------------------------------------------------------------------
# 3. Training Logic Optimized for CUDA
# ------------------------------------------------------------------------------
function train()
    # Select GPU device
    gdev = gpu_device()
    cdev = cpu_device()

    # Hyperparameters
    batchsize = 256 # Larger batch size exploits GPU better
    n_train = parse(Bool, get(ENV, "CI", "false")) ? 1024 : 60000
    n_eval = 10000

    # Load and move model to GPU
    dataloaders = load_dataset.((MNIST, FashionMNIST), n_train, n_eval, batchsize)
    model = Chain(
        Conv((3, 3), 1 => 16, relu; stride=2),
        Conv((3, 3), 16 => 32, relu; stride=2),
        Conv((3, 3), 32 => 64, relu; stride=2),
        GlobalMeanPool(),
        FlattenLayer(),
        Dense(64, 10),
        )

    # Core Network for param length
    core_net = model
    hnet = HyperNet(
        Chain(
            Embedding(2 => 32),
            Dense(32, 64, relu),
            Dense(64, Lux.parameterlength(core_net)),
            ),
        core_net
        )

        rng = Random.default_rng()
        ps, st = Lux.setup(rng, hnet) |> gdev
        opt = Optimisers.Adam(0.001f0)
        st_opt = Optimisers.setup(opt, ps)

        # Loss Function
        function loss_fn(p, (task_id, x), y, s)
            y_pred, s_new = hnet((task_id, x), p, s)
            return logitcrossentropy(y_pred, y), s_new
        end

        nepochs = 5
        for epoch in 1:nepochs, task_idx in 1:2
            train_loader, test_loader = dataloaders[task_idx]

            # Move task index to GPU once
            gpu_task_idx = fill(task_idx, 1) |> gdev

            stime = time()
            for (x, y) in train_loader
                # Move batch to GPU
                x_g, y_g = x |> gdev, y |> gdev

                # Compute gradients
                (loss, st), grads = Lux.value_and_gradient(
                    p -> loss_fn(p, (gpu_task_idx, x_g), y_g, st),
                    ps
                    )

                # Update parameters
                st_opt, ps = Optimisers.update(st_opt, ps, grads)
            end
            ttime = time() - stime

            @printf "[%d] Task %d | Time: %.2fs | Loss: %.4f\n" epoch task_idx ttime loss
        end

        return ps, st
end

@time train()

