# # Convolutional Neural Network on CIFAR-10 (Flux Version)

# ## Package Imports
using Comonicon, Flux, Optimisers, Printf, Random, Statistics
using Flux: DataLoader, train!, GLM

# Note: Enzyme is typically used for Lux/manual loops;
# Flux usually relies on Zygote.jl for AD.
using Zygote

# Assuming common.jl contains data loading utilities adapted for Flux
# If not, I have included a standard data setup below.
# Set some global flags that will improve performance
XLA_FLAGS = get(ENV, "XLA_FLAGS", "")
ENV["XLA_FLAGS"] = "$(XLA_FLAGS) --xla_gpu_enable_cublaslt=true"

# ## Load Common Packages

using ConcreteStructs,
    DataAugmentation,
    ImageShow,
    Lux,
    MLDatasets,
    MLUtils,
    OneHotArrays,
    Printf,
    ProgressTables,
    Random,
    BFloat16s
using Reactant

# ## Data Loading Functionality

@concrete struct TensorDataset
    dataset
    transform
end

Base.length(ds::TensorDataset) = length(ds.dataset)

function Base.getindex(ds::TensorDataset, idxs::Union{Vector{<:Integer},AbstractRange})
    img = Image.(eachslice(convert2image(ds.dataset, idxs); dims=3))
    y = onehotbatch(ds.dataset.targets[idxs], 0:9)
    return stack(parent ∘ itemdata ∘ Base.Fix1(apply, ds.transform), img), y
end

function get_cifar10_dataloaders(::Type{T}, batchsize; kwargs...) where {T}
    cifar10_mean = T.((0.4914, 0.4822, 0.4465))
    cifar10_std = T.((0.2471, 0.2435, 0.2616))

    train_transform =
        RandomResizeCrop((32, 32)) |>
            Maybe(FlipX{2}()) |>
                ImageToTensor() |>
                    Normalize(cifar10_mean, cifar10_std) |>
                        ToEltype(T)

                    test_transform = ImageToTensor() |> Normalize(cifar10_mean, cifar10_std) |> ToEltype(T)

                    trainset = TensorDataset(CIFAR10(; Tx=T, split=:train), train_transform)
                    trainloader = DataLoader(trainset; batchsize, shuffle=true, kwargs...)

                    testset = TensorDataset(CIFAR10(; Tx=T, split=:test), test_transform)
                    testloader = DataLoader(testset; batchsize, shuffle=false, kwargs...)

                    return trainloader, testloader
end

# ## Utility Functions

function accuracy(model, ps, st, dataloader)
    total_correct, total = 0, 0
    cdev = cpu_device()
    for (x, y) in dataloader
        target_class = onecold(cdev(y))
        predicted_class = onecold(cdev(first(model(x, ps, st))))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

# ## Training Loop

function train_model(
    model,
    opt,
    scheduler=nothing;
    batchsize::Int=512,
    seed::Int=1234,
    epochs::Int=25,
    bfloat16::Bool=false,
    )
    rng = Random.default_rng()
    Random.seed!(rng, seed)

    prec = bfloat16 ? bf16 : f32
    prec_jl = bfloat16 ? BFloat16 : Float32
    prec_str = bfloat16 ? "BFloat16" : "Float32"
    @printf "[Info] Using %s precision\n" prec_str

    dev = reactant_device(; force=true)

    trainloader, testloader =
        get_cifar10_dataloaders(prec_jl, batchsize; partial=false) |> dev

    ps, st = prec(Lux.setup(rng, model)) |> dev

    train_state = Training.TrainState(model, ps, st, opt)

    x_ra = rand(rng, prec_jl, size(first(trainloader)[1])) |> dev
    @printf "[Info] Compiling model with Reactant.jl\n"
    model_compiled = Reactant.with_config(;
                                          dot_general_precision=PrecisionConfig.HIGH,
                                          convolution_precision=PrecisionConfig.HIGH,
                                          ) do
        @compile model(x_ra, ps, Lux.testmode(st))
                                          end
                                          @printf "[Info] Model compiled!\n"

                                          loss_fn = CrossEntropyLoss(; logits=Val(true))

                                          pt = ProgressTable(;
                                                             header=[
                                                                 "Epoch", "Learning Rate", "Train Accuracy (%)", "Test Accuracy (%)", "Time (s)"
                                                                 ],
                                                             widths=[24, 24, 24, 24, 24],
                                                             format=["%3d", "%.6f", "%.6f", "%.6f", "%.6f"],
                                                             color=[:normal, :normal, :blue, :blue, :normal],
                                                             border=true,
                                                             alignment=[:center, :center, :center, :center, :center],
                                                             )

                                                             @printf "[Info] Training model\n"
                                                             initialize(pt)

                                                             for epoch in 1:epochs
                                                                 stime = time()
                                                                 lr = 0
                                                                 for (i, (x, y)) in enumerate(trainloader)
                                                                     if scheduler !== nothing
                                                                         lr = scheduler((epoch - 1) + (i + 1) / length(trainloader))
                                                                         train_state = Optimisers.adjust!(train_state, lr)
                                                                     end
                                                                     (_, loss, _, train_state) = Training.single_train_step!(
                                                                         AutoEnzyme(), loss_fn, (x, y), train_state; return_gradients=Val(false)
                                                                         )
                                                                     isnan(loss) && error("NaN loss encountered!")
                                                                 end
                                                                 ttime = time() - stime

                                                                 train_acc =
                                                                     accuracy(
                                                                         model_compiled,
                                                                         train_state.parameters,
                                                                         Lux.testmode(train_state.states),
                                                                         trainloader,
                                                                         ) * 100
                                                                     test_acc =
                                                                         accuracy(
                                                                             model_compiled,
                                                                             train_state.parameters,
                                                                             Lux.testmode(train_state.states),
                                                                             testloader,
                                                                             ) * 100

                                                                         scheduler === nothing && (lr = NaN32)
                                                                         next(pt, [epoch, lr, train_acc, test_acc, ttime])
                                                             end

                                                             finalize(pt)
                                                             return @printf "[Info] Finished training\n"
end



# ## Model Definition
# In Flux, we don't need to wrap everything in nested Chains unless
# for organization, as the model itself holds the parameters.

function SimpleCNN()
    return Flux.Chain(
        # Feature Extractor
        Flux.Conv((3, 3), 3 => 16, relu; stride=2, pad=1),
        Flux.BatchNorm(16),
        Flux.Conv((3, 3), 16 => 32, relu; stride=2, pad=1),
        Flux.BatchNorm(32),
        Flux.Conv((3, 3), 32 => 64, relu; stride=2, pad=1),
        Flux.BatchNorm(64),
        Flux.Conv((3, 3), 64 => 128, relu; stride=2, pad=1),
        Flux.BatchNorm(128),

        # Classifier
        Flux.GlobalMeanPool(),
        Flux.flatten, # Flux uses flatten function or Flatten layer
        Flux.Dense(128 => 64, relu),
        Flux.BatchNorm(64),
        Flux.Dense(64 => 10)
        )
end

# ## Training Logic
# Flux models are mutable. We use Optimisers.jl (same as your Lux setup)
# but apply updates to the model's parameters directly.

function train_model(model, opt_rule, args; batchsize, seed, epochs, bfloat16)
    Random.seed!(seed)

    # Data Loading (Placeholder for your common.jl implementation)
    train_loader, test_loader = get_cifar10_dataloaders(Float32, batchsize)
    # Initialize Optimiser State
    opt_state = Optimisers.setup(opt_rule, model)

    for epoch in 1:epochs
        losses = Float32[]
        for (x, y) in train_loader
            # Move to Float16 if requested (simplified)
            if bfloat16
                x = bfloat16.(x)
            end

            # Compute gradient and loss
            val, grads = Flux.withgradient(model) do m
                y_hat = m(x)
                Flux.logitcrossentropy(y_hat, y)
            end

            # Update parameters and optimiser state
            opt_state, model = Optimisers.update!(opt_state, model, grads[1])
            push!(losses, val)
        end

        @printf "Epoch [%d/%d]: Avg Loss: %.4f\n" epoch epochs mean(losses)
    end

    return model
end

# ## Entry Point
function main(;
                                  batchsize::Int=512,
                                  weight_decay::Float64=0.0001,
                                  clip_norm::Bool=false,
                                  seed::Int=1234,
                                  epochs::Int=2,
                                  lr::Float64=0.003,
                                  bfloat16::Bool=false,
                                  )
    # Initialize Model
    model = SimpleCNN()

    # Setup Optimiser (Optimisers.jl is compatible with both)
    opt = AdamW(lr, (0.9, 0.999), weight_decay)
    if clip_norm
        opt = OptimiserChain(ClipNorm(1.0), opt)
    end

    return train_model(model, opt, nothing; batchsize, seed, epochs, bfloat16)
end

main()


