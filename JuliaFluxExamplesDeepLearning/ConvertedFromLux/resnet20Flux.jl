using Flux, MLDatasets, MLUtils, OneHotArrays, Printf, Random, Statistics
using DataAugmentation, ConcreteStructs, Images
using Flux: DataLoader, @layer, trainmode!, testmode!, withgrad

# --- Model Definition ---

# Helper for Conv + BatchNorm
function ConvBN(kernel_size, (in_chs, out_chs), act; stride=1, pad=SamePad())
    return Chain(
        Conv(kernel_size, in_chs => out_chs, act; stride=stride, pad=pad),
        BatchNorm(out_chs)
        )
end

# Basic Residual Block
function BasicBlock(in_channels, out_channels; stride=1)
    # The skip connection path
    # If dimensions change (stride > 1), we use a 1x1 conv to match channels
    connection = if (stride == 1 && in_channels == out_channels)
        identity
    else
        # Downsampling/Channel matching projection
        Conv((1, 1), in_channels => out_channels; stride=stride)
    end

    return Chain(
        SkipConnection(
            Chain(
                ConvBN((3, 3), (in_channels, out_channels), relu; stride=stride),
                ConvBN((3, 3), (out_channels, out_channels), identity)
                ),
            (mx, x) -> relu.(mx .+ connection(x))
            )
            )
end

function ResNet20(; num_classes=10)
    # Block configuration for ResNet-20: 3 stages of 3 blocks each
    # Configuration: (in, out, blocks, stride)
    configs = [
        (16, 16, 3, 1),
        (16, 32, 3, 2),
        (32, 64, 3, 2)
        ]

    layers = []

    # Initial Layer
    push!(layers, ConvBN((3, 3), (3, 16), relu))

    # Residual Stages
    for (in_ch, out_ch, num_blocks, stride) in configs
        for i in 1:num_blocks
            s = (i == 1) ? stride : 1
            cur_in = (i == 1) ? in_ch : out_ch
            push!(layers, BasicBlock(cur_in, out_ch; stride=s))
        end
    end

    # Final Layers
    push!(layers, GlobalMeanPool())
    push!(layers, Flux.flatten)
    push!(layers, Dense(64 => num_classes))

    return Chain(layers...)
end

# --- Data Pipeline ---

@concrete struct TensorDataset
    dataset
    transform
end

Base.length(ds::TensorDataset) = length(ds.dataset)

function Base.getindex(ds::TensorDataset, idxs::Union{Vector{<:Integer},AbstractRange})
    img_data = convert2image(ds.dataset, idxs)
    img_slices = Image.(eachslice(img_data; dims=ndims(img_data)))
    x = stack(parent ∘ itemdata ∘ Base.Fix1(apply, ds.transform), img_slices)
    y = onehotbatch(ds.dataset.targets[idxs], 0:9)
    return x, y
end

function get_loaders(batchsize)
    means, stds = (0.4914f0, 0.4822f0, 0.4465f0), (0.2471f0, 0.2435f0, 0.2616f0)
    t_train = RandomResizeCrop((32, 32)) |> Maybe(FlipX{2}()) |> ImageToTensor() |> Normalize(means, stds)
    t_test = ImageToTensor() |> Normalize(means, stds)

    l_train = DataLoader(TensorDataset(CIFAR10(split=:train), t_train), batchsize=batchsize, shuffle=true)
    l_test = DataLoader(TensorDataset(CIFAR10(split=:test), t_test), batchsize=batchsize)
    return l_train, l_test
end

# --- Training Logic ---

function accuracy(model, loader, device)
    testmode!(model)
    correct, total = 0, 0
    for (x, y) in loader
        x, y = x |> device, y |> device
        correct += sum(onecold(model(x)) .== onecold(y))
        total += size(x, 4)
    end
    return correct / total
end

function train_resnet()
    # Settings
    epochs = 2
    batchsize = 128
    lr = 0.001f0
    device = Flux.get_device()

    # Initialize
    model = ResNet20() |> device
    train_loader, test_loader = get_loaders(batchsize)
    opt_state = Flux.setup(AdamW(lr), model)

    @info "Starting Training on $device"
    for epoch in 1:epochs
        trainmode!(model)
        loss_total = 0.0f0

        for (x, y) in train_loader
            x, y = x |> device, y |> device

            val, grads = withgradient(model) do m
                y_hat = m(x)
                Flux.logitcrossentropy(y_hat, y)
            end

            Flux.update!(opt_state, model, grads[1])
            loss_total += val
        end

        acc = accuracy(model, test_loader, device)
        @printf "Epoch %d | Avg Loss: %.4f | Test Acc: %.2f%%\n" epoch (loss_total/length(train_loader)) (acc*100)
    end
    return model
end

# Run
train_resnet()

