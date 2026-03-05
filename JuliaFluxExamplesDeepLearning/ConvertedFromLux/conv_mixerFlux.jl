using Flux, MLDatasets, MLUtils, OneHotArrays, Printf, Random, Statistics
using DataAugmentation, ConcreteStructs, Images, Interpolations
using Flux: DataLoader, gradient, setup, update!, trainmode!, testmode!

# 1. ConvMixer Architecture
#
function ConvMixer(; dim, depth, kernel_size=5, patch_size=2)
    return Chain(
        # Patch Embedding
        Conv((patch_size, patch_size), 3 => dim, relu; stride=patch_size),
        BatchNorm(dim),
        # ConvMixer Blocks
        [
            Chain(
                # Spatial Mixing (Depthwise Convolution)
                SkipConnection(
                    Chain(
                        Conv((kernel_size, kernel_size), dim => dim, relu;
                             groups=dim, pad=SamePad()),
                        BatchNorm(dim)
                        ),
                        +
                            ),
                        # Channel Mixing (Pointwise Convolution)
                        Conv((1, 1), dim => dim, relu),
                        BatchNorm(dim)
                        ) for _ in 1:depth
                            ]...,
                        GlobalMeanPool(),
                        Flux.flatten,
                        Dense(dim => 10)
                        )
                        end

                        # 2. Data Loading Utility
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

                        function get_dataloaders(batchsize)
                            means, stds = (0.4914f0, 0.4822f0, 0.4465f0), (0.2471f0, 0.2435f0, 0.2616f0)
                            train_trans = RandomResizeCrop((32, 32)) |> Maybe(FlipX{2}()) |> ImageToTensor() |> Normalize(means, stds)
                            test_trans = ImageToTensor() |> Normalize(means, stds)

                            train_loader = DataLoader(TensorDataset(CIFAR10(split=:train), train_trans), batchsize=batchsize, shuffle=true)
                            test_loader = DataLoader(TensorDataset(CIFAR10(split=:test), test_trans), batchsize=batchsize)
                            return train_loader, test_loader
                        end

                        # 3. Accuracy Utility
                        function get_accuracy(model, loader, device)
                            testmode!(model)
                            correct, total = 0, 0
                            for (x, y) in loader
                                x, y = x |> device, y |> device
                                correct += sum(onecold(model(x)) .== onecold(y))
                                total += size(x, 4)
                            end
                            return correct / total
                        end

                        # 4. Training Entry Point
                        function train_convmixer(;
                                                 batchsize=64, hidden_dim=32, depth=8, patch_size=2,
                                                 kernel_size=5, epochs=2, lr_max=0.05
                                                 )
                            Random.seed!(1234)
                            device = Flux.get_device()
                            @info "Training ConvMixer on $device"

                            # Initialize Model and Data
                            model = ConvMixer(dim=hidden_dim, depth=depth, kernel_size=kernel_size, patch_size=patch_size) |> device
                            train_loader, test_loader = get_dataloaders(batchsize)

                            # Optimizer and Learning Rate Schedule
                            #
                            opt_state = setup(OptimiserChain(ClipNorm(1.0), AdamW(lr_max, (0.9, 0.999), 0.0001)), model)
                            lr_schedule = LinearInterpolation([0, epochs*0.4, epochs*0.8, epochs+1], [0, lr_max, lr_max/20, 0])

                            for epoch in 1:epochs
                                trainmode!(model)
                                current_lr = lr_schedule(epoch)
                                Flux.adjust!(opt_state, current_lr)

                                loss_total = 0.0f0
                                for (x, y) in train_loader
                                    x, y = x |> device, y |> device

                                    l, grads = Flux.withgradient(model) do m
                                        y_hat = m(x)
                                        Flux.logitcrossentropy(y_hat, y)
                                    end

                                    update!(opt_state, model, grads[1])
                                    loss_total += l
                                end

                                test_acc = get_accuracy(model, test_loader, device)
                                @printf "Epoch %2d | LR: %.4f | Loss: %.4f | Test Acc: %.2f%%\n" epoch current_lr (loss_total/length(train_loader)) (test_acc*100)
                            end

                            return model
                        end

                        # Run training
                        model = train_convmixer()


