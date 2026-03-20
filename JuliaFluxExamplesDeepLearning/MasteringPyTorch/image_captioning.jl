using Flux
using Flux: DataLoader, onehotbatch, onecold, flatten, logitcrossentropy
using MLDatasets
using Statistics
using Optimisers  # Used for the newer Flux training API

# 1. Define the Model Architecture
# This matches the ConvNet class in your PyTorch notebook
function create_model()
    return Chain(
        # Conv2d(1, 16, 3, 1) -> (W, H, C, Batch)
        Conv((3, 3), 1 => 16, relu),

        # Conv2d(16, 32, 3, 1)
        Conv((3, 3), 16 => 32, relu),

        # MaxPool2d(2)
        MaxPool((2, 2)),

        # Dropout2d(0.10)
        Dropout(0.1),

        # Flatten
        flatten,

        # Linear(4608, 64)
        Dense(4608, 64, relu),

        # Dropout(0.25)
        Dropout(0.25),

        # Linear(64, 10)
        Dense(64, 10)
        )
end

# 2. Data Loading and Preprocessing
function get_data(batch_size)
    # Load MNIST
    xtrain, ytrain = MNIST(:train)[:]
    xtest, ytest = MNIST(:test)[:]

    # Preprocess: Normalization (mean=0.1307, std=0.3081)
    # Reshape to (28, 28, 1, Batch) for Flux
    preprocess(x) = reshape((x .- 0.1307f0) ./ 0.3081f0, 28, 28, 1, :)

    xtrain, xtest = preprocess(xtrain), preprocess(xtest)

    # One-hot encode targets
    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    # Create DataLoaders (Flux.DataLoader, not Flux.Data.DataLoader)
    train_loader = DataLoader((xtrain, ytrain), batchsize=batch_size, shuffle=true)
    test_loader = DataLoader((xtest, ytest), batchsize=batch_size)

    return train_loader, (xtest, ytest)
end

# 3. Training Logic
function train_mnist()
    # Hyperparameters
    batch_size = 32
    epochs = 2

    model = create_model()
    train_loader, (xtest, ytest) = get_data(batch_size)

    # Optimizer Setup
    # PyTorch Adadelta(lr=0.5) is matched by chaining AdaDelta with a Descent(0.5) rule
    # To this:
    opt_rule = Optimisers.OptimiserChain(Optimisers.AdaDelta(), Optimisers.Descent(0.5))
    opt_state = Flux.setup(opt_rule, model)

    println("Starting Training...")
    for epoch in 1:epochs
        for (x, y) in train_loader
            # Calculate gradients
            loss, grads = Flux.withgradient(model) do m
                # logitcrossentropy combines logsoftmax and nll_loss
                logitcrossentropy(m(x), y)
            end

            # Update parameters
            Flux.update!(opt_state, model, grads[1])
        end

        # Accuracy Check (Test Set)
        predictions = model(xtest)
        acc = mean(onecold(predictions) .== onecold(ytest))
        println("Epoch $epoch: Test Accuracy = $(round(acc * 100, digits=2))%")
    end

    return model
end

# Execute
final_model = train_mnist()

