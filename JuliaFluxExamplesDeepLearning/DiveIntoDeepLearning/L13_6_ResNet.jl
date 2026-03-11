using Flux
using Statistics
using MLUtils

# 1. Define the Residual Block
struct ResidualBlock
    branch::Chain
    shortcut # Can be identity or a Chain
end

# Constructor
function ResidualBlock(in_channels, out_channels; stride=1)
    branch = Chain(
        Conv((3, 3), in_channels => out_channels, pad=1, stride=stride, bias=false),
        BatchNorm(out_channels, relu),
        Conv((3, 3), out_channels => out_channels, pad=1, bias=false),
        BatchNorm(out_channels)
    )
    
    # Corrected Shortcut Logic
    # If dimensions match, use identity. If not, use 1x1 Conv to match them.
    shortcut = (stride != 1 || in_channels != out_channels) ? 
               Chain(Conv((1, 1), in_channels => out_channels, stride=stride, bias=false), BatchNorm(out_channels)) : 
               identity

    return ResidualBlock(branch, shortcut)
end

# Make the struct "visible" to Flux's parameter tracking
Flux.@functor ResidualBlock

# Forward pass: f(x) = relu(branch(x) + shortcut(x))
(m::ResidualBlock)(x) = relu.(m.branch(x) .+ m.shortcut(x))



# 2. Build the ResNet Model
function build_resnet(num_classes)
    return Chain(
        # Initial Layer (MNIST is 1-channel grayscale)
        Conv((7, 7), 1 => 64, pad=3, stride=2, bias=false),
        BatchNorm(64, relu),
        MaxPool((3, 3), pad=1, stride=2),
        
        # Residual Stages
        ResidualBlock(64, 64),
        ResidualBlock(64, 64),
        ResidualBlock(64, 128, stride=2),
        ResidualBlock(128, 128),
        
        # Head
        GlobalMeanPool(),
        Flux.flatten,
        Dense(128 => num_classes)
    )
end

# 3. Training Utilities
loss(m, x, y) = Flux.logitcrossentropy(m(x), y)

# 4. Corrected Main Function
function main()
    println("--- Starting Corrected ResNet Training ---")
    
    # Data Setup (MNIST dimensions: 28x28x1xBatch)
    batch_size = 32
    x_train = rand(Float32, 28, 28, 1, 128) 
    y_train = Flux.onehotbatch(rand(1:10, 128), 1:10)
    train_loader = DataLoader((x_train, y_train), batchsize=batch_size, shuffle=true)

    # Model & Optimizer Setup
    # Call build_resnet BEFORE setup to avoid structural mismatches
    model = build_resnet(10)
    opt_state = Flux.setup(Flux.Adam(0.001), model)

    # Single Training Loop
    for epoch in 1:3
        epoch_loss = 0.0
        for (x, y) in train_loader
            # Use withgradient for modern Flux training
            l, grads = Flux.withgradient(model) do m
                loss(m, x, y)
            end
            Flux.update!(opt_state, model, grads[1])
            epoch_loss += l
        end
        println("Epoch $epoch | Loss: $(round(epoch_loss/length(train_loader), digits=4))")
    end
    
    println("--- Main Call Successful ---")
end

# Call the entry point
main()

