using Flux
using Flux: onehotbatch, DataLoader, logitcrossentropy
using Statistics, Random, Printf

# --- 1. CONFIG ---
const SEQ_LEN = 250
const VOCAB_SIZE = 5
const INPUT_DIM = SEQ_LEN * VOCAB_SIZE # 1250 features

# --- 2. MLP MODEL CONSTRUCTION ---
function build_mlp()
    return Chain(
        # Flatten the input: (SEQ_LEN, BATCH) -> (INPUT_DIM, BATCH)
        # We use One-Hot encoding to represent the 5 possible bases
        x -> reshape(onehotbatch(x, 1:VOCAB_SIZE), INPUT_DIM, :),
        
        # Hidden Layer 1
        Dense(INPUT_DIM, 512, relu),
        Dropout(0.2), # Prevent overfitting to specific positions
        
        # Hidden Layer 2
        Dense(512, 128, relu),
        
        # Output Layer (2 classes: Promoter vs Non-Promoter)
        Dense(128, 2)
    )
end

# --- 3. TRAINING ---
function train_mlp()
    xt, yt, xv, yv = get_large_dataset() # From previous genomic script
    
    model = build_mlp()
    opt_state = Flux.setup(Adam(1e-3), model)
    
    loader = DataLoader((xt, yt), batchsize=32, shuffle=true)
    
    println("Training MLP Baseline...")
    for epoch in 1:10
        for (x, y) in loader
            loss, grads = Flux.withgradient(model) do m
                logitcrossentropy(m(x), onehotbatch(y, 1:2))
            end
            Flux.update!(opt_state, model, grads[1])
        end
        
        # Check Validation
        acc = mean(Flux.onecold(model(xv)) .== yv)
        @printf("Epoch %d | Val Acc: %.2f%%\n", epoch, acc * 100)
    end
end