using Flux
using Statistics
using Random
using MLDatasets
using Images
using Distributions

# --- 0. Parameters ---
const IMAGE_SIZE = 32
const CHANNELS = 1
const STEP_SIZE = 10.0f0
const STEPS = 60
const NOISE = 0.005f0
const ALPHA = 0.1f0
const GRADIENT_CLIP = 0.03f0
const BATCH_SIZE = 128
const BUFFER_SIZE = 8192
const LEARNING_RATE = 0.0001f0
const EPOCHS = 60

# --- 1. Data Preprocessing ---
function load_mnist_32()
    # Load MNIST training data
    x_train, _ = MNIST(:train)[:]
    
    # Preprocess: Normalize to [-1, 1] and pad to 32x32
    processed = zeros(Float32, IMAGE_SIZE, IMAGE_SIZE, 1, size(x_train, 3))
    for i in 1:size(x_train, 3)
        img = x_train[:, :, i]
        img = (img .- 0.5f0) .* 2.0f0
        # Manual padding: 28x28 -> 32x32
        processed[3:30, 3:30, 1, i] .= img
    end
    return Flux.DataLoader(processed, batchsize=BATCH_SIZE, shuffle=true)
end

# --- 2. Build the EBM Network ---
function build_model()
    # Using SamePad() instead of SameNumber
    return Chain(
        Conv((5, 5), 1 => 16, swish, pad=SamePad(), stride=2),
        Conv((3, 3), 16 => 32, swish, pad=SamePad(), stride=2),
        Conv((3, 3), 32 => 64, swish, pad=SamePad(), stride=2),
        Conv((3, 3), 64 => 64, swish, pad=SamePad(), stride=2),
        Flux.flatten,
        Dense(256 => 64, swish), 
        Dense(64 => 1)
    ) |> f32
end

# --- 3. Langevin Sampler ---
function generate_samples(model, inp_imgs, steps, step_size, noise)
    curr_imgs = copy(inp_imgs)
    for _ in 1:steps
        # Add noise
        curr_imgs .+= noise .* randn(Float32, size(curr_imgs))
        curr_imgs .= clamp.(curr_imgs, -1.0f0, 1.0f0)
        
        # Langevin Gradient Step
        grads = Flux.gradient(x -> sum(model(x)), curr_imgs)[1]
        grads = clamp.(grads, -GRADIENT_CLIP, GRADIENT_CLIP)
        
        curr_imgs .+= step_size .* grads
        curr_imgs .= clamp.(curr_imgs, -1.0f0, 1.0f0)
    end
    return curr_imgs
end

# --- 4. Buffer Management ---
mutable struct ReplayBuffer
    storage::Array{Float32, 4}
    ptr::Int
    full::Bool
end

function sample_buffer!(buffer::ReplayBuffer, batch_size)
    n_new = rand(Binomial(batch_size, 0.05))
    n_old = batch_size - n_new
    
    new_imgs = rand(Float32, IMAGE_SIZE, IMAGE_SIZE, 1, n_new) .* 2.0f0 .- 1.0f0
    
    upper = buffer.full ? BUFFER_SIZE : buffer.ptr - 1
    idx = rand(1:max(1, upper), n_old)
    old_imgs = buffer.storage[:, :, :, idx]
    
    return cat(new_imgs, old_imgs, dims=4)
end

function update_buffer!(buffer::ReplayBuffer, imgs)
    n = size(imgs, 4)
    for i in 1:n
        buffer.storage[:, :, :, buffer.ptr] .= imgs[:, :, :, i]
        buffer.ptr += 1
        if buffer.ptr > BUFFER_SIZE
            buffer.ptr = 1
            buffer.full = true
        end
    end
end

# --- 5. Training Loop ---
function train()
    loader = load_mnist_32()
    model = build_model()
    opt_state = Flux.setup(Adam(LEARNING_RATE), model)
    
    init_storage = rand(Float32, IMAGE_SIZE, IMAGE_SIZE, 1, BUFFER_SIZE) .* 2.0f0 .- 1.0f0
    buffer = ReplayBuffer(init_storage, 1, false)

    println("Starting Training...")
    for epoch in 1:EPOCHS
        total_loss = 0.0f0
        
        # Timing each epoch
        @time for (batch_idx, real_imgs) in enumerate(loader)
            # Langevin Dynamics to get "fake" samples
            fake_imgs = sample_buffer!(buffer, size(real_imgs, 4))
            fake_imgs = generate_samples(model, fake_imgs, STEPS, STEP_SIZE, NOISE)
            
            update_buffer!(buffer, fake_imgs)

            # Gradient Step
            loss, grads = Flux.withgradient(model) do m
                real_out = m(real_imgs)
                fake_out = m(fake_imgs)
                
                # Energy difference + Regularization
                cdiv = mean(fake_out) - mean(real_out)
                reg = ALPHA * mean(real_out.^2 .+ fake_out.^2)
                return cdiv + reg
            end

            Flux.update!(opt_state, model, grads[1])
            total_loss += loss
        end
        println("Epoch $epoch | Avg Loss: $(total_loss / length(loader))")
    end
end

train()



