using Flux
using Flux: DataLoader
using NPZ
using Statistics
using Random
using Distributions
using Zygote  # FIX: Added this to allow the use of Zygote.Buffer

# -----------------------------
# 1. LOAD DATA
# -----------------------------
# Ensuring (Time, Features, Batch) orientation
X_raw = npzread("data/X_train.npy")[1:10000, :, :]
X_train = Float32.(permutedims(X_raw, (2, 3, 1))) 

seq_len, feat_dim, n_samples = size(X_train)
latent_dim = seq_len * feat_dim
batch_size = 100
epochs = 5

# -----------------------------
# 2. AFFINE COUPLING LAYER
# -----------------------------
struct AffineCoupling
    mask::Vector{Bool}
    s_net::Chain
    t_net::Chain
end

Flux.@functor AffineCoupling

function (ac::AffineCoupling)(x)
    # x is (latent_dim, Batch)
    x1 = x[ac.mask, :]
    x2 = x[.!ac.mask, :]
    
    s = ac.s_net(x1)
    t = ac.t_net(x1)
    
    # y2 = x2 * exp(s) + t
    y2 = x2 .* exp.(s) .+ t
    
    # FIX: Use Zygote.Buffer to construct the output array
    # This allows "mutation-like" assignment in a differentiable way
    y_out = Zygote.Buffer(x)
    y_out[ac.mask, :] = x1
    y_out[.!ac.mask, :] = y2
    
    return copy(y_out), sum(s; dims=1)
end

# -----------------------------
# 3. NORMALIZING FLOW MODEL
# -----------------------------
n_layers = 6
model = Chain([
    begin
        # Alternating binary mask
        mask = [Bool(mod(j + i, 2) == 0) for j in 1:latent_dim]
        n1 = sum(mask)
        n2 = latent_dim - n1
        
        s_net = Chain(Dense(n1, 128, relu), Dense(128, n2))
        t_net = Chain(Dense(n1, 128, relu), Dense(128, n2))
        
        AffineCoupling(mask, s_net, t_net)
    end for i in 1:n_layers
]...)

# -----------------------------
# 4. FORWARD & LOSS
# -----------------------------
function flow_forward(m, x)
    x_flat = reshape(x, :, size(x, 3)) 
    log_det = zeros(Float32, 1, size(x_flat, 2))
    z = x_flat
    for layer in m
        z, ld = layer(z)
        log_det = log_det .+ ld
    end
    return z, log_det
end

function flow_loss(m, x)
    z, log_det = flow_forward(m, x)
    # log_prob of standard normal base distribution
    log_prob = sum(-0.5f0 .* (z.^2 .+ log(2f0 * Float32(π))); dims=1)
    return -mean(log_prob .+ log_det)
end

# -----------------------------
# 5. OPTIMIZER & DATA LOADER
# -----------------------------
opt_state = Flux.setup(Flux.Adam(1e-3), model)
train_loader = DataLoader((X_train,), batchsize=batch_size, shuffle=true)

# -----------------------------
# 6. TRAINING LOOP
# -----------------------------
println("Starting Normalizing Flow Training...")

for epoch in 1:epochs
    for (x_batch,) in train_loader
        # Standard modern Flux gradient pattern
        grads = Flux.gradient(model) do m
            flow_loss(m, x_batch)
        end
        Flux.update!(opt_state, model, grads[1])
    end
    
    # Validation
    current_l = flow_loss(model, X_train[:, :, 1:200])
    println("Epoch $epoch - Loss = $(round(current_l, digits=4))")
end

println("Training complete.")


