# Final training loss = 0.059831776
using Flux
using Flux: DataLoader
using NPZ
using Statistics

# -----------------------------
# Load data
# -----------------------------
X_raw = npzread("data/X_train.npy")[1:10000, :, :]
y_raw = npzread("data/y_train.npy")[1:10000, :]

X_train = Float32.(permutedims(X_raw, (2,3,1))) # (Time, Features, Batch)
y_train = Float32.(permutedims(y_raw))          # (Classes, Batch)

# -----------------------------
# BiLSTM definition
# -----------------------------
struct BiLSTM
    fwd::LSTM
    bwd::LSTM
end

Flux.@functor BiLSTM

function (m::BiLSTM)(x)
    x_lstm = permutedims(x, (2,3,1))       # (features, batch, time)
    y_f = m.fwd(x_lstm)
    x_rev = reverse(x_lstm, dims=3)
    y_b = m.bwd(x_rev)
    y_b = reverse(y_b, dims=3)
    cat(y_f, y_b; dims=1)                  # (2*hidden, batch, time)
end

# -----------------------------
# Global max pool over time
# -----------------------------
function global_maxpool(x)
    pooled = maximum(x, dims=3)           # (features, batch, 1)
    dropdims(pooled, dims=3)              # -> (features, batch)
end

# -----------------------------
# Model
# -----------------------------
model = Chain(
    Conv((26,), 4=>320, relu),
    MaxPool((13,)),
    Dropout(0.2),
    BiLSTM(LSTM(320=>256), LSTM(320=>256)),
    global_maxpool,    # (features, batch)
    Dense(512, 512, relu),
    Dropout(0.5),
    Dense(512, 690),
    σ
)

# -----------------------------
# Loss function
# -----------------------------
loss(model, x, y) = Flux.Losses.binarycrossentropy(model(x), y)

opt_state = Flux.setup(Flux.Adam(1e-3), model)

# -----------------------------
# Data loader
# -----------------------------
train_loader = DataLoader((X_train, y_train), batchsize=100, shuffle=true)

# -----------------------------
# Training loop
# -----------------------------
epochs = 5

for epoch in 1:epochs
    println("Epoch ", epoch)
    for (x, y) in train_loader
        x = Float32.(x)
        y = Float32.(y)
        Flux.reset!(model)
        Flux.train!(loss, model, [(x, y)], opt_state)
    end
    l = loss(model, X_train[:, :, 1:200], y_train[:, 1:200])
    println("loss = ", round(l,digits=4))
end

println("Final training loss = ", loss(model, X_train[:, :, 1:200], y_train[:, 1:200]))

