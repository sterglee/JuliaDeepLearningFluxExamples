using Flux
using Flux: onehotbatch, onecold, logitcrossentropy
using Statistics, Random
using Pickle

# -----------------------------
# 1. Load Data
# -----------------------------
data = open("mnist1d_data.pkl", "r") do io
    Pickle.load(io)
end

x_train = Float32.(data["x"])
y_train = Int.(data["y"])
x_test  = Float32.(data["x_test"])
y_test  = Int.(data["y_test"])

# Shift labels to 1–10 (Julia indexing)
y_train_shifted = y_train .+ 1
y_test_shifted  = y_test .+ 1

println("Loaded $(size(x_train, 1)) training examples.")

# -----------------------------
# 2. Augmentation
# -----------------------------
function augment(input_vector)
    shift = rand(0:length(input_vector)-1)
    data_out = circshift(input_vector, shift)

    scale_factor = rand() * 0.4f0 + 0.8f0   # [0.8, 1.2]
    return data_out .* scale_factor
end

# -----------------------------
# 3. Create Augmented Dataset
# -----------------------------
n_orig = size(x_train, 1)
n_aug_needed = 4000

aug_x = zeros(Float32, n_aug_needed, 40)
aug_y = zeros(Int, n_aug_needed)

for i in 1:n_aug_needed
    idx = rand(1:n_orig)
    aug_x[i, :] .= augment(x_train[idx, :])
    aug_y[i] = y_train_shifted[idx]
end

# Combine original + augmented
X_combined = vcat(x_train, aug_x)'   # (features, samples)
Y_combined = onehotbatch(vcat(y_train_shifted, aug_y), 1:10)

# Prepare test data (features, samples)
X_test = x_test'
Y_test = y_test

# -----------------------------
# 4. Model
# -----------------------------
model = Chain(
    Dense(40, 200, relu),
    Dense(200, 200, relu),
    Dense(200, 10)
)

opt = Flux.Momentum(0.05, 0.9)
opt_state = Flux.setup(opt, model)

loader = Flux.DataLoader((X_combined, Y_combined),
    batchsize = 100,
    shuffle = true)

# -----------------------------
# 5. Training Loop
# -----------------------------
epochs = 50

for epoch in 1:epochs
    for (batch_x, batch_y) in loader
        loss, grads = Flux.withgradient(model) do m
            ŷ = m(batch_x)
            logitcrossentropy(ŷ, batch_y)
        end
        Flux.update!(opt_state, model, grads)
    end

    # Test error
    y_pred = onecold(model(X_test)) .- 1
    test_error = 100 * mean(y_pred .!= Y_test)

    println("Epoch $epoch: Test Error = $(round(test_error, digits=2))%")
end
