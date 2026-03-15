using Flux, MLDatasets, Statistics, Printf, CUDA, Random
using Flux: DataLoader, onehotbatch, flatten, logitcrossentropy

# --- 1. Setup & Data Loading ---
const BATCH_SIZE = 128
const EPOCHS = 5
device = CUDA.functional() ? gpu : cpu

println("Loading and Splitting CIFAR-10...")
train_data = MLDatasets.CIFAR10(split=:train)
x_all, y_all = train_data[:]

# Διαχωρισμός (Validation Split)
n_samples = size(x_all, 4)
n_val = Int(floor(n_samples * 0.2))
n_train = n_samples - n_val

# Μετατροπή σε Float32 και μεταφορά ΣΕ ΟΛΟ το dataset στην GPU (αν χωράει)
# f32(x) είναι συντόμευση για Float32.(x)
x_train_gpu = f32(x_all[:, :, :, 1:n_train]) |> device
y_train_oh_gpu = onehotbatch(y_all[1:n_train], 0:9) |> device

x_val_gpu = f32(x_all[:, :, :, n_train+1:end]) |> device
y_val_oh_gpu = onehotbatch(y_all[n_train+1:end], 0:9) |> device

# Ο DataLoader τώρα δίνει batches που βρίσκονται ΗΔΗ στην GPU
train_loader = DataLoader((x_train_gpu, y_train_oh_gpu), batchsize=BATCH_SIZE, shuffle=true)

# --- 2. Model Definition ---
model = Chain(
    Conv((3, 3), 3 => 32, relu),
    MaxPool((2, 2)),
    Dropout(0.25),
    flatten,
    Dense(15 * 15 * 32 => 512, relu),
    Dropout(0.5),
    Dense(512 => 10)
) |> device

# --- 3. Optimized Functions ---
opt_state = Flux.setup(Flux.Adam(0.001), model)

# GPU-native accuracy (χωρίς cpu() calls μέσα στο loop)
function accuracy_gpu(m, x, y_oh)
    # Συγκρίνουμε τα indices των μέγιστων τιμών απευθείας στην GPU
    pred = argmax(m(x), dims=1)
    actual = argmax(y_oh, dims=1)
    return mean(pred .== actual)
end

# --- 4. Training Loop ---
println("Starting Training on $(device)...")
for epoch in 1:EPOCHS
    time_epoch = @elapsed begin
        for (batch_x, batch_y) in train_loader
            grads = Flux.gradient(model) do m
                logitcrossentropy(m(batch_x), batch_y)
            end
            Flux.update!(opt_state, model, grads[1])
        end
    end
    
    val_acc = accuracy_gpu(model, x_val_gpu, y_val_oh_gpu)
    @printf("Epoch %d: Val Acc: %.2f%% (Time: %.2f sec)\n", epoch, val_acc * 100, time_epoch)
end


