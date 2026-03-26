using Flux
using Flux: onehotbatch, onecold, crossentropy, train!
using MLDatasets
using Random
using Statistics # Για τη συνάρτηση mean

# --- 1. Φόρτωση και προετοιμασία δεδομένων ---
# Σημείωση: Το MNIST() επιστρέφει αντικείμενο, χρησιμοποιούμε features() και targets()
train_data = MNIST(split=:train)
test_data  = MNIST(split=:test)

train_x, train_y = train_data.features, train_data.targets
test_x,  test_y  = test_data.features,  test_data.targets

# Reshape για CNN: (Width, Height, Channels, BatchSize)
# Το MNIST του MLDatasets είναι ήδη 28x28x60000, προσθέτουμε το κανάλι (1)
train_x = reshape(Float32.(train_x), 28, 28, 1, :)
test_x  = reshape(Float32.(test_x), 28, 28, 1, :)

# One-hot encoding
labels = 0:9
train_y_oh = onehotbatch(train_y, labels)
test_y_oh  = onehotbatch(test_y, labels)

# Δημιουργία DataLoader (Πολύ πιο αποδοτικό από το χειροκίνητο slicing)
batchsize = 128
train_loader = Flux.DataLoader((train_x, train_y_oh), batchsize=batchsize, shuffle=true)

# --- 2. Ορισμός CNN μοντέλου ---
# Προσοχή στο μέγεθος του Dense: Μετά από 2 Conv και 2 MaxPool, το 28x28 γίνεται 4x4
model = Chain(
    Conv((3,3), 1=>16, relu, pad=SamePad()),
    MaxPool((2,2)),
    Conv((3,3), 16=>32, relu, pad=SamePad()),
    MaxPool((2,2)),
    Flux.flatten,
    Dense(7*7*32, 10) # 28 -> 14 -> 7 (με SamePad)
    # Δεν βάζουμε softmax εδώ αν χρησιμοποιούμε logitcrossentropy (πιο σταθερό)
    )

# --- 3. Ορισμός Loss και Optimizer ---
# Χρήση logitcrossentropy για καλύτερη αριθμητική ευστάθεια
loss(m, x, y) = Flux.logitcrossentropy(m(x), y)
opt_state = Flux.setup(Flux.Adam(0.001), model) # Νέος τρόπος ορισμού Optimizer

# --- 4. Training loop ---
for epoch in 1:3
    println("Epoch $epoch")

    for (xbatch, ybatch) in train_loader
        # Υπολογισμός gradient και update
        grads = Flux.gradient(m -> loss(m, xbatch, ybatch), model)
        Flux.update!(opt_state, model, grads[1])
    end

    # Υπολογισμός ακρίβειας στο test set
    pred = model(test_x)
    acc = mean(onecold(pred, labels) .== onecold(test_y_oh, labels))
    println("Test accuracy: ", round(acc*100, digits=2), "%")
end

