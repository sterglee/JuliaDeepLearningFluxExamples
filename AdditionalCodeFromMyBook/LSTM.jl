using Flux, Plots, Statistics

# 1. Δημιουργία Δεδομένων
t = 0:0.1:100
raw_data = Float32.(sin.(t) .+ 0.5f0 .* sin.(2t)) 

function make_data(data, window)
    x = [data[i:i+window-1] for i in 1:length(data)-window]
    y = [data[i+window] for i in 1:length(data)-window]
    return x, y
end

window_size = 15
X_raw, Y_raw = make_data(raw_data, window_size)

# Μετατροπή για Flux RNN: Vector of Vectors of Matrices
X = [[reshape([x_val], 1, 1) for x_val in x_seq] for x_seq in X_raw]
Y = Y_raw

# 2. Αρχιτεκτονική Μοντέλου
model = Chain(
    LSTM(1 => 32),
    Dropout(0.1),
    Dense(32 => 1)
)

# 3. Διορθωμένη Συνάρτηση Απώλειας (Προσθήκη του ορίσματος 'm')
function loss(m, x_seq, y_target)
    # Επεξεργασία ακολουθίας
    out = [m(xt) for xt in x_seq][end]
    Flux.reset!(m) 
    return Flux.mse(out, y_target)
end

# 4. Optimizer setup
opt_state = Flux.setup(Adam(0.005), model)

# 5. Βρόχος Εκπαίδευσης
println("Έναρξη εκπαίδευσης...")
for epoch in 1:100
    # Το Flux.train! περνάει αυτόματα το (model, x, y) στη συνάρτηση loss
    Flux.train!(loss, model, zip(X, Y), opt_state)
    
    if epoch % 20 == 0
        l = mean(loss(model, x, y) for (x, y) in zip(X[1:10], Y[1:10]))
        println("Epoch $epoch | Loss: $l")
    end
end

# 6. Πρόβλεψη
Flux.testmode!(model)
predictions = []
for x_seq in X
    push!(predictions, [model(xt) for xt in x_seq][end][1])
    Flux.reset!(model)
end

plot(raw_data[window_size+1:end], label="Πραγματικά", lw=1.5)
plot!(predictions, label="Πρόβλεψη LSTM", ls=:dash, lw=1.5)

