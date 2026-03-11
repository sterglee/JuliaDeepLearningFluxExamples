using Flux
using Statistics
using Plots
using CSV
using DataFrames
using Downloads
using LinearAlgebra

# 1. Load Data without Seaborn
# Downloading the flights dataset directly from the source
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv"
df = CSV.read(Downloads.download(url), DataFrame)
flight_data = Float32.(df.passengers)

# Plot initial data
plot(flight_data, title="Monthly Air Passengers", ylabel="# Passengers", xlabel="Time", label=false)

# 2. Data Preprocessing
# Normalize between -1 and 1
min_val, max_val = minimum(flight_data), maximum(flight_data)
normalized_data = 2 .* (flight_data .- min_val) ./ (max_val - min_val) .- 1

# Create sequences (Windowing)
# PyTorch logic: 12 previous months predict the 13th
window_size = 12
train_size = 120
dataset_size = length(normalized_data)

features = []
labels = []
for i in 1:(dataset_size - window_size)
    push!(features, normalized_data[i : i + window_size - 1])
    push!(labels, normalized_data[i + window_size])
end

# Format for Flux: (Features, Batch)
# Note: The PyTorch notebook treats the window as the input dimension (12 features) 
# and processes it in a single time step.
X = hcat(features...) # Shape: (12, 132)
Y = Float32.(labels)'  # Shape: (1, 132)

# Split into train and test
x_train, y_train = X[:, 1:train_size], Y[:, 1:train_size]
x_test, y_test = X[:, train_size+1:end], Y[:, train_size+1:end]

# 3. Model Definition (LSTM)
# Matching PyTorch: LSTM(12, 100) -> Dense(100, 1)
model = Chain(
    LSTM(window_size => 100),
    Dense(100 => 1)
)

# 4. Training Configuration
loss(m, x, y) = begin
    Flux.reset!(m) # PyTorch notebook resets state implicitly per call
    Flux.mse(m(x), y)
end

opt_state = Flux.setup(Flux.Adam(0.01), model)

# 5. Training Loop
epochs = 150
train_losses = []
test_losses = []

println("Starting training...")
for epoch in 1:epochs
    # Calculate gradients and update
    grads = Flux.gradient(model) do m
        loss(m, x_train, y_train)
    end
    Flux.update!(opt_state, model, grads[1])
    
    # Track performance
    l_train = loss(model, x_train, y_train)
    l_test = loss(model, x_test, y_test)
    push!(train_losses, l_train)
    push!(test_losses, l_test)
    
    if epoch % 25 == 0
        println("Epoch $epoch: Train Loss = $l_train, Test Loss = $l_test")
    end
end

# 6. Evaluation & Visualization
# Generate predictions
Flux.reset!(model)
train_preds = model(x_train)
test_preds = model(x_test)

# Denormalize
denormalize(x) = (x .+ 1) .* (max_val - min_val) ./ 2 .+ min_val
actual_preds = denormalize(test_preds)

# Plot Losses
p1 = plot(train_losses, label="Train Loss", title="Training Progress")
plot!(p1, test_losses, label="Test Loss")

# Plot Predictions
p2 = plot(flight_data, label="Actual Data", title="Flight Passenger Prediction")
test_range = (train_size + window_size + 1):dataset_size
plot!(p2, test_range, actual_preds', label="Predicted", color=:red)

display(plot(p1, p2, layout=(2,1), size=(800, 600)))

# 7. Switching to GRU or RNN (Equivalent to the last cells in the notebook)
# To use GRU, simply replace the model:
# model = Chain(GRU(window_size => 100), Dense(100 => 1))
# To use RNN:
# model = Chain(RNN(window_size => 100), Dense(100 => 1))

