using Flux
using Statistics
using Plots
using Random
using HTTP
using JSON

# ==========================================
# 1. DATA LOADING SECTION (Fixes UndefVarError)
# ==========================================
function load_mnist_1d()
    url = "https://raw.githubusercontent.com/greydanus/mnist1d/master/mnist1d_data.json"
    println("Downloading MNIST-1D data from GitHub...")
    
    response = HTTP.get(url)
    data = JSON.parse(String(response.body))
    
    # Extract and format: Flux expects (Features x Samples)
    x_train = Float32.(reduce(hcat, data["x"]))
    y_train = Int.(data["y"])
    x_test  = Float32.(reduce(hcat, data["x_test"]))
    y_test  = Int.(data["y_test"])
    
    return x_train, y_train, x_test, y_test
end

# Load the variables into Main
x_train, y_train, x_test, y_test = load_mnist_1d()

# Now we can safely one-hot encode
y_train_oh = Flux.onehotbatch(y_train, 0:9)
y_test_oh  = Flux.onehotbatch(y_test, 0:9)

println("Success! Data loaded. Training samples: ", size(x_train, 2))

# ==========================================
# 2. MODEL & EXPERIMENT SECTION
# ==========================================
# We will test various hidden layer sizes to observe Double Descent
hidden_configs = [2, 6, 10, 14, 18, 22, 30, 50, 80, 150]
train_errors = []
test_errors = []

println("Starting Double Descent Sweep...")

for n_h in hidden_configs
    # Define a simple MLP
    model = Chain(
        Dense(40 => n_h, relu), 
        Dense(n_h => 10)
    )
    
    opt_state = Flux.setup(Flux.Adam(0.001), model)
    loader = Flux.DataLoader((x_train, y_train_oh), batchsize=100, shuffle=true)
    
    # Train for enough epochs to reach the "interpolation" regime
    for epoch in 1:100
        Flux.train!(m -> Flux.logitcrossentropy(m(x_train), y_train_oh), model, loader, opt_state)
    end
    
    # Calculate Accuracy -> Error Rate
    train_acc = mean(Flux.onecold(model(x_train), 0:9) .== y_train)
    test_acc  = mean(Flux.onecold(model(x_test), 0:9) .== y_test)
    
    push!(train_errors, (1.0 - train_acc) * 100)
    push!(test_errors, (1.0 - test_acc) * 100)
    
    println("Hidden: $n_h | Train Err: $(round(train_errors[end], digits=1))% | Test Err: $(round(test_errors[end], digits=1))%")
end

# ==========================================
# 3. VISUALIZATION
# ==========================================


p = plot(hidden_configs, train_errors, label="Train Error", color=:red, lw=2, marker=:circle)
plot!(p, hidden_configs, test_errors, label="Test Error", color=:blue, lw=2, marker=:circle)
vline!([16], label="Approx. Interpolation Threshold", linestyle=:dash, color=:green)
xlabel!("Model Capacity (Hidden Units)")
ylabel!("Error Rate (%)")
title!("Double Descent on MNIST-1D")
display(p)

