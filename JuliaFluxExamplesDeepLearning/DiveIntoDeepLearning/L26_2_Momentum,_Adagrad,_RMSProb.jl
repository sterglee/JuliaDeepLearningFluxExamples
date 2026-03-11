using Flux
using Statistics
using Plots
using CSV
using DataFrames
using Downloads
using MLUtils

# 1. Configuration & Constants
const WINDOW_SIZE = 12
const HIDDEN_SIZE = 100
const TRAIN_SIZE = 120
const EPOCHS = 150
const LR = 0.01

# 2. Data Pipeline
function get_data()
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv"
    df = CSV.read(Downloads.download(url), DataFrame)
    data = Float32.(df.passengers)
    
    # Normalize (-1 to 1)
    min_v, max_v = minimum(data), maximum(data)
    norm_data = 2 .* (data .- min_v) ./ (max_v - min_v) .- 1
    
    # Create sequences: (Feature, Batch)
    features = []
    labels = []
    for i in 1:(length(norm_data) - WINDOW_SIZE)
        push!(features, norm_data[i : i + WINDOW_SIZE - 1])
        push!(labels, norm_data[i + WINDOW_SIZE])
    end
    
    X = hcat(features...) # (12, 132)
    Y = Float32.(labels)'  # (1, 132)
    
    return X, Y, min_v, max_v
end

# 3. Model Architecture
function build_model()
    # Modern Flux.Chain with LSTM and a Linear (Dense) output
    return Chain(
        LSTM(WINDOW_SIZE => HIDDEN_SIZE),
        Dense(HIDDEN_SIZE => 1)
    )
end

# 4. Training Function
function train_model!(model, x_train, y_train, x_test, y_test)
    # Define Loss (explicitly resetting state for time-series consistency)
    function loss_fn(m, x, y)
        Flux.reset!(m)
        return Flux.mse(m(x), y)
    end

    # Set up optimizer (RMSProp as explored in the previous notebook)
    opt_state = Flux.setup(Flux.RMSProp(LR, 0.9), model)
    
    train_log, test_log = [], []

    println("Training starting...")
    for epoch in 1:EPOCHS
        grads = Flux.gradient(model) do m
            loss_fn(m, x_train, y_train)
        end
        Flux.update!(opt_state, model, grads[1])
        
        # Logging
        push!(train_log, loss_fn(model, x_train, y_train))
        push!(test_log, loss_fn(model, x_test, y_test))
        
        if epoch % 50 == 0
            @info "Epoch $epoch" Train_Loss=train_log[end] Test_Loss=test_log[end]
        end
    end
    return train_log, test_log
end

# 5. Main Execution Block
function main()
    # Load and Split
    X, Y, min_v, max_v = get_data()
    x_train, y_train = X[:, 1:TRAIN_SIZE], Y[:, 1:TRAIN_SIZE]
    x_test, y_test = X[:, TRAIN_SIZE+1:end], Y[:, TRAIN_SIZE+1:end]

    # Initialize
    model = build_model()

    # Train
    train_hist, test_hist = train_model!(model, x_train, y_train, x_test, y_test)

    # Predict & Denormalize
    Flux.reset!(model)
    preds = model(x_test)
    denorm(x) = (x .+ 1) .* (max_v - min_v) ./ 2 .+ min_v
    actual_preds = denorm(preds)

    # Plotting
    p1 = plot(train_hist, label="Train", title="Loss History", ylabel="MSE")
    plot!(p1, test_hist, label="Test")

    full_data = denorm(vcat(y_train', y_test'))
    p2 = plot(full_data, label="Actual", title="Forecast", xlabel="Months", ylabel="Passengers")
    test_idx = (TRAIN_SIZE + 1):length(full_data)
    plot!(p2, test_idx, actual_preds', label="Predicted", lw=2, color=:red)

    display(plot(p1, p2, layout=(2,1), size=(800, 600)))
    println("Main routine complete.")
end

# Execute
main()

