using Flux
using Statistics
using Plots
using CSV
using DataFrames
using Downloads
using MLUtils

# --- 1. CONFIGURATION ---
const TAU = 12          # Markov window (embedding size)
const HIDDEN_DIM = 100
const EPOCHS = 150
const LEARNING_RATE = 0.01

# --- 2. DATA UTILITIES ---
function prepare_flight_data()
    # Direct download to avoid Seaborn dependency
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv"
    df = CSV.read(Downloads.download(url), DataFrame)
    raw_data = Float32.(df.passengers)
    
    # Min-Max Normalization (-1 to 1)
    min_v, max_v = minimum(raw_data), maximum(raw_data)
    norm_data = 2 .* (raw_data .- min_v) ./ (max_v - min_v) .- 1
    
    # Create Autoregressive Sequences (The Markov Assumption)
    features, labels = [], []
    for i in 1:(length(norm_data) - TAU)
        push!(features, norm_data[i : i + TAU - 1])
        push!(labels, norm_data[i + TAU])
    end
    
    # Flux layout: (Features, Samples)
    X = hcat(features...)
    Y = reshape(Float32.(labels), 1, :)
    
    return (X, Y), (min_v, max_v)
end

# --- 3. MODEL BUILDER ---
function build_recurrent_model(type=:LSTM)
    if type == :LSTM
        layer = LSTM(TAU => HIDDEN_DIM)
    elseif type == :GRU
        layer = GRU(TAU => HIDDEN_DIM)
    else
        layer = RNN(TAU => HIDDEN_DIM)
    end
    
    return Chain(layer, Dense(HIDDEN_DIM => 1))
end

# --- 4. TRAINING ENGINE ---
function train_engine!(model, data; epochs=EPOCHS)
    x_train, y_train = data
    
    # Modern Flux Optimizer setup (RMSProp handles the "valleys" in time series well)
    opt_state = Flux.setup(Flux.RMSProp(LEARNING_RATE, 0.9), model)
    
    loss_history = []

    println("Training on $(size(x_train, 2)) samples...")
    for epoch in 1:epochs
        # Flux recurrent layers are stateful; reset before each pass
        Flux.reset!(model)
        
        # Calculate gradients and update
        grads = Flux.gradient(model) do m
            Flux.mse(m(x_train), y_train)
        end
        Flux.update!(opt_state, model, grads[1])
        
        # Track Progress
        current_loss = Flux.mse(model(x_train), y_train)
        push!(loss_history, current_loss)
        
        if epoch % 50 == 0
            println("Epoch $epoch | MSE: $(round(current_loss, digits=6))")
        end
    end
    return loss_history
end

# --- 5. MAIN EXECUTION ---
function main()
    # 1. Setup Data
    (X, Y), (min_val, max_val) = prepare_flight_data()
    
    # Train/Test Split (First 120 months for training)
    train_split = 120
    train_data = (X[:, 1:train_split], Y[:, 1:train_split])
    test_x = X[:, train_split+1:end]
    test_y = Y[:, train_split+1:end]

    # 2. Build and Train
    # Switch between :LSTM, :GRU, or :RNN easily here
    model = build_recurrent_model(:LSTM)
    history = train_engine!(model, train_data)

    # 3. Forecast / Inference
    Flux.reset!(model)
    preds_norm = model(test_x)
    
    # Denormalize for plotting
    denorm(val) = (val .+ 1) .* (max_val - min_val) ./ 2 .+ min_val
    actual_preds = denorm(preds_norm)
    actual_truth = denorm(test_y)

    # 4. Visualization
    p1 = plot(history, title="Training Loss", xlabel="Epoch", ylabel="MSE", legend=false)
    
    p2 = plot(actual_truth', label="Ground Truth", title="Flight Forecast", lw=2)
    plot!(p2, actual_preds', label="LSTM Prediction", color=:red, linestyle=:dash, lw=2)
    
    display(plot(p1, p2, layout=(2,1), size=(800, 600)))
    println("Main complete. Plots generated.")
end

# Call the entry point
main()

