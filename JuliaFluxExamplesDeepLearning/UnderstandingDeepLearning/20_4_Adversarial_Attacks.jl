using Flux
using Flux: DataLoader
using MLDatasets
using Images
using Statistics
using LinearAlgebra

# --- 1. Load & Prepare MNIST Data ---
function load_data()
    xtrain, ytrain = MNIST(split=:train)[:]
    # Flux expects (Height, Width, Channels, Batch)
    xtrain = reshape(xtrain, 28, 28, 1, :)
    ytrain = Flux.onehotbatch(ytrain, 0:9)
    return xtrain, ytrain
end

# --- 2. Simple MNIST Classifier ---
# Architecture similar to the notebook's PyTorch model
function build_model()
    return Chain(
        Conv((5, 5), 1=>10, relu),
        MaxPool((2, 2)),
        Conv((5, 5), 10=>20, relu),
        Dropout(0.5),
        MaxPool((2, 2)),
        Flux.flatten,
        Dense(320, 50, relu),
        Dense(50, 10),
        softmax
        )
end

# --- 3. Fast Gradient Sign Method (FGSM) ---
# This function perturbs an image to trick the model
function fgsm_attack(model, x, y_true, epsilon)
    # 1. Calculate the gradient of the loss with respect to the input x
    # Note: We differentiate with respect to 'x', not 'model'
    grads = Flux.gradient(x) do x_input
        loss = Flux.logitcrossentropy(model(x_input), y_true)
        return loss
    end

    # 2. Extract the sign of the gradient
    # grads[1] corresponds to the first argument of the gradient block (x)
    data_grad = sign.(grads[1])

    # 3. Create the perturbed image: x_adv = x + epsilon * sign(grad)
    x_perturbed = x .+ epsilon .* data_grad

    # 4. Clip the values to stay within the valid [0, 1] pixel range
    return clamp.(x_perturbed, 0f0, 1f0)
end

# --- 4. Training & Attack Workflow ---
function run_adversarial_experiment()
    X, Y = load_data()
    model = build_model()
    opt_state = Flux.setup(Adam(0.001f0), model)

    # Selecting one sample to attack
    x_sample = X[:, :, :, 1:1]  # Size: (28, 28, 1, 1)
    y_sample = Y[:, 1:1]

    println("Original Prediction: ", argmax(model(x_sample))[1] - 1)

    # --- Perform Attack ---
    epsilon = 0.2f0
    x_adv = fgsm_attack(model, x_sample, y_sample, epsilon)

    # --- Check Result ---
    y_adv_pred = model(x_adv)
    println("Perturbed Prediction: ", argmax(y_adv_pred)[1] - 1)
    println("Confidence in wrong class: ", maximum(y_adv_pred))

    return x_sample, x_adv
end

# Execute
x_orig, x_adv = run_adversarial_experiment()

