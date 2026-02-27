using Flux
using Statistics
using Plots
using Random

# ---------------------------------------------------------
# 1. Data Setup
# ---------------------------------------------------------
true_func(x) = sin(x * 3) + 0.5f0 * x
x_train = Float32.(collect(-2:0.4:2)')
y_train = true_func.(x_train) .+ 0.2f0 * randn(Float32, size(x_train))

# ---------------------------------------------------------
# 2. Training the Ensemble
# ---------------------------------------------------------
n_models = 10
ensemble = []

# Define the loss function clearly
# It MUST take the model 'm' as the first argument for Flux to differentiate it
loss_fn(m, x, y) = Flux.mse(m(x), y)

println("Training ensemble...")

for i in 1:n_models
    model = Chain(Dense(1 => 20, relu), Dense(20 => 1))
    opt_state = Flux.setup(Flux.Adam(0.01), model)
    
    for epoch in 1:200
        # FIX: The "do" block or arrow function must pass the model 'm' 
        # to the loss function so Zygote knows what to differentiate.
        val, grads = Flux.withgradient(model) do m
            loss_fn(m, x_train, y_train)
        end
        Flux.update!(opt_state, model, grads[1])
    end
    push!(ensemble, model)
end

# ---------------------------------------------------------
# 3. Aggregating Predictions
# ---------------------------------------------------------
x_test = Float32.(collect(-2.5:0.01:2.5)')
all_preds = hcat([vec(m(x_test)) for m in ensemble]...)
ensemble_mean = mean(all_preds, dims=2)

# ---------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------


p = plot(x_test', true_func.(x_test)', label="True Function", color=:black, lw=2, ls=:dash)
for i in 1:n_models
    plot!(p, x_test', all_preds[:, i], color=:blue, alpha=0.2, label=i==1 ? "Individual Models" : "")
end
plot!(p, x_test', ensemble_mean, label="Ensemble Mean", color=:red, lw=3)
scatter!(p, x_train', y_train', label="Training Data", color=:green)

display(p)

