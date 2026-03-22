using Flux
using Statistics
using Random

# 1. Environment: Calculating "True" Gini Impurity
function calculate_gini(counts::Dict{Symbol, Int})
    total = sum(values(counts))
    total == 0 && return 0.0
    # Gini = 1 - Σ(p_i^2)
    sum_sq_proportions = sum((v / total)^2 for v in values(counts))
        return 1.0 - sum_sq_proportions
    end

    # 2. Rule-Based Reward Function
    # Incentivizes correctness and structure (DeepSeek-R1-Zero Innovation)
    function reward_fn(predicted_gini::Float32, true_gini::Float64; tol=0.1)
        error = abs(predicted_gini - true_gini)
        # Accuracy Reward: Higher reward for closer predictions
        return error < tol ? 1.0f0 : -0.5f0
    end

    # 3. Policy Model (The "Reasoner")
    # In this simulation, the model "reasons" by predicting the Gini value
    struct ReasoningModel
        network
        noise_level::Ref{Float32}
    end

    Flux.@functor ReasoningModel (network,)

    function ReasoningModel(in_dim::Int)
        net = Chain(Dense(in_dim => 16, relu), Dense(16 => 1, sigmoid))
        return ReasoningModel(net, Ref(0.2f0))
    end

    # 4. RL Training Loop (The "Aha Moment" Simulation)
    function train_r1_zero!(model, counts::Dict{Symbol, Int}, epochs=50)
        true_gini = calculate_gini(counts)
        # Convert counts to a feature vector
        input_vec = Float32[counts[:A], counts[:B]]

        opt_state = Flux.setup(Adam(0.01), model)

        println("True Gini: ", round(true_gini, digits=2))
        println("Starting RL Updates (Incentivizing Accuracy)...")

        for epoch in 1:epochs
            # 1. Generate Prediction (with exploration noise)
            pred = model.network(input_vec)[1] + (randn(Float32) * model.noise_level[])

            # 2. Calculate Reward
            reward = reward_fn(pred, true_gini)

            # 3. Policy Update: Update weights based on reward
            # We use the reward to scale the gradient (Simple Policy Gradient)
            grads = Flux.gradient(model) do m
                # Minimize MSE, scaled by the negative reward
                loss = (m.network(input_vec)[1] - Float32(true_gini))^2
                return loss * (1.0f0 - reward)
            end

            Flux.update!(opt_state, model, grads[1])

            # 4. Anneal Noise (Self-Correction Emergence)
            if reward > 0
                model.noise_level[] *= 0.95f0
            end

            if epoch % 10 == 0
                current_pred = model.network(input_vec)[1]
                println("Epoch $epoch | Pred: $(round(current_pred, digits=3)) | Reward: $reward")
            end
        end
    end

    # 5. Execution
    Random.seed!(42)
    node_counts = Dict(:A => 4, :B => 6)
    r1_model = ReasoningModel(2)

    train_r1_zero!(r1_model, node_counts)

