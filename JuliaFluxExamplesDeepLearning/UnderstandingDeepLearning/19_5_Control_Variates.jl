using Flux
using Distributions
using Random
using Statistics
using LinearAlgebra

# --- 1. Environment Setup ---
# A simple 1D GridWorld (States 1 to 10)
struct SimpleEnv
    n_states::Int
    n_actions::Int
    goal_state::Int
end

# --- 2. Dual-Head Model Setup ---
# We use a Policy network to choose actions and a Value network as a Control Variate
function create_agent(n_states, n_actions)
    # Policy Network: Outputs probability distribution over actions
    policy = Chain(
        Dense(n_states, 32, relu),
        Dense(32, n_actions),
        softmax
        )

    # Value Network: Acts as the Baseline (Control Variate)
    baseline = Chain(
        Dense(n_states, 32, relu),
        Dense(32, 1)
        )

    return policy, baseline
end

# Helper to convert state index to one-hot vector
one_hot(s, n) = Float32.(1:n .== s)

# --- 3. The Corrected Training Function ---
function train_with_control_variate!(env::SimpleEnv; n_episodes=500, gamma=0.99f0, lr=0.001f0)
    policy, baseline = create_agent(env.n_states, env.n_actions)

    # Setup Optimizers
    opt_p = Flux.setup(Adam(lr), policy)
    opt_b = Flux.setup(Adam(lr), baseline)

    for ep in 1:n_episodes
        # --- Episode Rollout ---
        states, actions, rewards = [], [], []
        s = 1
        while s != env.goal_state && length(states) < 50
            probs = policy(one_hot(s, env.n_states))

            # FIX: Ensure probs sum to 1 and use Categorical from Distributions.jl
            dist = Categorical(probs ./ sum(probs))
            a = rand(dist)

            next_s = (a == 2) ? min(env.n_states, s + 1) : max(1, s - 1)
            rew = (next_s == env.goal_state) ? 10.0f0 : -0.1f0

            push!(states, s); push!(actions, a); push!(rewards, rew)
            s = next_s
        end

        # --- Compute Discounted Returns (G_t) ---
        G = 0.0f0
        returns = zeros(Float32, length(rewards))
        for t in length(rewards):-1:1
            G = rewards[t] + gamma * G
            returns[t] = G
        end

        # --- Gradient Update ---
        # We use Flux.withgradient to get both the loss value and the gradients

    loss_val, grads = Flux.withgradient(policy, baseline) do p, b
            total_loss_p = 0.0f0
            total_loss_b = 0.0f0

            for t in 1:length(states)
                s_vec = one_hot(states[t], env.n_states)

                # Control Variate Logic:
                # Advantage = Return - Baseline(s)
                val_pred = b(s_vec)[1]
                advantage = returns[t] - val_pred

                # Policy Gradient Loss
                total_loss_p -= log(p(s_vec)[actions[t]] + 1e-8) * advantage

                # Baseline Loss (MSE)
                total_loss_b += (val_pred - returns[t])^2
            end
            return (total_loss_p + total_loss_b) / length(states)
        end
    # 2. Update using the individual gradients stored inside 'grads'
    # grads[1] corresponds to policy, grads[2] corresponds to baseline
    Flux.update!(opt_p, policy, grads[1])
    Flux.update!(opt_b, baseline, grads[2])

        if ep % 100 == 0
            println("Episode $ep: Steps = $(length(states))")
        end
    end
    return policy, baseline
end

# --- 4. Run Training ---
env = SimpleEnv(10, 2, 10)
println("Starting Training...")
final_policy, final_baseline = train_with_control_variate!(env)

# Test the policy
test_s = one_hot(1, 10)
println("\nAction probabilities for start state: ", final_policy(test_s))


