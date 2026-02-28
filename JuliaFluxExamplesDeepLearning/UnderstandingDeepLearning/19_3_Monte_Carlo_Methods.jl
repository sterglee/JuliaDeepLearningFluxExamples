using Flux
using Random
using Statistics
using LinearAlgebra

# --- 1. Environment Setup (Reflecting Notebook 19.3) ---
struct GridWorld
    rows::Int
    cols::Int
    n_states::Int
    n_actions::Int
    terminal_states::Vector{Int}
    rewards::Vector{Float32}
end

function init_gridworld()
    rows, cols = 4, 4
    n_states = rows * cols
    rewards = zeros(Float32, n_states)

    # Matching the "Holes" and "Fish" logic from the notebook
    # Julia 1-indexed: 10, 11, 15 are holes (-2), 16 is goal (+3)
    rewards[[10, 11, 15]] .= -2.0f0
    rewards[16] = 3.0f0

    terminal_states = [10, 11, 15, 16]
    return GridWorld(rows, cols, n_states, 4, terminal_states, rewards)
end

# --- 2. Neural Network Model (Value Approximator) ---
function create_model(n_states, n_actions)
    return Chain(
        Dense(n_states, 64, relu),
        Dense(64, 32, relu),
        Dense(32, n_actions)
        )
end

one_hot_state(s, n) = Float32.(1:n .== s)

# --- 3. Monte Carlo Rollout Logic ---
# Replaces 'get_one_episode' from the notebook
function run_episode(model, world, epsilon)
    states, actions, rewards = Int[], Int[], Float32[]
    s = 1 # Start state

    while !(s in world.terminal_states)
        # Epsilon-greedy action selection
        a = if rand() < epsilon
            rand(1:world.n_actions)
        else
            argmax(model(one_hot_state(s, world.n_states)))
        end

        # Simplified transition logic
        next_s = s
        if a == 1 && (s % world.cols != 0) next_s += 1        # Right
            elseif a == 2 && (s % world.cols != 1) next_s -= 1   # Left
            elseif a == 3 && (s <= world.n_states - world.cols) next_s += world.cols # Down
            elseif a == 4 && (s > world.cols) next_s -= world.cols # Up
        end

        push!(states, s)
        push!(actions, a)
        push!(rewards, world.rewards[next_s])
        s = next_s

        # Prevent infinite loops in early training
        length(states) > 50 && break
    end
    return states, actions, rewards
end

# --- 4. Modern Flux Training Logic (Monte Carlo Policy Update) ---
# Translates the Monte Carlo update logic into a gradient-based approach
function train_mc!(world::GridWorld; n_episodes=500, gamma=0.9f0, lr=0.001f0)
    model = create_model(world.n_states, world.n_actions)
    opt_state = Flux.setup(Adam(lr), model)
    epsilon = 0.3

    for ep in 1:n_episodes
        # 1. Generate an episode following the current policy
        states, actions, rewards = run_episode(model, world, epsilon)

        # 2. Compute discounted returns (G_t) - Core Monte Carlo logic
        returns = zeros(Float32, length(rewards))
        G = 0.0f0
        for t in length(rewards):-1:1
            G = rewards[t] + gamma * G
            returns[t] = G
        end

        # 3. Policy Update: Move Q(s,a) toward the observed return G
        grads = Flux.gradient(model) do m
            loss = 0.0f0
            for t in 1:length(states)
                q_values = m(one_hot_state(states[t], world.n_states))
                # MSE between current Q-prediction and the Monte Carlo return
                loss += Flux.mse(q_values[actions[t]], returns[t])
            end
            return loss / length(states)
        end

        Flux.update!(opt_state, model, grads[1])

        # Anneal epsilon
        epsilon = max(0.05, epsilon * 0.995)
    end
    return model
end

# --- 5. Execution ---
world = init_gridworld()
println("Training with Monte Carlo Methods...")
trained_model = train_mc!(world)

# Output final learned policy
println("\nFinal Policy (1=Right, 2=Left, 3=Down, 4=Up):")
for s in 1:world.n_states
    q = trained_model(one_hot_state(s, world.n_states))
    print(argmax(q), " ")
    s % world.cols == 0 && println()
end

