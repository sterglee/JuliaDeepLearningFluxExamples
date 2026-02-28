using Flux
using Random
using Statistics
using LinearAlgebra

# --- 1. Environment Setup (Based on Notebook 19.4) ---
# Translating the 4x4 grid logic to 1-based Julia indexing
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

    # Matching reward structure from the notebook:
    # Holes (indices 10, 11, 15) and Goal (index 16)
    rewards[[10, 11, 15]] .= -2.0f0
    rewards[16] = 3.0f0

    terminal_states = [10, 11, 15, 16]
    return GridWorld(rows, cols, n_states, 4, terminal_states, rewards)
end

# --- 2. Neural Network Model (Flux) ---
# A modern Chain to approximate the Q-function
function create_model(n_states, n_actions)
    return Chain(
        Dense(n_states, 64, relu),
        Dense(64, 32, relu),
        Dense(32, n_actions)
        )
end

# Helper to convert state index to one-hot vector for Flux
one_hot_state(s, n) = Float32.(1:n .== s)

# --- 3. Exploration Strategy ---
function get_action(model, state, epsilon, n_actions)
    if rand() < epsilon
        return rand(1:n_actions) # Explore
    else
        q_values = model(one_hot_state(state, 16))
        return argmax(q_values)    # Exploit
    end
end

# --- 4. Modern Training Logic (SARSA / Q-Learning) ---
# Replaces the 'get_one_episode' tabular update from the notebook
function train!(world::GridWorld; n_episodes=500, gamma=0.95f0, lr=0.001f0)
    model = create_model(world.n_states, world.n_actions)
    opt_state = Flux.setup(Adam(lr), model)

    epsilon = 0.2

    for ep in 1:n_episodes
        state = 1 # Start state

        while !(state in world.terminal_states)
            action = get_action(model, state, epsilon, world.n_actions)

            # Simple grid movement logic
            next_state = state
            if action == 1 && (state % world.cols != 0) next_state += 1        # Right
                elseif action == 2 && (state % world.cols != 1) next_state -= 1   # Left
                elseif action == 3 && (state <= world.n_states - world.cols) next_state += world.cols # Down
                elseif action == 4 && (state > world.cols) next_state -= world.cols # Up
            end

            reward = world.rewards[next_state]

            # Flux Gradient Update: Minimize TD Error
            grads = Flux.gradient(model) do m
                q_values = m(one_hot_state(state, world.n_states))

                # Q-Learning Target: r + gamma * max(Q(s', a'))
                # For SARSA, replace maximum() with the Q-value of the next selected action
                target_q = reward + (next_state in world.terminal_states ?
                                     0f0 :
                                         gamma * maximum(m(one_hot_state(next_state, world.n_states))))

                Flux.mse(q_values[action], target_q)
            end

            Flux.update!(opt_state, model, grads[1])
            state = next_state
        end

        if ep % 100 == 0
            println("Completed Episode $ep")
        end
    end
    return model
end

# --- 5. Execution ---
world = init_gridworld()
println("Training TD Agent...")
trained_model = train!(world)

# Visualizing the final learned policy
println("\nLearned Policy (1=Right, 2=Left, 3=Down, 4=Up):")
for s in 1:world.n_states
    q = trained_model(one_hot_state(s, world.n_states))
    print(argmax(q), " ")
    s % 4 == 0 && println()
end

