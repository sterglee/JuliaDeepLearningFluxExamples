using Flux
using Random
using Statistics
using LinearAlgebra
using BenchmarkTools
using MKL
# --- 1. Environment Setup (From Notebook 19.1) ---
# Transitioning from Python's 0-indexed logic to Julia's 1-indexed logic
struct GridWorld
    rows::Int
    cols::Int
    n_states::Int
    n_actions::Int
    terminal_states::Vector{Int}
    rewards::Vector{Float32}
end

function init_gridworld(rows=4, cols=4)
    n_states = rows * cols
    rewards = zeros(Float32, n_states)
    rewards[end] = 1.0f0  # Goal state reward, similar to Figure 19.2
    terminal_states = [n_states]
    return GridWorld(rows, cols, n_states, 4, terminal_states, rewards)
end

# --- 2. Neural Network Model (Flux) ---
# A modern approach to approximating value functions
function create_model(n_states, n_actions)
    return Chain(
        Dense(n_states, 32, relu),
        Dense(32, 32, relu),
        Dense(32, n_actions)
        )
end

# Helper: Convert state index to one-hot vector for Flux
one_hot_state(s, n) = Float32.(1:n .== s)

# --- 3. Exploration Strategy ---
function get_action(model, state, epsilon, world)
    if rand() < epsilon
        return rand(1:world.n_actions) # Random exploration
    else
        q_values = model(one_hot_state(state, world.n_states))
        return argmax(q_values) # Greedy action
    end
end

# --- 4. Training Logic (Deep TD Learning) ---
# Modernizing the Temporal Difference methods described in the MDP notebook
function train!(world, n_episodes=200, gamma=0.95f0, lr=0.01f0)
    model = create_model(world.n_states, world.n_actions)
    opt_state = Flux.setup(Adam(lr), model)
    epsilon = 0.2

    for ep in 1:n_episodes
        state = 1 # Initial state
        total_reward = 0

        while !(state in world.terminal_states)
            action = get_action(model, state, epsilon, world)

            # Simple grid movement logic (Deterministic for this example)
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
                target_q = reward + (next_state in world.terminal_states ? 0f0 :
                    gamma * maximum(m(one_hot_state(next_state, world.n_states))))

                    Flux.mse(q_values[action], target_q)
            end

            Flux.update!(opt_state, model, grads[1])
            state = next_state
            total_reward += reward
        end
    end
    return model
end

# --- 5. Execution ---
world = init_gridworld()
@time trained_model = train!(world)


# Visualizing the final learned policy
println("Learned Policy (1=Right, 2=Left, 3=Down, 4=Up):")
for s in 1:world.n_states
    q = trained_model(one_hot_state(s, world.n_states))
    print(argmax(q), " ")
    s % world.cols == 0 && println()
end

