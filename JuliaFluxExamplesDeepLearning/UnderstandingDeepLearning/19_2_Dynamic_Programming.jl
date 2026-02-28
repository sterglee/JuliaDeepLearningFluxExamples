using Flux
using Random
using Statistics
using LinearAlgebra

# --- 1. Environment & Grid Setup ---
# Translating the 4x4 GridWorld logic from Notebook 19.2
# Note: Julia uses 1-based indexing. State 0 in Python becomes State 1 here.
struct GridWorldMDP
    rows::Int
    cols::Int
    n_states::Int
    n_actions::Int
    rewards::Vector{Float32}
    terminal_states::Vector{Int}
end

function init_grid_mdp()
    rows, cols = 4, 4
    n_states = rows * cols
    rewards = zeros(Float32, n_states)

    # Matching reward structure from the notebook:
    # Python indices 9, 10, 14 (holes) -> Julia indices 10, 11, 15
    # Python index 15 (fish/goal) -> Julia index 16
    rewards[[10, 11, 15]] .= -2.0f0
    rewards[16] = 3.0f0

    terminal_states = [10, 11, 15, 16]
    return GridWorldMDP(rows, cols, n_states, 4, rewards, terminal_states)
end

# --- 2. Neural Network (Value Approximator) ---
# Modern Flux Chain to estimate Q-values for each action
function create_value_net(n_states, n_actions)
    return Chain(
        Dense(n_states, 64, relu),
        Dense(64, 32, relu),
        Dense(32, n_actions)
        )
end

# Helper to convert state index to one-hot vector for network input
one_hot_state(s, n) = Float32.(1:n .== s)

# --- 3. Modern Flux Training Logic ---
function train!(world::GridWorldMDP; n_episodes=300, gamma=0.9f0, lr=0.001f0)
    model = create_value_net(world.n_states, world.n_actions)
    opt_state = Flux.setup(Adam(lr), model)
    epsilon = 0.2 # Exploration rate

    for ep in 1:n_episodes
        state = 1 # Start at top-left

        while !(state in world.terminal_states)
            # Epsilon-greedy action selection
            action = if rand() < epsilon
                rand(1:world.n_actions)
            else
                argmax(model(one_hot_state(state, world.n_states)))
            end

            # Simplified transition logic (replaces the notebook's transition matrix)
            # In a full port, you would sample from 'transition_probabilities_given_action'
            next_state = state
            if action == 1 && (state % world.cols != 0) next_state += 1        # Right
                elseif action == 2 && (state % world.cols != 1) next_state -= 1   # Left
                elseif action == 3 && (state <= world.n_states - world.cols) next_state += world.cols # Down
                elseif action == 4 && (state > world.cols) next_state -= world.cols # Up
            end

            reward = world.rewards[next_state]

            # Compute Gradients using the Bellman Optimality Equation
            grads = Flux.gradient(model) do m
                q_values = m(one_hot_state(state, world.n_states))

                # TD Target logic
                target = reward + (next_state in world.terminal_states ? 0f0 :
                    gamma * maximum(m(one_hot_state(next_state, world.n_states))))

                    # Mean Squared Error between predicted and target Q-value
                    Flux.mse(q_values[action], target)
            end

            # Modern weight update
            Flux.update!(opt_state, model, grads[1])
            state = next_state
        end
    end
    return model
end

# --- 4. Execution ---
world = init_grid_mdp()
trained_model = train!(world)

# Output the learned policy
println("Learned Policy for 4x4 Grid (1=R, 2=L, 3=D, 4=U):")
    for s in 1:world.n_states
        q = trained_model(one_hot_state(s, world.n_states))
        print(argmax(q), " ")
        s % 4 == 0 && println()
    end

