using Random
using Distributions  # <-- required for Categorical

# Define the states and transition probabilities
states = ["Sunny", "Rainy"]
transition_matrix = [0.8 0.2;  # P(next | current) for Sunny
                     0.4 0.6]  # P(next | current) for Rainy

# Map states to indices
state_index = Dict("Sunny"=>1, "Rainy"=>2)

# Simulate a Markov chain
function simulate_markov_chain(n_steps::Int, start_state::String)
    chain = Vector{String}(undef, n_steps)
    chain[1] = start_state
    for t in 2:n_steps
        prev_idx = state_index[chain[t-1]]
        chain[t] = states[rand(Categorical(transition_matrix[prev_idx, :]))]
    end
    return chain
end

# Example simulation
Random.seed!(42)
sequence = simulate_markov_chain(10, "Sunny")
println("Simulated weather sequence: ", sequence)

