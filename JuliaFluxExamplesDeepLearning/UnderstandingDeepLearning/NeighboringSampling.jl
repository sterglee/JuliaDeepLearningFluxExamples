using LinearAlgebra
using Random

# Define the Adjacency Matrix from Figure 13.4
A = [0 1 0 0 1 0 1 0;
     1 0 1 0 1 1 0 1;
     0 1 0 1 0 0 0 0;
     0 0 1 0 1 0 0 0;
     1 1 0 1 0 0 0 0;
     0 1 0 0 0 0 0 1;
     1 0 0 0 0 0 0 1;
     0 1 0 0 0 1 1 0]

"""
sample_neighbors(A, target_nodes, n_sample)

Implements neighborhood sampling as shown in Figure 13.10.
Returns a subset of nodes that are neighbors of the target_nodes.
"""
function sample_neighbors(A, target_nodes, n_sample)
    # 1. Find all indices where there is a connection to any target node
    # sum(A[target_nodes, :], dims=1) results in a 1xN row vector
    connections = sum(A[target_nodes, :], dims=1)
    neighbor_indices = findall(x -> x > 0, connections)

    # Extract the column index from the CartesianIndices
    neighbor_indices = [idx[2] for idx in neighbor_indices]

        # 2. Filter out nodes already in the current layer/set
        new_candidates = setdiff(neighbor_indices, target_nodes)

        # 3. Randomly sample n_sample nodes without replacement
        if length(new_candidates) > n_sample
            return shuffle(new_candidates)[1:n_sample]
        else
            return new_candidates
        end
    end

    # --- Workflow: Sampling for a 2-layer GNN ---
    Random.seed!(42)

    output_node = [1]
    println("Output Layer Node: ", output_node)

    # Layer 1 Sampling (Hidden Layer 1)
    layer1_nodes = sample_neighbors(A, output_node, 2)
    println("Hidden Layer 1 Nodes (sampled from output): ", layer1_nodes)

    # Layer 2 Sampling (Input Layer)
    # We sample neighbors for all nodes currently in the "computation graph"
    active_nodes = union(output_node, layer1_nodes)
    input_layer_nodes = sample_neighbors(A, active_nodes, 2)
    println("Input Layer Nodes (sampled from Layer 1): ", input_layer_nodes)

    println("\nComplete Computation Graph Nodes: ", union(active_nodes, input_layer_nodes))


