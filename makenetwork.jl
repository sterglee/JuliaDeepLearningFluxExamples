using LinearAlgebra, Random, Plots, Statistics

# --- 1. The Matrix Generator ---
function makenetwork(nodes, connections, ruleweights; trials=500)
    # Get all possible unique edges in the lower triangle
    possible_edges = [(i, j) for i in 1:nodes for j in 1:i-1]
        n_possible = length(possible_edges)
        n_conn = min(connections, n_possible)

        N = zeros(Int, nodes, nodes)
        A = zeros(Float64, nodes, nodes)
        connected = false

        for tr in 1:trials
            # Randomly pick edges
            selected = shuffle(possible_edges)[1:n_conn]
            N_temp = zeros(Int, nodes, nodes)
            for (i, j) in selected
                N_temp[i, j] = 1
            end

            # Symmetric + diagonal
            N_temp = N_temp + N_temp' + I

            # Check connectivity: (I + Adjacency)^nodes should have all positive entries
            if all((Matrix(N_temp)^nodes) .> 0)
                N = N_temp
                connected = true
                break
            end
        end

        if !connected
            return N, A, false
        end

        # Calculate Weights
        node_degrees = sum(N, dims=1) # Vector of degrees for each node

        if ruleweights == "metropolis"
            for j in 1:nodes, i in 1:nodes
                if N[i, j] == 1 && i != j
                    A[i, j] = 1.0 / max(node_degrees[i], node_degrees[j])
                end
            end
            for j in 1:nodes
                A[j, j] = 1.0 - sum(A[:, j])
            end
            elseif ruleweights == "uniform"
            for j in 1:nodes
                A[:, j] .= N[:, j] ./ node_degrees[j]
            end
        else # noncooperation
            A = Matrix(1.0I, nodes, nodes)
        end

        return N, A, connected
    end

    # --- 2. THE EXAMPLE CALL ---

    # Request 10 nodes with 15 random connections using Metropolis weights
    nodes, conns = 10, 15
    adj_matrix, weight_matrix, success = makenetwork(nodes, conns, "metropolis")

    if success
        println("Successfully generated a connected network!")

        # Visualization
        p1 = heatmap(adj_matrix, title="Adjacency (Connections)", c=:blues, yflip=true)
        p2 = heatmap(weight_matrix, title="Metropolis Weights", c=:viridis, yflip=true)

        display(plot(p1, p2, layout=(1,2), size=(800, 350)))
    else
        println("Could not find a connected graph. Try increasing connections.")
    end


