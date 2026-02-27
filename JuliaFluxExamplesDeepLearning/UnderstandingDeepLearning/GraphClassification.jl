using LinearAlgebra, Plots

# Define the adjacency matrix (8 nodes)
A = [0 1 0 0 1 0 1 0;
     1 0 1 0 1 1 0 1;
     0 1 0 1 0 0 0 0;
     0 0 1 0 1 0 0 0;
     1 1 0 1 0 0 0 0;
     0 1 0 0 0 0 0 1;
     1 0 0 0 0 0 0 1;
     0 1 0 0 0 1 1 0]

# Calculate number of walks of length 3
A3 = A^3
println("Number of walks of length 3 between nodes 4 and 8: ", A3[4, 8])

# TODO: Algorithm to find minimum path distance between nodes 1 and 7
function min_path(Adj, start_node, end_node)
    N = size(Adj, 1)
    for length in 1:N
        if (Adj^length)[start_node, end_node] > 0
            return length
        end
    end
    return -1
end

println("Minimum distance between node 1 and 7: ", min_path(A, 1, 7))

