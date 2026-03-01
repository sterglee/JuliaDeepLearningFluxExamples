using LinearAlgebra, Random, Graphs, Plots

function makenetwork(nodes, connections; ruleweights="Metropolis", trials=1, visualize=false)
    lowtri_pos = findall(tril(ones(Bool, nodes, nodes), -1))

    if connections > length(lowtri_pos)
        connections = length(lowtri_pos)
    end

    N = BitMatrix(zeros(nodes, nodes))
    connected = false
    tr = 1

    while !connected && tr <= trials
        temp_N = zeros(Bool, nodes, nodes)

        # Pick random edges
        edge_indices = randperm(length(lowtri_pos))[1:min(connections, length(lowtri_pos))]
        for idx in edge_indices
            pos = lowtri_pos[idx]
            temp_N[pos] = true
            temp_N[pos[2], pos[1]] = true
        end

        # Add self-loops (Identity)
        for i in 1:nodes
            temp_N[i, i] = true
        end

        N = BitMatrix(temp_N)
        g = SimpleGraph(N)
        connected = is_connected(g)
        tr += 1
    end

    # Degrees (including self-loop)
    degrees = [count(N[i, :]) for i in 1:nodes]
        A = zeros(Float64, nodes, nodes)

        rule = lowercase(ruleweights)
        if rule == "metropolis"
            for j in 1:nodes
                for i in 1:nodes
                    if i != j && N[i, j]
                        # Metropolis rule: a_ij = 1 / max(deg(i), deg(j))
                        A[i, j] = 1.0 / max(degrees[i], degrees[j])
                    end
                end
                # Sum of off-diagonals for column j
                off_diag_sum = sum(A[1:end .!= j, j])
                A[j, j] = 1.0 - off_diag_sum
            end
            elseif rule == "uniform"
            for j in 1:nodes
                A[:, j] = Vector(N[:, j]) ./ degrees[j]
            end
            elseif rule == "noncooperation"
            A = Matrix{Float64}(I, nodes, nodes)
        end

        if visualize
            # Use a simple circular layout if you don't have GraphRecipes
            θ = range(0, 2π, length=nodes+1)[1:end-1]
            xy = hcat(cos.(θ), sin.(θ))

            plt = plot(legend=false, aspect_ratio=:equal, title="Topology: $ruleweights")
            for i in 1:nodes, j in i+1:nodes
                if N[i, j]
                    plot!([xy[i, 1], xy[j, 1]], [xy[i, 2], xy[j, 2]], color=:gray, alpha=0.5)
                end
            end
            scatter!(xy[:, 1], xy[:, 2], color=:red, markersize=10)
            for i in 1:nodes
                annotate!(xy[i, 1]*1.2, xy[i, 2]*1.2, text("$i", 10))
            end
            display(plt)
        end

        return N, A, connected
    end

    # Verification
    N, A, is_conn = makenetwork(10, 15, ruleweights="Metropolis", trials=1000, visualize=true)

