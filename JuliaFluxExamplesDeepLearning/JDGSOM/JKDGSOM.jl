using Statistics
using LinearAlgebra
using Printf

# ==========================================
# 1. Structures (NNPattern, NetSOMNode, JKDGSOM)
# ==========================================
mutable struct NNPattern
    p_data::Matrix{Float64}
    class_labels::Matrix{Int}
    num_patterns::Int
    num_labels::Int
    size_vector::Int
    feature_names::Vector{String}
    pattern_names::Vector{String}
    winner_row::Vector{Int}
    winner_col::Vector{Int}

    function NNPattern(npats, svec, nlabs)
        new(
            zeros(npats, svec),
            zeros(Int, npats, nlabs),
            npats, nlabs, svec,
            fill("", svec),
            fill("", npats),
            zeros(Int, npats),
            zeros(Int, npats)
            )
    end
end

mutable struct NetSOMNode
    row::Int
    col::Int
    weights::Vector{Float64}
    NetSOMNode(r, c, svec) = new(r, c, rand(Float64, svec))
end

mutable struct JKDGSOM
    rows::Int
    cols::Int
    pattern::NNPattern
    nodes::Matrix{NetSOMNode}

    function JKDGSOM(r, c, np::NNPattern)
        grid = [NetSOMNode(i, j, np.size_vector) for i in 1:r, j in 1:c]
            new(r, c, np, grid)
        end
    end

    # ==========================================
    # 2. Performance Metrics Structure
    # ==========================================
    struct SOMMetrics
        quantization_error::Float64
        topographic_error::Float64
    end

    # ==========================================
    # 3. Core Logic & Evaluation
    # ==========================================

    function calculate_dist(v1, v2)
        d = 0.0
        for i in eachindex(v1)
            d += (v1[i] - v2[i])^2
        end
        return sqrt(d)
    end

    function evaluate_som(som::JKDGSOM)
        total_qe = 0.0
        topographic_violations = 0
        np = som.pattern.num_patterns

        for i in 1:np
            input_vec = @view som.pattern.p_data[i, :]

            # Track the two best matching units (BMUs)
            d1, d2 = Inf, Inf
            bmu1_pos = (1, 1)
            bmu2_pos = (1, 1)

            for r in 1:som.rows, c in 1:som.cols
                d = calculate_dist(input_vec, som.nodes[r, c].weights)

                if d < d1
                    d2 = d1
                    bmu2_pos = bmu1_pos
                    d1 = d
                    bmu1_pos = (r, c)
                    elseif d < d2
                    d2 = d
                    bmu2_pos = (r, c)
                end
            end

            # 1. Quantization Error: Dist from input to its BMU
            total_qe += d1

            # 2. Topographic Error: Check if 1st and 2nd BMUs are neighbors
            row_dist = abs(bmu1_pos[1] - bmu2_pos[1])
            col_dist = abs(bmu1_pos[2] - bmu2_pos[2])

            # If they are not adjacent (including diagonals), it's a violation
            if row_dist > 1 || col_dist > 1
                topographic_violations += 1
            end
        end

        return SOMMetrics(total_qe / np, topographic_violations / np)
    end

    function find_winner!(som::JKDGSOM, pattern_idx::Int)
        min_dist = Inf
        best_pos = (1, 1)
        input_vec = @view som.pattern.p_data[pattern_idx, :]

        for r in 1:som.rows, c in 1:som.cols
            d = calculate_dist(input_vec, som.nodes[r, c].weights)
            if d < min_dist
                min_dist = d
                best_pos = (r, c)
            end
        end

        som.pattern.winner_row[pattern_idx] = best_pos[1]
        som.pattern.winner_col[pattern_idx] = best_pos[2]
        return best_pos
    end

    # ==========================================
    # 4. Data Loading
    # ==========================================
    function load_brca_data(filepath::String)
        lines = filter(l -> !isempty(strip(l)), readlines(filepath))
        header = split(lines[1])
        num_pats, size_vec, num_labs = parse.(Int, header[1:3])

        pattern = NNPattern(num_pats, size_vec, num_labs)

        f_names = split(lines[2])
        for i in 1:min(length(f_names), size_vec)
            pattern.feature_names[i] = String(f_names[i])
        end

        for i in 1:num_pats
            row_parts = split(lines[i+3])
            pattern.pattern_names[i] = String(row_parts[1])
            for j in 1:size_vec
                pattern.p_data[i, j] = parse(Float64, row_parts[j+1])
            end
            for k in 1:num_labs
                pattern.class_labels[i, k] = parse(Int, row_parts[size_vec + 1 + k])
            end
        end
        return pattern
    end

    # ==========================================
    # 5. Full Pipeline
    # ==========================================
    function process_pipeline(file_path::String)
        println("--- Starting SOM Analysis ---")

        # 1. Load Data
        pattern_data = load_brca_data(file_path)
        println("Step 1: Data Loaded ($(pattern_data.num_patterns) patterns).")

        # 2. Init SOM
        som = JKDGSOM(10, 10, pattern_data)

        # 3. Initial Baseline
        initial_metrics = evaluate_som(som)
        @printf("Initial QE: %.4f | Initial TE: %.4f\n", initial_metrics.quantization_error, initial_metrics.topographic_error)

        # 4. Training
        epochs = 20
        println("Step 2: Training for $epochs epochs...")
            for epoch in 1:epochs
                # Decaying learning rate
                alpha = 0.1 * (1.0 - epoch/epochs)

                for p_idx in 1:pattern_data.num_patterns
                    winner = find_winner!(som, p_idx)
                    bmu = som.nodes[winner[1], winner[2]]

                    # Weight Update
                    for i in 1:pattern_data.size_vector
                        bmu.weights[i] += alpha * (pattern_data.p_data[p_idx, i] - bmu.weights[i])
                    end
                end
            end

            # 5. Final Evaluation
            final_metrics = evaluate_som(som)
            println("\n--- Final Performance Evaluation ---")
            @printf("Final Quantization Error: %.4f\n", final_metrics.quantization_error)
            @printf("Final Topographic Error:  %.4f\n", final_metrics.topographic_error)
            @printf("Error Improvement (QE):   %.2f%%\n",
                    (1 - final_metrics.quantization_error/initial_metrics.quantization_error) * 100)

            println("\nProcessing Complete!")
        end

        # To execute:
        process_pipeline("BRCA.dat")




