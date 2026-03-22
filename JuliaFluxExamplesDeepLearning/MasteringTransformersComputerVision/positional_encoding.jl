using LinearAlgebra
using Statistics

# 1. Sinusoidal Positional Encoding
# Standard formula:
# PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
function get_positional_encoding(seq_len::Int, d_model::Int)
    pe = zeros(Float32, d_model, seq_len)
    for pos in 1:seq_len
        for i in 0:2:(d_model-1)
            div_term = exp(i * -log(10000.0) / d_model)
            # Julia uses 1-based indexing
            pe[i+1, pos] = sin((pos-1) * div_term)
            if i + 2 <= d_model
                pe[i+2, pos] = cos((pos-1) * div_term)
            end
        end
    end
    return pe
end

# 2. Cosine Similarity Utility
function cosine_sim(a, b)
    # dot(a, b) / (norm(a) * norm(b))
    return dot(a, b) / (norm(a) * norm(b))
end

# 3. Execution & Comparison logic
function main_positional_demo()
    d_model = 512
    seq_len = 10

    # Generate static positional encodings
    pe = get_positional_encoding(seq_len, d_model)

    # Simulate two words at different positions
    # Word "A" at position 2, Word "B" at position 3
    pos2_encoding = pe[:, 2]
    pos3_encoding = pe[:, 3]
    pos9_encoding = pe[:, 9]

    println("--- Positional Similarity Analysis ---")

    sim_2_3 = cosine_sim(pos2_encoding, pos3_encoding)
    sim_2_9 = cosine_sim(pos2_encoding, pos9_encoding)

    println("Similarity between adjacent positions (2 and 3): ", round(sim_2_3, digits=4))
    println("Similarity between distant positions (2 and 9): ", round(sim_2_9, digits=4))

    # 4. Applying Encoding to Embeddings
    # In the original Transformer, we add the PE to the Word Embedding
    word_embedding = randn(Float32, d_model)
    encoded_embedding = word_embedding .+ pos2_encoding

    println("\nFinal Encoded Vector Norm: ", norm(encoded_embedding))
end

main_positional_demo()
