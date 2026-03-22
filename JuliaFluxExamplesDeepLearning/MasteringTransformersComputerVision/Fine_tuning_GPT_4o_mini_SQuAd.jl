begin
    using HTTP, JSON3, Random, LinearAlgebra

    # --- 1. GENERATE SYNTHETIC DATA ---
    function generate_synthetic_squad(filename="synthetic_train.jsonl")
        println("Generating synthetic training data...")
        # Synthetic examples following the Chat format
        data = [
            Dict("messages" => [
                Dict("role" => "system", "content" => "You are a SQuAD assistant."),
                Dict("role" => "user", "content" => "Context: Julia is a high-level, high-performance, dynamic programming language. Question: What kind of language is Julia?"),
                Dict("role" => "assistant", "content" => "Julia is a high-level, high-performance, dynamic programming language.")
                ]),
            Dict("messages" => [
                Dict("role" => "system", "content" => "You are a SQuAD assistant."),
                Dict("role" => "user", "content" => "Context: GPT-4o-mini is a cost-efficient model. Question: Is GPT-4o-mini expensive?"),
                Dict("role" => "assistant", "content" => "No, GPT-4o-mini is a cost-efficient model.")
                ])
            ]

            open(filename, "w") do f
                for entry in data
                    println(f, JSON3.write(entry))
                end
            end
            println("Created $filename with $(length(data)) examples.")
    end

    # --- 2. INTEGRATED RAG RETRIEVAL (Synthetic) ---
    function synthetic_retrieval_test()
        println("\nTesting Synthetic RAG Retrieval...")

        # Mock Vector DB
        docs = ["Julia is fast.", "Python is popular.", "OpenAI made GPT."]
        # Create synthetic embeddings (3-dimensional for simplicity)
        doc_embeddings = [[1.0, 0.1, 0.1], [0.1, 1.0, 0.1], [0.1, 0.1, 1.0]]
        query_vec = [0.9, 0.2, 0.1] # Closest to "Julia is fast"

        # Cosine Similarity Retrieval
        scores = [dot(query_vec, d) / (norm(query_vec) * norm(d)) for d in doc_embeddings]
            best_match = docs[argmax(scores)]

            println("Query: 'Tell me about Julia performance'")
            println("Retrieved Context: $best_match")
        end

        # --- 3. EXECUTION ---
        generate_synthetic_squad()
        synthetic_retrieval_test()

        println("\n[FINISH] Synthetic environment ready for 'Fine_tuning_GPT_4_1_mini_SQuAd.jl'")
            end

        begin
    using HTTP, JSON3, Random, LinearAlgebra

    # --- 1. GENERATE SYNTHETIC DATA ---
    function generate_synthetic_squad(filename="synthetic_train.jsonl")
        println("Generating synthetic training data...")
        # Synthetic examples following the Chat format
        data = [
            Dict("messages" => [
                Dict("role" => "system", "content" => "You are a SQuAD assistant."),
                Dict("role" => "user", "content" => "Context: Julia is a high-level, high-performance, dynamic programming language. Question: What kind of language is Julia?"),
                Dict("role" => "assistant", "content" => "Julia is a high-level, high-performance, dynamic programming language.")
                ]),
            Dict("messages" => [
                Dict("role" => "system", "content" => "You are a SQuAD assistant."),
                Dict("role" => "user", "content" => "Context: GPT-4o-mini is a cost-efficient model. Question: Is GPT-4o-mini expensive?"),
                Dict("role" => "assistant", "content" => "No, GPT-4o-mini is a cost-efficient model.")
                ])
            ]

            open(filename, "w") do f
                for entry in data
                    println(f, JSON3.write(entry))
                end
            end
            println("Created $filename with $(length(data)) examples.")
    end

    # --- 2. INTEGRATED RAG RETRIEVAL (Synthetic) ---
    function synthetic_retrieval_test()
        println("\nTesting Synthetic RAG Retrieval...")

        # Mock Vector DB
        docs = ["Julia is fast.", "Python is popular.", "OpenAI made GPT."]
        # Create synthetic embeddings (3-dimensional for simplicity)
        doc_embeddings = [[1.0, 0.1, 0.1], [0.1, 1.0, 0.1], [0.1, 0.1, 1.0]]
        query_vec = [0.9, 0.2, 0.1] # Closest to "Julia is fast"

        # Cosine Similarity Retrieval
        scores = [dot(query_vec, d) / (norm(query_vec) * norm(d)) for d in doc_embeddings]
            best_match = docs[argmax(scores)]

            println("Query: 'Tell me about Julia performance'")
            println("Retrieved Context: $best_match")
        end

        # --- 3. EXECUTION ---
        generate_synthetic_squad()
        synthetic_retrieval_test()

        println("\n[FINISH] Synthetic environment ready for 'Fine_tuning_GPT_4_1_mini_SQuAd.jl'")
        end

