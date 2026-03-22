    using HTTP, JSON3, LinearAlgebra, StatsBase

    # 1. Configuration & Mock Vector Database
    # In RAG, we compare the query embedding to document embeddings
    const API_KEY = get(ENV, "OPENAI_API_KEY", "YOUR_KEY")

    documents = [
        "The Big Bang theory explains the origin of the universe.",
        "Photosynthesis is how plants convert sunlight into energy.",
        "The Julia programming language was designed for high performance."
            ]

    # Mock Embeddings (In production, use OpenAI Embeddings API)
    doc_embeddings = [randn(1536) for _ in 1:length(documents)]

        # 2. Retrieval Logic: Cosine Similarity
        function retrieve_context(query_vec, doc_vecs, docs)
            # dot(a, b) / (norm(a) * norm(b))
            similarities = [dot(query_vec, dv) / (norm(query_vec) * norm(dv)) for dv in doc_vecs]
                best_idx = argmax(similarities)
                return docs[best_idx]
            end

            # 3. Augmented Generation
            function run_rag_query(query_text)
                println("--- Starting RAG Process ---")

                # Step A: Retrieval
                query_vec = randn(1536) # Mock query embedding
                context = retrieve_context(query_vec, doc_embeddings, documents)
                println("Retrieved Context: ", context)

                # Step B: Generation
                endpoint = "https://api.openai.com/v1/chat/completions"
                body = Dict(
                    "model" => "gpt-4o",
                    "messages" => [
                        Dict("role" => "system", "content" => "Use this context to answer: $context"),
                        Dict("role" => "user", "content" => query_text)
                        ]
                    )

                    try
                        response = HTTP.post(endpoint,
                                             ["Authorization" => "Bearer $API_KEY",
                                              "Content-Type" => "application/json"],
                                             JSON3.write(body))

                                             result = JSON3.read(response.body)
                                             println("\nGPT-4o Response:\n", result.choices[1].message.content)
                                             catch err
                        @error "API Call failed. Ensure your OPENAI_API_KEY is set." exception=err
                    end
            end

            # 4. EXECUTION
            run_rag_query("Explain how plants eat.")

