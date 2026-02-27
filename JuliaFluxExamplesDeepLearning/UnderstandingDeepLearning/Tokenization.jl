using DataStructures

# ==========================================
# 1. Setup Input Text
# ==========================================
text = "How much wood could a woodchuck chuck if a woodchuck could chuck wood"

    # ==========================================
    # 2. Tokenization Logic (BPE)
    # ==========================================

    function tokenize(text_in, num_merges)
        # Initially, tokens are individual characters (and spaces)
        tokens = [string(c) for c in text_in]

            # Initialize vocabulary with unique characters
            vocab = unique(tokens)

            for i in 1:num_merges
                # 1. Count frequencies of all adjacent pairs
                pair_counts = DefaultDict{Tuple{String, String}, Int}(0)
                for j in 1:(length(tokens)-1)
                    pair = (tokens[j], tokens[j+1])
                    pair_counts[pair] += 1
                end

                # If no pairs exist, break
                isempty(pair_counts) && break

                # 2. Find the most frequent pair
                # In case of ties, the first one found is used
                best_pair = argmax(pair_counts)
                new_token = best_pair[1] * best_pair[2]

                # 3. Add new merged token to vocabulary
                push!(vocab, new_token)

                # 4. Replace occurrences of the pair with the new token
                new_tokens = String[]
                skip = false
                for j in 1:length(tokens)
                    if skip
                        skip = false
                        continue
                    end

                    if j < length(tokens) && tokens[j] == best_pair[1] && tokens[j+1] == best_pair[2]
                        push!(new_tokens, new_token)
                        skip = true
                    else
                        push!(new_tokens, tokens[j])
                    end
                end
                tokens = new_tokens
            end

            return tokens, vocab
        end

        # ==========================================
        # 3. Execution and Results
        # ==========================================
        # Perform 22 merges as requested in the notebook
        num_merges = 22
        tokens, vocab = tokenize(text, num_merges)

        println("Tokens: ", tokens)
        println("Number of tokens: ", length(tokens))
        println("Vocabulary size: ", length(vocab))
        println("Final Vocabulary: ", vocab)

