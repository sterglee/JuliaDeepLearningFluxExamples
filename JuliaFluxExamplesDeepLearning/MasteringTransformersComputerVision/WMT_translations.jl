using Statistics

# 1. Fixed N-gram Logic
# We use AbstractString to accept both String and SubString
function get_ngrams(tokens::Vector{<:AbstractString}, n::Int)
    ngrams = String[]
    for i in 1:(length(tokens) - n + 1)
        # Join the slice into a single string n-gram
        push!(ngrams, join(tokens[i:i+n-1], " "))
    end
    return ngrams
end

# 2. BLEU Score Calculation
function calculate_bleu(reference::String, candidate::String; max_n=4)
    # split() returns SubStrings
    ref_tokens = split(lowercase(reference))
    cand_tokens = split(lowercase(candidate))
    
    precisions = Float64[]
    
    for n in 1:max_n
        ref_ngrams = get_ngrams(ref_tokens, n)
        cand_ngrams = get_ngrams(cand_tokens, n)
        
        if isempty(cand_ngrams)
            push!(precisions, 0.0)
            continue
        end
        
        # Count matches: how many candidate n-grams appear in the reference
        matches = filter(x -> x in ref_ngrams, cand_ngrams)
        # Modified precision to handle zero matches gracefully
        p_n = length(matches) / length(cand_ngrams)
        push!(precisions, p_n)
    end
    
    # Brevity Penalty (BP)
    c = length(cand_tokens)
    r = length(ref_tokens)
    bp = c > r ? 1.0 : exp(1 - r/c)
    
    # Calculate geometric mean of precisions
    # We use a small epsilon (1e-9) to prevent log(0) errors
    smoothed_precisions = [p == 0 ? 1e-9 : p for p in precisions]
    geo_mean = exp(sum(log.(smoothed_precisions)) / max_n)
    
    return bp * geo_mean
end

# 3. Validation
function main_bleu_fix()
    println("--- Fixed BLEU Evaluation ---")
    
    # From WMT_translations.ipynb example
    reference = "je vous invite a vous lever pour cette minute de silence"
    candidate = "levez vous svp pour cette minute de silence"
    
    score = calculate_bleu(reference, candidate)
    
    println("Reference: ", reference)
    println("Candidate: ", candidate)
    println("BLEU Score: ", round(score, digits=4))
end

main_bleu_fix()

