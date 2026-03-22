begin
    using Transformers
    using Transformers.HuggingFace
    using Flux
    using NearestNeighbors
    using StatsBase

    import Transformers.HuggingFace: load_model, load_tokenizer, encode

    # 1. k-NN Logic
    function run_knn()
        println("[1/3] Testing k-NN...")
        X = randn(Float32, 2, 50)
        y = rand(1:2, 50)
        tree = KDTree(X)
        idxs, _ = knn(tree, randn(Float32, 2), 3)
        println("      Success: Prediction is $(mode(y[idxs]))")
    end

    # 2. Transformer Logic (The part giving you errors)
    function run_transformer()
        println("[2/3] Testing Transformer Scratch Init...")
        model_name = "roberta-base"
        try
            # Variables are local to this function!
            model = load_model(model_name; vocab_size=1000, num_hidden_layers=1, num_attention_heads=2)
            tokenizer = load_tokenizer(model_name)
            
            sample_input = "Customer: Help! Support: OK."
            data = encode(tokenizer, sample_input)
            
            logits = model((token = reshape(data.token, :, 1),)).logit
            println("      Success: Logit Shape: $(size(logits))")
        catch err
            @warn "      Transformer test failed: $err"
        end
    end

    # 3. RL Logic
    function run_rl()
        println("[3/3] Testing RL Gini...")
        counts = [10, 5, 2]
        total = sum(counts)
        gini = 1.0 - sum((c/total)^2 for c in counts)
        println("      Success: Gini: $(round(gini, digits=4))")
    end

    # EXECUTION: This is where the functions actually run
    run_knn()
    run_rl()
    run_transformer()
end

