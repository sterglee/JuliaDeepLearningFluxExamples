using Transformers
using Transformers.HuggingFace
using Flux

function setup_scratch_roberta()
    println("--- Initializing RoBERTa for Customer Support ---")
    
    model_name = "roberta-base"
    
    # 1. Load the actual Config Object from Hugging Face
    # This provides the base structure that getconfigname() requires
    config = load_config(model_name)
    
    # 2. Use keyword arguments in load_model to override the config.
    # Transformers.jl will merge these into the 'config' object automatically.
    # This matches Step 5 of the notebook: Initializing from scratch.
    model = load_model(model_name; 
        vocab_size = 50265,
        max_position_embeddings = 514,
        num_hidden_layers = 6,      # Reduced for CPU performance
        num_attention_heads = 12,
        intermediate_size = 1536
    )
    
    tokenizer = load_tokenizer(model_name)
    return tokenizer, model
end

# Verification
function test_run()
    tokenizer, model = setup_scratch_roberta()
    
    # Sample tweet input
    input_text = "Customer: Order status? Support: Processing now."
    tokens = lookup(tokenizer, tokenize(tokenizer, input_text))
    
    # Reshape to (Seq_Len, Batch_Size)
    input_batch = reshape(tokens, :, 1)
    
    # Forward pass
    output = model((token = input_batch,))
    println("Forward pass successful. Logit shape: ", size(output.logit))
end

test_run()


