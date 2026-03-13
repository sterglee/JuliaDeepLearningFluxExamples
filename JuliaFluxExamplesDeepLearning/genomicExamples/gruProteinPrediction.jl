using Flux
using Flux: DataLoader, logitcrossentropy, onecold, reset!, trainmode!, testmode!
using HTTP
using Statistics
using JLD2
using MLUtils

# --- 1. Data Acquisition ---

    
function get_protein_data()
    # Reliable URL for Cytochrome C sequences
    url = "https://raw.githubusercontent.com/cmbi/protein-sequence-analysis/master/data/cyc.fasta"
    println("Attempting to download sequences from: $url")
    
    try
        response = HTTP.get(url, update_period=0.5)
        return String(response.body)
    catch e
        @warn "Download failed ($e). Generating synthetic protein data for demonstration..."
        # Fallback: Generate 50 random protein-like sequences
        amino_acids = "ARNDCEQGHILKMFPSTWYV"
        return join([">seq$i\n" * join([rand(amino_acids) for _ in 1:100]) for i in 1:50], "\n")
    end
end

function parse_fasta(raw_data)
    sequences = String[]
    current_seq = ""
    for line in split(raw_data, "\n")
        line = strip(line)
        if startswith(line, ">")
            if !isempty(current_seq) push!(sequences, current_seq) end
            current_seq = ""
        elseif !isempty(line)
            current_seq *= line
        end
    end
    if !isempty(current_seq) push!(sequences, current_seq) end
    return sequences
end

# --- 2. Encoding & Windowing ---

function build_vocab(sequences)
    all_chars = sort(unique(join(sequences)))
    char_to_int = Dict(c => i for (i, c) in enumerate(all_chars))
    int_to_char = Dict(i => c for (i, c) in enumerate(all_chars))
    return char_to_int, int_to_char, length(all_chars)
end

function create_dataset(sequences, char_to_int, window_size=10)
    X_data = Vector{Int}[]
    Y_data = Int[]
    
    for seq in sequences
        if length(seq) <= window_size continue end
        for i in 1:(length(seq) - window_size)
            push!(X_data, [char_to_int[c] for c in seq[i:(i+window_size-1)]])
            push!(Y_data, char_to_int[seq[i+window_size]])
        end
    end
    
    vocab_size = length(char_to_int)
    # One-hot encoding
    X_batch = Flux.onehotbatch(MLUtils.stack(X_data), 1:vocab_size) # (Vocab, Window, Samples)
    Y_batch = Flux.onehotbatch(Y_data, 1:vocab_size)               # (Vocab, Samples)
    
    return X_batch, Y_batch
end

# --- 3. The GRU Architecture ---

function create_model(vocab_size)
    return Chain(
        GRU(vocab_size => 128),      # Process the sequence step-by-step
        x -> x[:, end, :],           # Get the state at the last amino acid
        Dense(128, 64, relu),
        Dense(64, vocab_size)        # Predict probability for next amino acid
    )
end

# --- 4. Evaluation Metrics ---

function evaluate_metrics(model, loader, vocab_size)
    testmode!(model)
    preds, actuals = Int[], Int[]
    
    for (x, y) in loader
        reset!(model)
        y_hat = model(x)
        append!(preds, onecold(y_hat, 1:vocab_size))
        append!(actuals, onecold(y, 1:vocab_size))
    end
    
    accuracy = mean(preds .== actuals)
    return accuracy
end

# --- 5. Main Loop ---

function main()
    # 1. Prepare Data
    fasta_txt = get_protein_data()
    sequences = parse_fasta(fasta_txt)
    char_to_int, int_to_char, vocab_size = build_vocab(sequences)
    
    window_size = 12
    X, Y = create_dataset(sequences, char_to_int, window_size)
    
    # Split 80/20
    (x_train, y_train), (x_test, y_test) = splitobs((X, Y), at=0.8)
    train_loader = DataLoader((x_train, y_train), batchsize=32, shuffle=true)
    test_loader = DataLoader((x_test, y_test), batchsize=32)

    # 2. Setup Model
    model = create_model(vocab_size)
    opt_state = Flux.setup(Adam(0.001), model)
    
    println("Vocabulary Size: $vocab_size | Samples: $(size(X, 3))")
    println("Training starting...")

    for epoch in 1:100
        trainmode!(model)
        total_loss = 0.0
        for (x, y) in train_loader
            reset!(model)
            loss, grads = Flux.withgradient(model) do m
                y_hat = m(x)
                logitcrossentropy(y_hat, y)
            end
            Flux.update!(opt_state, model, grads[1])
            total_loss += loss
        end
        
        if epoch % 5 == 0 || epoch == 1
            acc = evaluate_metrics(model, test_loader, vocab_size)
            println("Epoch $epoch | Loss: $(round(total_loss/length(train_loader), digits=4)) | Test Acc: $(round(acc*100, digits=2))%")
        end
    end

    # 3. Example Prediction
    sample_seq = sequences[1][1:window_size]
    println("\nInput sequence:  $sample_seq")
    
    input_encoded = Flux.onehotbatch([char_to_int[c] for c in sample_seq], 1:vocab_size)
    input_tensor = reshape(input_encoded, vocab_size, window_size, 1)
    
    reset!(model)
    pred_idx = onecold(model(input_tensor), 1:vocab_size)[1]
    println("Predicted next amino acid: $(int_to_char[pred_idx])")
end

main()

