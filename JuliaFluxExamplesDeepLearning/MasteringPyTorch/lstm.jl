using Flux
using Flux: DataLoader, reset!, logitbinarycrossentropy, sigmoid, LSTM, Embedding, Dense
using Statistics
using Optimisers
using Printf
using Random

# 1. Data Generation (CPU)
function generate_synthetic_data(num_samples=2000, max_len=20, vocab_size=500, batch_size=32)
    Random.seed!(123)
    # X: (Sequence Length, Batch) - Integers for Embedding
    X = rand(1:vocab_size, max_len, num_samples)

    # y: (1, Batch) - Binary labels
    y = reshape(Float32.(rand(0:1, num_samples)), 1, :)

    split_idx = floor(Int, num_samples * 0.8)

    train_loader = DataLoader((X[:, 1:split_idx], y[:, 1:split_idx]),
                              batchsize=batch_size, shuffle=true)
    val_loader = DataLoader((X[:, split_idx+1:end], y[:, split_idx+1:end]),
                            batchsize=batch_size)

    return train_loader, val_loader, vocab_size
end

# 2. Model Architecture (Manual Bidirectional)
struct SentimentLSTM
    embedding
    lstm_f
    lstm_b
    fc
end

# Make the struct "Flux-aware" for gradient calculations
Flux.@functor SentimentLSTM

function SentimentLSTM(vocab_size, embed_dim, hidden_dim)
    return SentimentLSTM(
        Embedding(vocab_size => embed_dim),
        LSTM(embed_dim => hidden_dim),
        LSTM(embed_dim => hidden_dim),
        Dense(hidden_dim * 2 => 1) # *2 for concatenated forward/backward states
        )
end



function (m::SentimentLSTM)(x)
    # Reset RNN hidden states for a new batch
    reset!(m.lstm_f)
    reset!(m.lstm_b)

    # 1. Embedding: (EmbedDim, SeqLen, Batch)
    x_emb = m.embedding(x)

    # 2. Forward Pass: Process normal sequence
    h_f_all = m.lstm_f(x_emb)
    h_f_last = h_f_all[:, end, :] # Extract last timestep

    # 3. Backward Pass: Process reversed sequence
    # On CPU, reverse(..., dims=2) works perfectly without pointer errors
    x_rev = reverse(x_emb, dims=2)
    h_b_all = m.lstm_b(x_rev)
    h_b_last = h_b_all[:, end, :] # Extract last timestep of reversed sequence

    # 4. Concatenate: (HiddenDim * 2, Batch)
    h = vcat(h_f_last, h_b_last)

    # 5. Output logit
    return m.fc(h)
end

# 3. Training Loop
function train_cpu()
    # Hyperparameters
    VOCAB_SIZE = 500
    EMBED_DIM = 64
    HIDDEN_DIM = 32
    BATCH_SIZE = 32
    EPOCHS = 5

    train_loader, val_loader, v_size = generate_synthetic_data(2000, 20, VOCAB_SIZE, BATCH_SIZE)

    # Initialize model on CPU
    model = SentimentLSTM(v_size, EMBED_DIM, HIDDEN_DIM)

    # Optimizer setup
    opt_state = Flux.setup(Optimisers.Adam(0.001), model)

    println("Starting CPU Training...")

    for epoch in 1:EPOCHS
        total_loss = 0.0

        for (x, y) in train_loader
            # Standard backpropagation
            loss, grads = Flux.withgradient(model) do m
                logitbinarycrossentropy(m(x), y)
            end

            Flux.update!(opt_state, model, grads[1])
            total_loss += loss
        end

        # Validation (Accuracy)
        correct = 0
        total = 0
        for (x, y) in val_loader
            # model(x) returns logits, sigmoid converts to probabilities
            preds = sigmoid.(model(x)) .> 0.5f0
            correct += sum(preds .== y)
            total += size(y, 2)
        end

        @printf("Epoch %02d | Avg Loss: %.4f | Val Accuracy: %.2f%%\n",
                epoch, total_loss / length(train_loader), (correct / total) * 100)
    end
    println("Training Complete.")
end

# 4. Execution
train_cpu()

