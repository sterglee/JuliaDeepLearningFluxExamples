using Flux
using Flux: DataLoader, logitcrossentropy, onecold, trainmode!, testmode!
using Flux: Dense, Chain, LayerNorm, MultiHeadAttention
using Functors
using HTTP, Statistics, MLUtils, Random

# ------------------------------------------------------------
# 1. DATA
# ------------------------------------------------------------

function get_protein_data()
    url = "https://raw.githubusercontent.com/cmbi/protein-sequence-analysis/master/data/cyc.fasta"

    try
        return String(HTTP.get(url).body)
    catch
        println("Download failed, generating synthetic proteins")

        aa = collect("ARNDCEQGHILKMFPSTWYV")
        return join([">seq$i\n" * join(rand(aa,100)) for i in 1:1000], "\n")
    end
end


function parse_kmers(raw_data, window)

    sequences = String[]
    current = ""

    for line in split(raw_data,"\n")
        line = strip(line)

        if startswith(line,">")
            !isempty(current) && push!(sequences,current)
            current = ""
        elseif !isempty(line)
            current *= line
        end
    end

    !isempty(current) && push!(sequences,current)

    chars = sort(unique(join(sequences)))
    c2i = Dict(c=>i for (i,c) in enumerate(chars))

    X = Vector{Vector{Int}}()
    Y = Int[]

    for seq in sequences

        L = length(seq)

        if L <= window
            continue
        end

        for i in 1:(L-window)

            push!(X,[c2i[c] for c in seq[i:i+window-1]])
            push!(Y,c2i[seq[i+window]])

        end
    end

    return X,Y,length(chars)
end

# ------------------------------------------------------------
# 2. POSITIONAL ENCODING
# ------------------------------------------------------------

struct PositionalEncoding
    weight
end

Functors.@functor PositionalEncoding

(m::PositionalEncoding)(x) = x .+ m.weight[:,1:size(x,2),:]


function positional_weights(dim,max_len)

    pe = zeros(Float32,dim,max_len,1)

    for pos in 1:max_len
        for i in 1:2:(dim-1)

            pe[i,pos,1]   = sin(pos / 10000^((i-1)/dim))
            pe[i+1,pos,1] = cos(pos / 10000^((i-1)/dim))

        end
    end

    pe
end

# ------------------------------------------------------------
# 3. TRANSFORMER BLOCK
# ------------------------------------------------------------

struct TransformerBlock
    norm1
    attn
    norm2
    mlp
end

Functors.@functor TransformerBlock


function TransformerBlock(dim,heads,mlp_dim)

    TransformerBlock(

        LayerNorm(dim),
        MultiHeadAttention(dim; nheads=heads),
        LayerNorm(dim),

        Chain(
            Dense(dim,mlp_dim,relu),
            Dense(mlp_dim,dim)
        )
    )
end


function (m::TransformerBlock)(x)

    h = m.norm1(x)

    a,_ = m.attn(h,h,h)

    x = x .+ a

    x = x .+ m.mlp(m.norm2(x))

    return x
end


# ------------------------------------------------------------
# 4. MAIN
# ------------------------------------------------------------
# ------------------------------------------------------------
# 4. MAIN (Corrected)
# ------------------------------------------------------------

function main()
    window = 20
    embed  = 64
    batch  = 64

    raw = get_protein_data()
    Xidx, Yidx, vocab = parse_kmers(raw, window)
    labels = 1:vocab

    println("Dataset windows: ", length(Xidx))
    println("Vocabulary size: ", vocab)

    # --------------------
    # Build dataset 
    # --------------------
    # We use mapobs or manual transformation to ensure shapes are (Vocab, Window, Batch)
    Xdata = [Float32.(Flux.onehotbatch(x, labels)) for x in Xidx]
    Ydata = [Float32.(Flux.onehot(y, labels)) for y in Yidx]

    # Use MLUtils to stack them into proper tensors before loading
    # This prevents the "tuple of matrices" issue
    X_tensor = MLUtils.stack(Xdata) # Result: (Vocab, Window, Samples)
    Y_tensor = MLUtils.stack(Ydata) # Result: (Vocab, Samples)

    train_data, test_data = splitobs((X_tensor, Y_tensor), at=0.9)

    train_loader = DataLoader(train_data, batchsize=batch, shuffle=true)
    test_loader  = DataLoader(test_data, batchsize=batch)

    # --------------------
    # Model
    # --------------------
    model = Chain(
        Dense(vocab, embed),
        PositionalEncoding(positional_weights(embed, window)),
        TransformerBlock(embed, 4, 128),
        TransformerBlock(embed, 4, 128),
        x -> x[:, end, :], # Extract summary of sequence
        Dense(embed, vocab)
    )

    opt = Flux.setup(Flux.Adam(1f-3), model)

    # --------------------
    # Training Loop
    # --------------------
    for epoch in 1:100
        trainmode!(model)
        total_loss = 0.0

        for (x, y) in train_loader
            # x is (Vocab, Window, Batch), y is (Vocab, Batch)
            grads = Flux.gradient(model) do m
                y_hat = m(x)
                logitcrossentropy(y_hat, y)
            end

            Flux.update!(opt, model, grads[1])
            total_loss += logitcrossentropy(model(x), y)
        end

        # --------------------
        # Evaluation
        # --------------------
        testmode!(model)
        
        # Calculate accuracy across all test batches
        acc_sum = 0.0
        batch_count = 0
        for (x, y) in test_loader
            y_hat = model(x)
            acc_sum += mean(onecold(y_hat) .== onecold(y))
            batch_count += 1
        end

        println("Epoch $epoch | Loss = $(round(total_loss/length(train_loader), digits=4)) | Test Acc = $(round((acc_sum/batch_count)*100, digits=2))%")
    end
end


main()


