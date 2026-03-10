# Flux time: 16.465041 seconds

using Flux,
    MLDatasets,
    Random,
    Statistics,
    Printf,
    OneHotArrays,
    Optimisers,
    LinearAlgebra,
    SparseArrays

# --- 1. Data Loading & Sparse Laplacian ---

function load_cora_sparse()
    data = Cora()
    gph = data.graphs[1]

    features = gph.node_data.features # (1433, 2708)
    num_nodes = size(features, 2)
    num_classes = length(data.metadata["classes"])
    targets = onehotbatch(gph.node_data.targets, 1:num_classes)

    # FIX: Handle edge_index as a Tuple (source_nodes, target_nodes)
    u, v = gph.edge_index

    # Build Sparse Adjacency Matrix: A_hat = A + I
    # Undirected: include (u,v) and (v,u)
    rows = vcat(u, v, 1:num_nodes)
    cols = vcat(v, u, 1:num_nodes)
    vals = ones(Float32, length(rows))

    # Create sparse matrix; entries > 0 becomes 1.0 to handle multi-edges
    A_hat = (sparse(rows, cols, vals, num_nodes, num_nodes) .> 0) .* 1.0f0

    # Symmetric Normalization: L_norm = D^-0.5 * A_hat * D^-0.5
    deg = vec(sum(A_hat, dims=2))
    d_inv_sqrt = @. 1.0f0 / sqrt(deg)
    d_inv_sqrt[isinf.(d_inv_sqrt)] .= 0

    # Scaling sparse matrix rows and columns by diagonal D^-0.5
    D_inv_sqrt = sparse(1:num_nodes, 1:num_nodes, d_inv_sqrt)
    L_norm = D_inv_sqrt * A_hat * D_inv_sqrt

    # Split Indices
    train_idx = 1:140
    val_idx = 141:640
    test_idx = 1709:2708

    return features, targets, L_norm, (train_idx, val_idx, test_idx)
end

# --- 2. Model Definition ---

struct GCNLayer
    W
    b
end

Flux.@functor GCNLayer

function GCNLayer(in_dim::Int, out_dim::Int)
    return GCNLayer(
        Flux.glorot_uniform(out_dim, in_dim),
        Flux.zeros32(out_dim)
        )
end

# Propagation: H = (W * X + b) * L_norm
# L_norm is (N, N), X is (F, N). Result is (F_out, N)
function (m::GCNLayer)(x::AbstractMatrix, L_norm::AbstractSparseMatrix)
    return (m.W * x .+ m.b) * L_norm
end



mutable struct GCN
    layers
    dropout_rate
end

Flux.@functor GCN

function build_gcn(in_dim, h_dim, out_dim; dropout=0.5)
    return GCN(
        (GCNLayer(in_dim, h_dim), GCNLayer(h_dim, out_dim)),
        dropout
        )
end

function (m::GCN)(x, L_norm; training=false)
    # Hidden Layer + ReLU
    x = relu.(m.layers[1](x, L_norm))

    # Manual Dropout
    if training
        x = x .* (rand(Float32, size(x)) .> m.dropout_rate) ./ (1.0f0 - m.dropout_rate)
    end

    # Output Layer (Logits)
    return m.layers[2](x, L_norm)
end

# --- 3. Training Loop ---

function accuracy(y_pred, y_true)
    return mean(onecold(y_pred) .== onecold(y_true)) * 100
end

function train_gcn()
    features, targets, L_norm, (train_idx, val_idx, test_idx) = load_cora_sparse()

    # Cora usually uses 16 hidden units
    model = build_gcn(size(features, 1), 16, size(targets, 1); dropout=0.5)
    opt_state = Optimisers.setup(Optimisers.Adam(0.01), model)

    best_loss = Inf
    patience = 20
    counter = 0

    println("Training Sparse Flux GCN on Cora...")

    for epoch in 1:200
        # Gradient Step
        loss, grads = Flux.withgradient(model) do m
            logits = m(features, L_norm; training=true)
            Flux.Losses.logitcrossentropy(logits[:, train_idx], targets[:, train_idx])
        end

        opt_state, model = Optimisers.update!(opt_state, model, grads[1])

        # Validation
        logits_full = model(features, L_norm; training=false)
        val_loss = Flux.Losses.logitcrossentropy(logits_full[:, val_idx], targets[:, val_idx])
        val_acc = accuracy(logits_full[:, val_idx], targets[:, val_idx])

        @printf "Epoch %3d | Loss: %.4f | Val Loss: %.4f | Val Acc: %.2f%%\n" epoch loss val_loss val_acc

        # Early Stopping
        if val_loss < best_loss
            best_loss = val_loss
            counter = 0
        else
            counter += 1
            if counter >= patience
                @printf "Early stopping at epoch %d\n" epoch
                break
            end
        end
    end

    # Final Test
    final_logits = model(features, L_norm; training=false)
    @printf "--- Final Test Accuracy: %.2f%% ---\n" accuracy(final_logits[:, test_idx], targets[:, test_idx])

    return model
end

@time train_gcn()

