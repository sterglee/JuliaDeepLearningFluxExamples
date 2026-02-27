function graph_attention(X, ω, β, ϕ, A)
    # 1. Transform input features
    X_transformed = (X * ω) .+ β'

# 2. Compute Attention Scores
# In GAT, we compute scores for edges that exist in A
scores = X_transformed * ϕ  # Simple dot product projection

# 3. Masking: Only keep scores for connected nodes
# We apply softmax only over neighbors
adj_mask = A .> 0
raw_att = (X_transformed * X_transformed') ./ sqrt(size(X, 2))
masked_att = exp.(raw_att) .* adj_mask

# Normalize per row (node)
attention_coeffs = masked_att ./ sum(masked_att, dims=2)

# 4. Weighted sum of features
return relu.(attention_coeffs * X_transformed)
end

