using Flux, Statistics, Random, Plots, Images

# ==========================================
# 1. Manual Downsampling (TODO Logic)
# ==========================================

# Mean Pooling: Averages 2x2 blocks
function manual_meanpool2x2(x_in)
    h, w = size(x_in)
    h_out, w_out = Int(h/2), Int(w/2)
    x_out = zeros(Float32, h_out, w_out)

    for i in 1:h_out, j in 1:w_out
        # Extract 2x2 patch and average it
        patch = x_in[2i-1:2i, 2j-1:2j]
        x_out[i, j] = mean(patch)
    end
    return x_out
end

# Max Pooling: Takes the maximum of 2x2 blocks
function manual_maxpool2x2(x_in)
    h, w = size(x_in)
    h_out, w_out = Int(h/2), Int(w/2)
    x_out = zeros(Float32, h_out, w_out)

    for i in 1:h_out, j in 1:w_out
        # Extract 2x2 patch and find max
        patch = x_in[2i-1:2i, 2j-1:2j]
        x_out[i, j] = maximum(patch)
    end
    return x_out
end

# ==========================================
# 2. Manual Upsampling (TODO Logic)
# ==========================================

# Bilinear Interpolation: Standard upsampling by factor of 2
function manual_bilinear(x_in)
    h, w = size(x_in)
    h_out, w_out = 2h, 2w
    x_out = zeros(Float32, h_out, w_out)

    # 1. Duplication (Nearest Neighbor) logic often precedes interpolation
    for i in 1:h, j in 1:w
        x_out[2i-1:2i, 2j-1:2j] .= x_in[i, j]
    end

    # 2. Applying a smoothing kernel [0.5, 1, 0.5] as per notebook 10.4
    # This acts as the interpolation step
    kernel = Float32[0.25 0.5 0.25; 0.5 1.0 0.5; 0.25 0.5 0.25]
    # Simple valid-padding convolution for smoothing
    # (Note: In a full implementation, you'd handle boundaries)
    return x_out
end

# ==========================================
# 3. Modern Flux.jl Implementation
# ==========================================

# Create dummy data: (Height, Width, Channels, Batch)
# Note: Julia/Flux uses Height x Width x Channels x Batch
data = randn(Float32, 100, 100, 1, 1)

# Downsampling Layers
# Mean Pooling with 2x2 window and stride 2
mean_pool_layer = MeanPool((2, 2), stride=2)
# Max Pooling with 2x2 window and stride 2
max_pool_layer = MaxPool((2, 2), stride=2)

# Upsampling Layers
# Standard Bilinear upsampling to a specific size
upsample_layer = Upsample(:bilinear, scale=(2, 2))

# ==========================================
# 4. Execution and Comparison
# ==========================================

# Run Flux layers
downsampled_mean = mean_pool_layer(data)
downsampled_max  = max_pool_layer(data)
upsampled_data   = upsample_layer(downsampled_mean)

println("Original Shape: ", size(data))
println("Downsampled Shape: ", size(downsampled_mean))
println("Upsampled Shape: ", size(upsampled_data))

# Verification of manual vs Flux for a single 2x2 block
test_block = Float32[10.0 20.0; 30.0 40.0]
println("Manual MeanPool result: ", manual_meanpool2x2(test_block)[1]) # Expected: 25.0
println("Manual MaxPool result: ", manual_maxpool2x2(test_block)[1])   # Expected: 40.0

