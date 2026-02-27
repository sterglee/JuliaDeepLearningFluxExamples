using Flux, Statistics, Random, Plots

# ==========================================
# 1. Manual 2D Convolution (Notebook Logic)
# ==========================================
# This function replicates the full batch/multi-channel logic
# from the notebook.
function manual_conv2d(input_image, weights, stride=1, padding=1)
    # input_image shape: (Height, Width, Channels_In, Batch)
    # weights shape: (Kernel_H, Kernel_W, Channels_In, Channels_Out)
    (h_in, w_in, c_in, n_batch) = size(input_image)
    (k_h, k_w, _, c_out) = size(weights)

    # 1. Padding the input with zeros
    img_padded = zeros(Float32, h_in + 2padding, w_in + 2padding, c_in, n_batch)
    img_padded[(padding+1):(padding+h_in), (padding+1):(padding+w_in), :, :] .= input_image

    # 2. Calculate output dimensions based on stride
    h_out = Int(floor((h_in + 2padding - k_h) / stride)) + 1
    w_out = Int(floor((w_in + 2padding - k_w) / stride)) + 1

    output = zeros(Float32, h_out, w_out, c_out, n_batch)

    # 3. Sliding window operation
    for b in 1:n_batch, co in 1:c_out, i in 1:h_out, j in 1:w_out
        h_start = (i - 1) * stride + 1
        w_start = (j - 1) * stride + 1

        # Extract the image patch across all input channels
        patch = img_padded[h_start:h_start+k_h-1, w_start:w_start+k_w-1, :, b]

        # Element-wise multiplication and sum
        output[i, j, co, b] = sum(patch .* weights[:, :, :, co])
    end
    return output
end

# ==========================================
# 2. Setup and Execution
# ==========================================
# Parameters from the notebook example
Random.seed!(1)
n_batch, h, w, c_in, c_out, k_size = 2, 4, 6, 5, 2, 3

# Flux data format: (Height, Width, Channels, Batch)
input_data = randn(Float32, h, w, c_in, n_batch)

# Define Flux Conv layer
# (k, k) kernel, in => out channels, pad=1
flux_layer = Conv((k_size, k_size), c_in => c_out, stride=1, pad=1, bias=false)

# Run Flux version
output_flux = flux_layer(input_data)

# Run Manual version using weights from Flux layer
output_manual = manual_conv2d(input_data, flux_layer.weight, 1, 1)

# ==========================================
# 3. Validation
# ==========================================
println("Flux Output Shape: ", size(output_flux))
println("Manual Output Shape: ", size(output_manual))

# Check if results match
if isapprox(output_flux, output_manual, atol=1e-5)
    println("SUCCESS: Manual 2D convolution matches Flux implementation.")
else
    println("FAILURE: Discrepancy detected between implementations.")
end

