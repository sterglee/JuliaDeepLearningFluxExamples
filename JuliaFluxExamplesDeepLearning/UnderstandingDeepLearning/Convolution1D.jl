using Flux
using Statistics
using LinearAlgebra
using Plots

# ==========================================
# 1. Signal and Parameter Setup
# ==========================================
# Define the signal from the notebook
x_data = Float32[5.2, 5.3, 5.4, 5.1, 10.1, 10.3, 9.9, 10.3, 3.2, 3.4, 3.3, 3.1]
n = length(x_data)

# ==========================================
# 2. Manual Convolution Functions (TODOs)
# ==========================================

# Kernel size 3, Stride 1, Dilation 1 (Zero Padded)
function conv_3_1_1_zp(x_in, omega)
    n_in = length(x_in)
    x_out = zeros(Float32, n_in)
    for i in 1:n_in
        val = 0.0f0
        for j in 1:3
            # Index logic: center of kernel (j=2) aligns with i
            idx = i + (j - 2)
            if idx >= 1 && idx <= n_in
                val += x_in[idx] * omega[j]
            end
        end
        x_out[i] = val
    end
    return x_out
end

# Kernel size 3, Stride 2, Dilation 1 (Zero Padded)
function conv_3_2_1_zp(x_in, omega)
    n_out = Int(ceil(length(x_in) / 2))
    x_out = zeros(Float32, n_out)
    for i in 1:n_out
        val = 0.0f0
        # Stride logic: jumping by 2 indices in the input
        center_idx = (i - 1) * 2 + 1
        for j in 1:3
            idx = center_idx + (j - 2)
            if idx >= 1 && idx <= length(x_in)
                val += x_in[idx] * omega[j]
            end
        end
        x_out[i] = val
    end
    return x_out
end

# Kernel size 3, Stride 1, Dilation 2 (Zero Padded)
function conv_3_1_2_zp(x_in, omega)
    n_in = length(x_in)
    x_out = zeros(Float32, n_in)
    for i in 1:n_in
        val = 0.0f0
        # Dilation logic: spacing between elements is 2
        for j in 1:3
            idx = i + (j - 2) * 2
            if idx >= 1 && idx <= n_in
                val += x_in[idx] * omega[j]
            end
        end
        x_out[i] = val
    end
    return x_out
end

# ==========================================
# 3. Convolution as a Matrix (TODO)
# ==========================================

# Represents a 3x1x1 convolution as a Toeplitz-like matrix
function get_conv_mat_3_1_1_zp(n_out, omega)
    omega_mat = zeros(Float32, n_out, n_out)
    for i in 1:n_out
        for j in 1:3
            idx = i + (j - 2)
            if idx >= 1 && idx <= n_out
                omega_mat[i, idx] = omega[j]
            end
        end
    end
    return omega_mat
end

# ==========================================
# 4. Modern Flux.jl Implementation
# ==========================================

# Flux expects data as (Width, Channels, Batch)
x_flux = reshape(x_data, (n, 1, 1))

# Define a smoothing filter
# Weights format: (KernelWidth, InChannels, OutChannels)
w_smooth = reshape(Float32[0.33, 0.33, 0.33], (3, 1, 1))
b = [0.0f0]

# Modern Flux Layer
smoothing_layer = Conv(w_smooth, b, pad=SamePad())
h_flux = smoothing_layer(x_flux)

# ==========================================
# 5. Execution and Validation
# ==========================================

println("--- Manual Results ---")
omega_avg = Float32[0.33, 0.33, 0.33]
h_manual = conv_3_1_1_zp(x_data, omega_avg)
println("Sum of smoothed output: ", sum(h_manual), " (Target: 71.1)")

omega_edge = Float32[-0.5, 0.0, 0.5]
h_edge = conv_3_1_1_zp(x_data, omega_edge)
println("Edge detection complete.")

# Matrix check
omega_check = Float32[-1.0, 0.5, -0.2]
h_fn = conv_3_1_1_zp(x_data, omega_check)
mat = get_conv_mat_3_1_1_zp(n, omega_check)
h_mat = mat * x_data
println("Matrix multiplication matches function: ", isapprox(h_fn, h_mat))

    # ==========================================
    # 6. Visualization
    # ==========================================
    p1 = plot(x_data, color=:black, label="Original", title="Smoothing Convolution")
    plot!(p1, h_manual, color=:red, label="Smoothed")

    p2 = plot(x_data, color=:black, label="Original", title="Edge Detection")
    plot!(p2, h_edge, color=:blue, label="Edges")

    plot(p1, p2, layout=(2,1), size=(800, 600))

