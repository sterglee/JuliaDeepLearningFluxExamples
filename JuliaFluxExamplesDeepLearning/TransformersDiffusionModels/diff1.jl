using Flux, CUDA, Statistics, Images

# 1. Device Setup
# Automatically use GPU (CUDA) if available, otherwise fallback to CPU
device = CUDA.functional() ? gpu : cpu
@info "Execution Device: $device"

# 2. Model Definition (Conditional UNet)
# This structure allows the model to "know" which timestep 't' it is denoising
struct ConditionalUNet
    conv_in
    time_mlp
    conv_out
end

# The @functor macro allows Flux to move all internal layers to the GPU via |> device
Flux.@functor ConditionalUNet

# Forward Pass: Accepts image 'x' and timestep 't'
function (m::ConditionalUNet)(x, t)
    # Create a time vector on the same device as the input 'x'
    # This prevents the "unsafe_convert" MethodError
    t_vec = fill(Float32(t), (1, size(x, 4))) |> device
    t_embed = m.time_mlp(t_vec)
    
    # Inject time information into the feature map using Broadcasting
    # Reshaped to (1, 1, Channels, Batch) to align with Convolutional output
    h = m.conv_in(x) .+ reshape(t_embed, 1, 1, :, size(x, 4))
    return m.conv_out(h)
end

# Construct the model layers
model = ConditionalUNet(
    Conv((3,3), 3=>64, pad=1, swish),
    Chain(Dense(1 => 64, swish), Dense(64 => 64)),
    Conv((3,3), 64=>3, pad=1)
) |> device

# 3. Scheduler Parameters (Must stay on CPU for indexing)
# Keeping these on the CPU prevents the "Scalar indexing is disallowed" error
num_steps = 100
betas = collect(range(0.0001f0, 0.02f0, length=num_steps)) # CPU Array
alphas = 1.0f0 .- betas
alphas_cumprod = cumprod(alphas)

# 4. Sampling Loop (DDPM Algorithm)
println("Starting Image Generation...")

# Initialize with pure Gaussian noise (Width, Height, Channels, Batch)
image = randn(Float32, 128, 128, 3, 1) |> device

for t in reverse(1:num_steps)
    # Access scalar values from CPU (Safe indexing)
    α = alphas[t]
    ᾱ = alphas_cumprod[t]
    β = betas[t]
    
    # Predict noise using the UNet (GPU computation)
    noise_pred = model(image, t)
    
    # Calculate x_{t-1} using the DDPM mathematical formula
    # x_{t-1} = (1/√α) * (x_t - (β / √(1-ᾱ)) * noise_pred)
    direction = (β / sqrt(1.0f0 - ᾱ)) .* noise_pred
    image = (1.0f0 / sqrt(α)) .* (image .- direction)
    
    # Add Langevin noise for all steps except the final one (t=1)
    if t > 1
        image = image .+ sqrt(β) .* (randn(Float32, size(image)...) |> device)
    end
    
    if t % 20 == 0
        println("Progress: Step $t / $num_steps")
    end
end

# 5. Post-Processing and Saving
# Transfer result back to CPU for image conversion
cpu_tensor = cpu(image[:, :, :, 1]) 

# Permute dimensions: From (W, H, C) to (C, W, H) as required by Images.jl
permuted_tensor = permutedims(cpu_tensor, (3, 1, 2)) 

# Re-scale from [-1, 1] to [0, 1] and clamp to avoid out-of-bounds pixels
final_img = colorview(RGB, clamp.(permuted_tensor .* 0.5f0 .+ 0.5f0, 0, 1))

# Save the final result
save("final_diffusion_output.png", final_img)
println("Success! Image saved as 'final_diffusion_output.png'")