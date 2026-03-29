using Images, Augmentor, Flux, MLUtils, Plots

# ---------------------------------------------------------
# 1. Data Loading & Preprocessing
# ---------------------------------------------------------

"""
Loads all images from a directory, applies augmentation, 
and returns a Float32 4D Tensor: (Channels, Width, Height, Batch)
"""
function load_butterfly_dataset(folder_path; img_size=64)
    # Get all image paths (jpg, png, jpeg)
    extensions = [".jpg", ".png", ".jpeg"]
    img_paths = readdir(folder_path, join=true)
    img_paths = filter(p -> lowercase(splitext(p)[2]) in extensions, img_paths)
    
    # Define Augmentation: Flip and Resize
    # Note: We handle Float32 conversion via Julia broadcasting later
    pipeline = FlipX(0.5) + Resize(img_size, img_size)
    
    processed_images = []
    
    for path in img_paths
        img = load(path)
        # Apply augmentation (returns a resized/flipped image)
        augmented = augment(img, pipeline)
        
        # Convert to Float32 and Channel-First (C, W, H)
        # Normalizing to [-1, 1] for Diffusion
        data = Float32.(channelview(augmented))
        data_norm = (data .- 0.5f0) ./ 0.5f0
        
        push!(processed_images, data_norm)
    end
    
    # Concatenate into a 4D Tensor: (C, W, H, N)
    return cat(processed_images..., dims=4)
end

# ---------------------------------------------------------
# 2. Linear Noise Scheduler (DDPM)
# ---------------------------------------------------------



struct DDPMScheduler
    betas::Vector{Float32}
    alphas::Vector{Float32}
    alphas_cumprod::Vector{Float32}
end

function DDPMScheduler(steps=1000, b_start=0.0001f0, b_end=0.02f0)
    betas = collect(range(b_start, b_end, length=steps))
    alphas = 1.0f0 .- betas
    alphas_cumprod = cumprod(alphas)
    return DDPMScheduler(betas, alphas, alphas_cumprod)
end

"""
Forward Diffusion Process (Adding noise)
x_t = sqrt(α_t_bar) * x_0 + sqrt(1 - α_t_bar) * ε
"""
function add_noise(scheduler, x_0, noise, t)
    sqrt_alpha_cumprod = sqrt(scheduler.alphas_cumprod[t])
    sqrt_one_minus_alpha_cumprod = sqrt(1.0f0 - scheduler.alphas_cumprod[t])
    return sqrt_alpha_cumprod .* x_0 .+ sqrt_one_minus_alpha_cumprod .* noise
end

# ---------------------------------------------------------
# 3. Execution & Visualization
# ---------------------------------------------------------

# 1. Initialize Scheduler
scheduler = DDPMScheduler(1000)

# 2. Mock Data (Or use load_butterfly_dataset("your_path"))
# Creating a dummy 64x64 RGB image batch of 1
x_0 = randn(Float32, 3, 64, 64, 1) 

# 3. Generate noise versions for visualization
timesteps = [1, 100, 250, 500, 750, 1000]
noise = randn(Float32, size(x_0))

# Displaying the noise progression
# Reverse normalization for viewing: (x * 0.5 + 0.5)
function get_displayable_image(tensor)
    # Convert CHW back to WHC for plotting
    img_data = permutedims(tensor[:, :, :, 1], (2, 3, 1))
    return colorview(RGB, clamp.(img_data .* 0.5f0 .+ 0.5f0, 0, 1))
end

println("Diffusion steps generated for timesteps: $timesteps")

