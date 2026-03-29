using Flux, CUDA, Images, Statistics, MLUtils

# ---------------------------------------------------------
# 1. Architecture: The UNet
# ---------------------------------------------------------
struct ConvBlock
    chain::Chain
end
Flux.@functor ConvBlock

function ConvBlock(kernel::Tuple{Int,Int}, in_ch::Int, out_ch::Int; act=swish, pad=1)
    return ConvBlock(Chain(Conv(kernel, in_ch => out_ch; pad=pad), act))
end
(m::ConvBlock)(x) = m.chain(x)

struct DiffusionUNet
    down::Chain
    bottleneck::Chain
    up::Chain
    final::Conv
end
Flux.@functor DiffusionUNet

function DiffusionUNet()
    down = Chain(ConvBlock((3,3), 3, 32), ConvBlock((3,3), 32, 64))
    bottleneck = Chain(ConvBlock((3,3), 64, 128))
    up = Chain(ConvBlock((3,3), 128, 64), ConvBlock((3,3), 64, 32))
    final = Conv((1,1), 32 => 3)
    return DiffusionUNet(down, bottleneck, up, final)
end

function (m::DiffusionUNet)(x, t)
    # Simple forward pass (Skip connections omitted for brevity)
    d = m.down(x)
    b = m.bottleneck(d)
    u = m.up(b)
    return m.final(u)
end

# ---------------------------------------------------------
# 2. Model Initialization (Fixes your Error)
# ---------------------------------------------------------
# Define the model instance and move it to GPU
model = DiffusionUNet() |> gpu

# NOW this will work:
Flux.testmode!(model) 

# ---------------------------------------------------------
# 3. Scheduler & Sampling Logic
# ---------------------------------------------------------
struct MyScheduler
    betas::Vector{Float32}
    alphas::Vector{Float32}
    alphas_cumprod::Vector{Float32}
end

steps = 1000
β_vec = collect(range(1f-4, 0.02f0, length=steps))
α_vec = 1f0 .- β_vec
ᾱ_vec = cumprod(α_vec)
scheduler = MyScheduler(β_vec, α_vec, ᾱ_vec)

function scheduler_step(scheduler, noise_pred, t, x_t)
    β, α, ᾱ = scheduler.betas[t], scheduler.alphas[t], scheduler.alphas_cumprod[t]
    coeff = 1.0f0 / sqrt(α)
    noise_coeff = β / sqrt(1.0f0 - ᾱ)
    
    mean_xt_prev = coeff .* (x_t .- noise_coeff .* noise_pred)
    
    if t > 1
        noise = randn(Float32, size(x_t)) |> gpu
        return mean_xt_prev .+ sqrt(β) .* noise
    end
    return mean_xt_prev
end

# ---------------------------------------------------------
# 4. Iterative Sampling
# ---------------------------------------------------------
sample = randn(Float32, 64, 64, 3, 4) |> gpu

println("Sampling...")
for t in reverse(1:steps)
    # Model predicts the noise present in 'sample' at time 't'
    noise_pred = model(sample, t)
    
    # Step back one level of noise
    sample = scheduler_step(scheduler, noise_pred, t, sample)
    
    if t % 250 == 0
        @info "Step $t complete"
    end
end

# Save result
final_output = cpu(sample)
final_output = clamp.(final_output .* 0.5f0 .+ 0.5f0, 0, 1)
save("output.png", colorview(RGB, permutedims(final_output[:,:,:,1], (3,1,2))))
println("Done! Check output.png")

