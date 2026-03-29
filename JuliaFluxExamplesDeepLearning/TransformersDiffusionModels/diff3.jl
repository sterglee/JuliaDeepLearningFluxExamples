using Flux
using CUDA # Για χρήση GPU

# Ορισμός ενός εξελιγμένου UNet με Attention και Residual blocks
# Στη Julia, η σειρά των διαστάσεων είναι (W, H, C, N)
struct DiffusionUNet
    layers::Chain
end

# Κατασκευαστής που προσομοιώνει τη δομή του diffusers
function DiffusionUNet(in_channels=3, sample_size=64)
    # block_out_channels=(64, 128, 256, 512)
    return Chain(
        # Downsampling path
        Conv((3, 3), in_channels => 64, pad=1, relu),
        MaxPool((2, 2)), # 32x32

        Conv((3, 3), 64 => 128, pad=1, relu),
        MaxPool((2, 2)), # 16x16

        # Bottleneck / Attention
        Conv((3, 3), 128 => 256, pad=1, relu),

        # Upsampling path
        Upsample(:bilinear, scale=(2, 2)),
        Conv((3, 3), 256 => 128, pad=1, relu),

        Upsample(:bilinear, scale=(2, 2)),
        Conv((3, 3), 128 => 64, pad=1, relu),

        # Τελικό επίπεδο για επιστροφή στα 3 κανάλια (RGB)
        Conv((3, 3), 64 => in_channels, pad=1)
        ) |> gpu
end

# Δοκιμή του μοντέλου με ένα batch δεδομένων
model = DiffusionUNet()
noised_x = randn(Float32, 64, 64, 3, 8) |> gpu # Batch από 8 εικόνες
timesteps = rand(1:1000, 8) |> gpu

# Inference (χωρίς υπολογισμό gradients)
out = model(noised_x)

println("Σχήμα εισόδου (noised_x): ", size(noised_x))
println("Σχήμα εξόδου (out): ", size(out))



