using Flux, MLDatasets, Statistics, Dates
using Flux: Conv, ConvTranspose, BatchNorm, Dense, flatten, leakyrelu, relu, SamePad
using Base.Threads


# 0. Parameters
latent_dim = 100
batch_size = 128
epochs = 20
lr = 0.0002f0
beta1 = 0.5f0

# 1. Prepare Data
data = FashionMNIST(split = :train)

x_train = Float32.(data.features) ./ 255f0

# reshape to (W,H,C,N)
x_train = reshape(x_train, 28, 28, 1, :)

loader = Flux.DataLoader(x_train, batchsize=batch_size, shuffle=true)

# 2. Generator
generator = Chain(
    Dense(latent_dim, 7 * 7 * 128),
    x -> reshape(x, 7, 7, 128, :),
    BatchNorm(128),
    relu,
    ConvTranspose((4,4), 128 => 64, stride=2, pad=SamePad()),
    BatchNorm(64),
    relu,
    ConvTranspose((4,4), 64 => 1, stride=2, pad=SamePad()),
    x -> tanh.(x)
)

# 3. Discriminator
discriminator = Chain(
    Conv((4,4), 1 => 64, stride=2, pad=SamePad()),
    leakyrelu,
    Conv((4,4), 64 => 128, stride=2, pad=SamePad()),
    BatchNorm(128),
    leakyrelu,
    flatten,
    Dense(7 * 7 * 128, 1),
    x -> sigmoid.(x)
)

# 4. Optimizers
opt_g = Flux.setup(Flux.Adam(lr, (beta1, 0.999)), generator)
opt_d = Flux.setup(Flux.Adam(lr, (beta1, 0.999)), discriminator)

# Helper for BCE targets
ones_like(x) = ones(Float32, size(x))
zeros_like(x) = zeros(Float32, size(x))

# 5. Training
total_time = @elapsed begin

for epoch in 1:epochs

    epoch_time = @elapsed begin

    for x_real in loader

        b = size(x_real,4)

        noise = randn(Float32, latent_dim, b)


        # -----------------
        # Train Discriminator
        # -----------------

        loss_d, grad_d = Flux.withgradient(discriminator) do d

            y_real = d(x_real)

            fake = generator(noise)
            y_fake = d(fake)

            loss_real = Flux.binarycrossentropy(y_real, ones_like(y_real))
            loss_fake = Flux.binarycrossentropy(y_fake, zeros_like(y_fake))

            loss_real + loss_fake
        end

        Flux.update!(opt_d, discriminator, grad_d)

        # -----------------
        # Train Generator
        # -----------------

        loss_g, grad_g = Flux.withgradient(generator) do g

            fake = g(noise)
            y_fake = discriminator(fake)

            Flux.binarycrossentropy(y_fake, ones_like(y_fake))

        end

        Flux.update!(opt_g, generator, grad_g)

    end

    end

    println("Epoch $epoch | Time: $(round(epoch_time,digits=2)) s")

end

end

# 6. Performance Summary
println("\n" * "="^35)
println("Final Performance Summary")
println("-"^35)
println("Total Training Time:    $(round(total_time,digits=2)) s")
println("Average Time per Epoch: $(round(total_time/epochs,digits=2)) s")
println("="^35)

