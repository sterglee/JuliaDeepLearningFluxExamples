using Flux
using Statistics

# =========================
# DATA
# =========================
real_data(n) = randn(Float32, n) * 2 .+ 3

# =========================
# MODELS
# =========================
G = Chain(
    Dense(10, 16, relu),
    Dense(16, 1)
    )

D = Chain(
    Dense(1, 16, relu),
    Dense(16, 1),
    σ
    )

# =========================
# OPTIMIZERS + STATE (IMPORTANT FIX)
# =========================
optG = ADAM(0.001)
optD = ADAM(0.001)

stG = Flux.setup(optG, G)
stD = Flux.setup(optD, D)

# =========================
# LOSSES
# =========================
function d_loss(real, fake)
    r = D(real)
    f = D(fake)
    return -mean(log.(r .+ 1e-8) .+ log.(1 .- f .+ 1e-8))
end

function g_loss(z)
    fake = G(z)
    return -mean(log.(D(fake) .+ 1e-8))
end

# =========================
# TRAINING
# =========================
epochs = 2000
batch_size = 32

for epoch in 1:epochs

    real = reshape(real_data(batch_size), 1, batch_size)
    z = randn(Float32, 10, batch_size)

    fake = G(z)

    # -------------------
    # Discriminator step
    # -------------------
    gradsD = gradient(() -> d_loss(real, fake), D)
    Flux.update!(stD, D, gradsD)

    # -------------------
    # Generator step
    # -------------------
    gradsG = gradient(() -> g_loss(z), G)
    Flux.update!(stG, G, gradsG)

    if epoch % 200 == 0
        println("Epoch $epoch | D loss = $(d_loss(real, fake))")
    end
end

# =========================
# TEST
# =========================
z_test = randn(Float32, 10, 100)
samples = G(z_test)

println("Generated mean: ", mean(samples))
println("Real mean approx: 3.0")

