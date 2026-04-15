using Flux
using Flux: onehotbatch, logitbinarycrossentropy, DataLoader, @functor
using Statistics, Random, Printf

# --- 1. CONFIG ---
const SEQ_LEN = 250
const VOCAB_SIZE = 4 # A, C, G, T (ignoring N for simplicity)
const LATENT_DIM = 100
const BATCH_SIZE = 32

# --- 2. DATA UTILITY ---
function get_real_batch(batch_size=BATCH_SIZE)
    # Simulating a batch of real one-hot DNA: (VOCAB, SEQ, BATCH)
    # In a real run, this would pull from your human_nontata_promoters dataset
    data = rand(1:VOCAB_SIZE, SEQ_LEN, batch_size)
    return Float32.(onehotbatch(data, 1:VOCAB_SIZE))
end

# --- 3. MODELS ---
function build_generator()
    return Chain(
        Dense(LATENT_DIM, 256, relu),
        Dense(256, 512, relu),
        Dense(512, VOCAB_SIZE * SEQ_LEN, tanh),
        x -> reshape(x, VOCAB_SIZE, SEQ_LEN, :)
    )
end

function build_discriminator()
    return Chain(
        Flux.flatten,
        Dense(VOCAB_SIZE * SEQ_LEN, 512, leakyrelu),
        Dense(512, 256, leakyrelu),
        Dense(256, 1) # Probability of being real
    )
end

# --- 4. STABLE TRAINING LOOP ---
function train_gan(epochs=50)
    gen = build_generator()
    dis = build_discriminator()
    
    # Specific GAN hyperparams: Lower learning rate, specific Betas
    opt_gen = Flux.setup(Adam(0.0001, (0.5, 0.999)), gen)
    opt_dis = Flux.setup(Adam(0.0001, (0.5, 0.999)), dis)

    println("Starting GAN Competition: Generator vs. Discriminator...")

    for epoch in 1:epochs
        # --- Train Discriminator ---
        real_data = get_real_batch()
        noise = randn(Float32, LATENT_DIM, BATCH_SIZE)
        
        loss_d, grads_d = Flux.withgradient(dis) do d_net
            fake_data = gen(noise)
            # Binary Cross Entropy: Real=1, Fake=0
            real_loss = logitbinarycrossentropy(d_net(real_data), 1)
            fake_loss = logitbinarycrossentropy(d_net(fake_data), 0)
            return real_loss + fake_loss
        end
        Flux.update!(opt_dis, dis, grads_d[1])

        # --- Train Generator ---
        loss_g, grads_g = Flux.withgradient(gen) do g_net
            # Generator wants Discriminator to think fakes are Real (1)
            fake_data = g_net(noise)
            return logitbinarycrossentropy(dis(fake_data), 1)
        end
        Flux.update!(opt_gen, gen, grads_g[1])

        if epoch % 10 == 0
            @printf("Epoch %d | D_Loss: %.4f | G_Loss: %.4f\n", epoch, loss_d, loss_g)
        end
    end
    return gen
end

# Run training
generator = train_gan()

