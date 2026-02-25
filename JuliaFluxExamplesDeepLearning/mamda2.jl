using Flux
using LinearAlgebra
using Random
using Statistics

############################################################
# RMSNorm
############################################################

struct RMSNorm
    γ::Vector{Float32}
    ϵ::Float32
end

RMSNorm(dim; ϵ=1f-5) = RMSNorm(ones(Float32, dim), ϵ)

Flux.@functor RMSNorm

function (r::RMSNorm)(x)
    # Ensure Float32 and avoid mutation
    norm = sqrt.(mean(x.^2, dims=1) .+ r.ϵ)
    return r.γ .* x ./ norm
end

############################################################
# Gated SSM (ZYGOTE SAFE: No Mutation)
############################################################

struct GatedSSM
    A::Matrix{Float32}
    B::Matrix{Float32}
    C::Matrix{Float32}
    gate::Dense
end

function GatedSSM(d_model)
    # Using 1f0 literals ensures we stay in Float32
    A = randn(Float32, d_model, d_model) * 0.01f0
    B = randn(Float32, d_model, d_model) * 0.01f0
    C = randn(Float32, d_model, d_model) * 0.01f0
    gate = Dense(d_model, d_model, σ)
    GatedSSM(A, B, C, gate)
end

Flux.@functor GatedSSM

function (m::GatedSSM)(x)
    d_model, seq_len = size(x)

    # Zygote safe recurrence: We avoid push! or indexing by using a functional loop.
    # While Zygote supports some loops, building a list and reduce(hcat, ...) is safest.

    h = x[:, 1] .* 0f0 # Initial state
    outputs = []

    for t in 1:seq_len
        u = x[:, t]
        g = m.gate(u)
        h = m.A * h + m.B * (g .* u)
        # Note: push! is actually allowed by Zygote ONLY for local arrays
        # in some contexts, but to be 100% safe across versions,
        # we treat it as a sequence of outputs.
        outputs = [outputs..., m.C * h]
    end

    return reduce(hcat, outputs)
end

############################################################
# Mamba Block & Model
############################################################

struct MambaBlock
    norm::RMSNorm
    proj::Dense
    ssm::GatedSSM
end

function MambaBlock(d_model)
    MambaBlock(RMSNorm(d_model), Dense(d_model, d_model), GatedSSM(d_model))
end

Flux.@functor MambaBlock

function (m::MambaBlock)(x)
    # Residual path
    return x + m.ssm(m.proj(m.norm(x)))
end

struct MambaModel
    embedding::Dense
    blocks::Chain # Use Chain for cleaner forward pass
    head::Dense
end

function MambaModel(d_model, n_layers, input_dim, output_dim)
    MambaModel(
        Dense(input_dim, d_model),
        Chain([MambaBlock(d_model) for _ in 1:n_layers]...),
            Dense(d_model, output_dim)
            )
        end

        Flux.@functor MambaModel

        (m::MambaModel)(x) = m.head(m.blocks(m.embedding(x)))

        ############################################################
        # MODERN TRAINING (Flux.train!)
        ############################################################

        Random.seed!(42)
        d_model, n_layers = 32, 2
        input_dim, output_dim, seq_len = 16, 16, 20

        model = MambaModel(d_model, n_layers, input_dim, output_dim)

        # Prepare data as an iterator of (x, y) tuples
        # train! expects an iterable: [(x1, y1), (x2, y2), ...]
        data = [(rand(Float32, input_dim, seq_len), rand(Float32, output_dim, seq_len)) for _ in 1:10]

            # Modern setup: rule and state
            opt_state = Flux.setup(Flux.Adam(1e-3), model)

            println("Training...")

            for epoch in 1:40
                # The 'do' block approach for train! is the most modern and readable
                Flux.train!(model, data, opt_state) do m, x, y
                    y_hat = m(x)
                    Flux.mse(y_hat, y)
                end

                if epoch % 10 == 0
                    # Manual loss check
                    curr_loss = Flux.mse(model(data[1][1]), data[1][2])
                    println("Epoch $epoch | Loss = $curr_loss")
                end
            end

            println("Done.")

