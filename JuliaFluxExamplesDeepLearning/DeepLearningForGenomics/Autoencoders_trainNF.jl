using Flux
using Statistics
using Random
using Distributions
using Plots

# -----------------------------
# 1. DATA
# -----------------------------
function get_synthetic_data(n_samples=1000, n_features=500)
    Random.seed!(42)
    return Float32.(rand(Beta(0.5, 2.0), n_features, n_samples))
end

# -----------------------------
# 2. COUPLING LAYER (NO MUTATION)
# -----------------------------
struct CouplingLayer
    mask::Vector{Bool}
    s_net
    t_net
end

Flux.@functor CouplingLayer

function CouplingLayer(dim::Int, hidden_dim::Int)
    mask = rand(Bool, dim)

    s_net = Chain(
        Dense(dim => hidden_dim, swish),
        Dense(hidden_dim => hidden_dim, swish),
        Dense(hidden_dim => dim)
        )

    t_net = Chain(
        Dense(dim => hidden_dim, swish),
        Dense(hidden_dim => hidden_dim, swish),
        Dense(hidden_dim => dim)
        )

    return CouplingLayer(mask, s_net, t_net)
end

function (l::CouplingLayer)(x)
    mask = reshape(Float32.(l.mask), :, 1)

    x_masked = x .* mask

    s = l.s_net(x_masked)
    t = l.t_net(x_masked)

    # --- NO MUTATION ---
    y = x_masked .+ (1 .- mask) .* (x .* exp.(s) .+ t)

    log_det = sum(s .* (1 .- Float32.(l.mask)), dims=1)

    return y, log_det
end

function inverse(l::CouplingLayer, y)
    mask = reshape(Float32.(l.mask), :, 1)

    y_masked = y .* mask

    s = l.s_net(y_masked)
    t = l.t_net(y_masked)

    x = y_masked .+ (1 .- mask) .* ((y .- t) .* exp.(-s))

    log_det = -sum(s .* (1 .- Float32.(l.mask)), dims=1)

    return x, log_det
end

# -----------------------------
# 3. FLOW MODEL
# -----------------------------
struct NormalizingFlow
    layers::Vector{CouplingLayer}
end

Flux.@functor NormalizingFlow

function NormalizingFlow(dim::Int, n_layers::Int=4, hidden_dim::Int=256)
    layers = [CouplingLayer(dim, hidden_dim) for _ in 1:n_layers]
        return NormalizingFlow(layers)
    end

    function (m::NormalizingFlow)(x)
        log_det_total = zeros(Float32, 1, size(x, 2))

        h = x

        for layer in m.layers
            h, log_det = layer(h)
            log_det_total = log_det_total .+ log_det   # NO MUTATION
        end

        return h, log_det_total
    end

    function inverse(m::NormalizingFlow, z)
        h = z

        for layer in reverse(m.layers)
            h, _ = inverse(layer, h)
        end

        return h
    end

    # -----------------------------
    # 4. LOG PROBABILITY
    # -----------------------------
    function log_prob(model, x)
        z, log_det = model(x)

        log_pz = sum(logpdf.(Normal(0,1), z); dims=1)

        return log_pz .+ log_det
    end

    # -----------------------------
    # 5. LOSS
    # -----------------------------
    loss(model, x) = -mean(log_prob(model, x))

    # -----------------------------
    # 6. TRAINING
    # -----------------------------
    function main()
        X = get_synthetic_data()
        dim = size(X, 1)

        model = NormalizingFlow(dim, 4, 256)
        opt = Flux.setup(Adam(1e-3), model)

        loader = Flux.DataLoader(X, batchsize=64, shuffle=true)

        println("Training Normalizing Flow...")

        for epoch in 1:10
            total = 0f0

            for batch in loader
                l, grads = Flux.withgradient(model) do m
                    loss(m, batch)
                end

                Flux.update!(opt, model, grads[1])
                total += l
            end

            println("Epoch $epoch | Loss: $(total / length(loader))")
        end

        # -----------------------------
        # 7. SAMPLING
        # -----------------------------
        println("Generating samples...")

        z = randn(Float32, dim, 1)
        x_gen = inverse(model, z)

        # -----------------------------
        # 8. PLOT
        # -----------------------------
        p = plot(X[:,1], label="Real", alpha=0.5, title="Normalizing Flow (Fixed)")
        plot!(x_gen[:,1], label="Generated", linestyle=:dash)
        display(p)
    end

    main()


