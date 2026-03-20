using Flux, Statistics

struct MyModel
    embed; rnn; fc
end
Flux.@functor MyModel

function (m::MyModel)(x)
    Flux.reset!(m.rnn)
    x_emb = m.embed(x) 
    h_all = m.rnn(x_emb) 
    # Slicing the last time step to match target y shape (1, 32)
    return m.fc(h_all[:, end, :])
end

# Init with Pair syntax
model = MyModel(
    Embedding(1000 => 64),
    LSTM(64 => 128),
    Dense(128 => 1)
)

X = rand(1:1000, 10, 32)
y = rand(Float32, 1, 32)

opt_state = Flux.setup(Adam(1e-3), model)

for epoch in 1:5
    val, grads = Flux.withgradient(model) do m
        Flux.mse(m(X), y)
    end
    Flux.update!(opt_state, model, grads[1])
    println("Loss: $val")
end

