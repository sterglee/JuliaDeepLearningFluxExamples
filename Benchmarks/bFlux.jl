import Flux

# Define and initialize the neural network
MT = Flux.MersenneTwister(1234)
flux_model = Flux.Chain(
    Flux.Dense(1, 9, Flux.relu, init=Flux.randn32(MT)),
    Flux.Dense(9, 9, Flux.relu, init=Flux.randn32(MT)),
    Flux.Dense(9, 1, Flux.relu, init=Flux.randn32(MT)),
    )

# Save and display initial values
initial_parameters = deepcopy(Flux.state(flux_model))
@show initial_parameters

# Define training data
X = Float32.(-10:1.0:10)'
Y = X .^ 2

# Define loss function
flux_loss_function(model, X) = Flux.Losses.mse(model(X), Y)

# Define optimizer
flux_optimizer = Flux.setup(Flux.Adam(), flux_model)

# Train and display progress
println("\ni   \tloss")
println("-----\t----------")
@time for i in 0:15000
    if rem(i,1000) == 0
        loss = flux_loss_function(flux_model, X)
        println("$i\t$loss")
    end
    Flux.train!(flux_loss_function, flux_model, (X,), flux_optimizer)
end

# Trained model
f(x) = sum(flux_model([x]))

# Plotting
using CairoMakie
fig = Figure(size=(420,300), fontsize=11.5, backgroundcolor=:transparent)
axis = Axis(fig[1,1], xlabel=L"$x$", ylabel=L"$y$", xlabelsize=16.5, ylabelsize=16.5)
            lines!(axis, -10..10, x -> f(x), label="model")
            scatter!(axis, X', x -> x^2, color=:black, label="exact")
            axislegend(axis, position=:rb, framevisible=false)
            fig

