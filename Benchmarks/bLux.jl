import Lux
import Random
import Optimisers

using Flux
MT = Flux.MersenneTwister(1234)
flux_model = Flux.Chain(
    Flux.Dense(1, 9, Flux.relu, init=Flux.randn32(MT)),
    Flux.Dense(9, 9, Flux.relu, init=Flux.randn32(MT)),
    Flux.Dense(9, 1, Flux.relu, init=Flux.randn32(MT)),
    )

# Save and display initial values
initial_parameters = deepcopy(Flux.state(flux_model))


# Define and initialize the neural network
lux_model = Lux.Chain(
    Lux.Dense(1 => 9, Lux.relu),
    Lux.Dense(9 => 9, Lux.relu),
    Lux.Dense(9 => 1, Lux.relu),
    )
MT = Random.MersenneTwister(1234)
parameters, state = Lux.setup(MT, lux_model)

# Load and display initial values from Flux.jl
parameters = (
    layer_1 = (weight = initial_parameters[1][1].weight, bias = initial_parameters[1][1].bias),
    layer_2 = (weight = initial_parameters[1][2].weight, bias = initial_parameters[1][2].bias),
    layer_3 = (weight = initial_parameters[1][3].weight, bias = initial_parameters[1][3].bias),
    )
@show parameters

# Define training data
X = Float32.(-10:1.0:10)' # Note that a transpose is applied here
Y = X .^ 2

# Define loss function
lux_loss_function = Lux.MSELoss()

# Define optimizer
lux_optimizer = Optimisers.Adam()

# Train and display progress
println("\ni   \tloss")
println("-----\t----------")
gtstate = Lux.Training.TrainState(lux_model, parameters, state, lux_optimizer)
@time for i in 0:15000
    grads, loss, _, tstate = Lux.Training.single_train_step!(Lux.AutoZygote(), lux_loss_function, (X,Y), gtstate)
    if rem(i,1000) == 0
        println("$i\t$loss")
    end
end

# Trained model
model = gtstate.model
parameters = gtstate.parameters
states = gtstate.states
f(x) = sum(first(model([x], parameters, states)))

# Plotting
using CairoMakie
fig = Figure(size=(420,300), fontsize=11.5, backgroundcolor=:transparent)
axis = Axis(fig[1,1], xlabel=L"$x$", ylabel=L"$y$", xlabelsize=16.5, ylabelsize=16.5)
            lines!(axis, -10..10, x -> f(x), label="model")
            scatter!(axis, X', x -> x^2, color=:black, label="exact")
            axislegend(axis, position=:rb, framevisible=false)
            fig

