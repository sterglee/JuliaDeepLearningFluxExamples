using Flux, Random

# 1. Define the Score Network (The model that predicts the mean)
# In 1D, we use a simple MLP to predict mu_t from z_t and t
#
model = Chain(
    Dense(2, 64, relu), # Inputs: z_t and time t
    Dense(64, 64, relu),
    Dense(64, 1)
    )

# 2. Reverse Sampling Algorithm (Equation 18.15)
# z_{t-1} ~ N(mu(z_t, t), sigma_t^2)
#
function sample_reverse(model, T, beta)
    z = randn(Float32, 1) # Start with pure noise
    path = [z[1]]

    for t in T:-1:1
        # Model predicts the mean based on current z and time
        μ = model(vcat(z, Float32[t/T]))[1]
        σ = sqrt(beta[t]) # Fixed variance schedule

        z = μ .+ σ .* randn(Float32, 1)
        push!(path, z[1])
    end
    return path
end


