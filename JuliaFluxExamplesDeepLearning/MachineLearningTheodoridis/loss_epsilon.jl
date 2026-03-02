# Use the same Loss hierarchy defined previously
abstract type Loss end
struct L2Loss <: Loss end
struct L1Loss <: Loss end
struct HuberLoss <: Loss
    sigma::Float64
end

# --- Dispatch Methods for ε-Insensitive Loss ---

# L2 ε-insensitive Loss
function loss_epsilon(ksi, epsilon, ::L2Loss)
    return max(0.0, ksi^2 - epsilon)
end

# L1 ε-insensitive Loss
function loss_epsilon(ksi, epsilon, ::L1Loss)
    return max(0.0, abs(ksi) - epsilon)
end

# Huber ε-insensitive Loss
function loss_epsilon(ksi, epsilon, loss::HuberLoss)
    abs_ksi = abs(ksi)
    if abs_ksi <= loss.sigma
        val = ksi^2 / (2 * loss.sigma) - epsilon
    else
        val = abs_ksi - loss.sigma / 2 - epsilon
    end
    return max(0.0, val)
end

    ksi = 1.5      # The error or residual
    epsilon = 0.1  # The insensitive threshold

    # 2. Choose and initialize the loss type
    l2_type = L2Loss()
    l1_type = L1Loss()
    huber_type = HuberLoss(1.0) # sigma = 1.0

    # 3. Call the function
    val_l2    = loss_epsilon(ksi, epsilon, l2_type)
    val_huber = loss_epsilon(ksi, epsilon, huber_type)

    println("L2 ε-loss: ", val_l2)
    println("Huber ε-loss: ", val_huber)


    # A vector of 1,000 random residuals
    residuals = randn(1000)
    epsilon = 0.05
    huber = HuberLoss(0.8)

    # This calculates the loss for ALL 1,000 values instantly
    all_losses = loss_epsilon.(residuals, epsilon, Ref(huber))

    # Show the first 5 results
    println(all_losses[1:5])


