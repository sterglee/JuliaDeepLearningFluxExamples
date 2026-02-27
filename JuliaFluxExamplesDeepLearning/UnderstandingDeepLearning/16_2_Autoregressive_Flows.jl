using Flux, LinearAlgebra, Random, Plots

# ==========================================
# 1. Piecewise Linear Mapping (Invertible)
# ==========================================

"""
g(h, phi)
Implements the piecewise linear mapping from Equation 16.12.
h: input value (scalar)
phi: vector of 5 parameters that sum to 1
"""
function g(h, phi)
    # The mapping divides the [0,1] interval into K segments (here K=5)
    # The output is the sum of full segments plus the fractional part of the current segment
    K = length(phi)
    # Determine which segment h falls into
    segment = Int(floor(h * K)) + 1
    segment = min(segment, K) # Handle boundary h=1.0

    # Sum of previous segments
    h_prime = sum(phi[1:segment-1])
    # Add fractional part of current segment
    h_prime += phi[segment] * (h * K - (segment - 1))

    return h_prime
end

"""
g_inverse(h_prime, phi)
Inverts the piecewise mapping using a bisection/bracketing method.
"""
function g_inverse(h_prime, phi)
    h_low, h_high = 0.0f0, 1.0f0
    thresh = 0.0001f0

    for _ in 1:20
        h_mid = (h_low + h_high) / 2
        if g(h_mid, phi) < h_prime
            h_low = h_mid
        else
            h_high = h_mid
        end
        if (h_high - h_low) < thresh break end
    end
    return (h_low + h_high) / 2
end

# ==========================================
# 2. Autoregressive Parameter Networks
# ==========================================
# These networks output the 'phi' parameters based on previous inputs

function get_phi_fixed()
    return Float32[0.2, 0.1, 0.4, 0.05, 0.25]
end

# Helper to create a small MLP that outputs a valid 'phi' (summing to 1 via softmax)
function make_phi_net(n_in)
    Random.seed!(n_in)
    return Chain(
        Dense(n_in, 10, relu),
        Dense(10, 5),
        softmax
        )
end

net_h1 = make_phi_net(1)
net_h1h2 = make_phi_net(2)
net_h1h2h3 = make_phi_net(3)

# ==========================================
# 3. Forward and Backward Mappings
# ==========================================

"""
Forward mapping: Transforms [h1, h2, h3, h4] -> [h1', h2', h3', h4']
This can be done in parallel for all dimensions.
    """
    function forward_mapping(h1, h2, h3, h4)
        # h1' depends only on fixed phi
        hp1 = g(h1, get_phi_fixed())
        # h2' depends on h1
        hp2 = g(h2, net_h1([h1]))
        # h3' depends on h1, h2
        hp3 = g(h3, net_h1h2([h1, h2]))
        # h4' depends on h1, h2, h3
        hp4 = g(h4, net_h1h2h3([h1, h2, h3]))

        return hp1, hp2, hp3, hp4
    end

    """
    Backward mapping: Reconstructs [h1, h2, h3, h4] from [h1', h2', h3', h4']
    This MUST be done sequentially.
    """
    function backward_mapping(hp1, hp2, hp3, hp4)
        # 1. Recover h1 first
        h1 = g_inverse(hp1, get_phi_fixed())
        # 2. Recover h2 using the newly found h1
        h2 = g_inverse(hp2, net_h1([h1]))
        # 3. Recover h3 using h1 and h2
        h3 = g_inverse(hp3, net_h1h2([h1, h2]))
        # 4. Recover h4 using h1, h2, and h3
        h4 = g_inverse(hp4, net_h1h2h3([h1, h2, h3]))

        return h1, h2, h3, h4
    end

    # ==========================================
    # 4. Verification
    # ==========================================
    h_orig = (0.22f0, 0.41f0, 0.83f0, 0.53f0)
    println("Original h: ", h_orig)

    h_primes = forward_mapping(h_orig...)
    println("Forward mapped (h'): ", h_primes)

    h_recon = backward_mapping(h_primes...)
    println("Reconstructed h: ", h_recon)

