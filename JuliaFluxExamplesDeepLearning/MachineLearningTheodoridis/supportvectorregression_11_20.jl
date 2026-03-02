using WAV, LinearAlgebra, Random, Statistics, Plots

# --- 1. Infrastructure ---
abstract type Kernel end
struct Gaussian <: Kernel sigma::Float64 end
function kappa(x, y, k::Gaussian)
    return exp(-norm(x - y)^2 / k.sigma^2)
end

# --- 2. Data Preparation ---
Random.seed!(0)

N = 100
samples = 1000
start_sample = 100000
indices = round.(Int, range(1, stop=samples, length=N))

# Initialize variables to avoid UndefVarError
x = zeros(N)
y = zeros(N)
t_full = zeros(samples + 1)
Ts = 0.0

# Load Audio or Generate Synthetic
try
    # subrange is a Tuple (start, end)
    y_raw, fs = wavread("wavs/BladeRunner.wav", subrange=(start_sample, start_sample + samples))
    global Ts = 1/fs
    global t_full = (0:samples) .* Ts
    global x = t_full[indices]
    global y = y_raw[indices, 1]
    catch e
    @warn "WAV file not found or error occurred: $e. Using synthetic data."
    fs = 44100
    global Ts = 1/fs
    global t_full = (0:samples) .* Ts
    global x = t_full[indices]
    # Generate a noisy sine wave as a fallback
    global y = sin.(2π * 440 * x)
end

# Add white Gaussian noise (SNR 15dB)
sig_pwr = mean(y.^2)
noise_pwr = sig_pwr / (10^(15/10))
y .+= sqrt(noise_pwr) .* randn(N)

# Add outliers (10% of points)
O = 0.8 * maximum(abs.(y))
M_out = floor(Int, 0.1 * N)
out_ind = randperm(N)[1:M_out]
y[out_ind] .+= sign.(randn(M_out)) .* O

# --- 3. Run SVR Training ---
C = 1.0
epsilon = 0.003
kernel = Gaussian(0.004)

println("Starting SMO...")
# Note: Use the 'smo_regression' function defined in previous steps
# Reshaping x to (N, 1) to match the Matrix input requirement
a1, a2, b, rmse = smo_regression(reshape(x, :, 1), y, C, epsilon, kernel)

# --- 4. Generate Regressor ---
# We calculate the prediction for every point in the original high-res time array
z = [(sum((a1 .- a2) .* [kappa(xi, ti, kernel) for xi in x]) + b) for ti in t_full]

    # Identify Support Vectors
    SV = findall(i -> abs(a1[i]) > 0.001 || abs(a2[i]) > 0.001, 1:N)

    # --- 5. Plot Output ---
    # Equivalent to figure(1) and hold on
    p = plot(t_full, z, color=:red, lw=1.5, label="SVR Regressor")
    plot!(t_full, z .+ epsilon, color=:black, ls=:dash, alpha=0.3, label="ε-tube")
    plot!(t_full, z .- epsilon, color=:black, ls=:dash, alpha=0.3, label="")

    scatter!(x, y, markersize=3, markercolor=:gray, alpha=0.5, label="Noisy Data")
    scatter!(x[SV], y[SV], markershape=:circle, mc=:white, ma=0.0,
             markerstrokecolor=:black, label="Support Vectors")

    plot!(title="SVR Output (C=$C, SVs=$(length(SV)))", xlabel="Time (sec)", ylabel="Amplitude")
    display(p)

