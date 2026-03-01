using LinearAlgebra
using Random
using Plots
using DSP

function bat_signal_tfa()
        # --- Synthetic Bat Call Signal ---
        FS = 44100
        t = 0:1/FS:0.05
        x_exact = sin.(2π*500*t .* exp.(3*t))  # exponential chirp like bat call
        N = length(x_exact)

        # --- Measurement Operator (Random Subsampling) ---
        M = round(Int, N/8)  # 8x compression
        rng = MersenneTwister(1234)
        rows = randperm(rng, N)[1:M]

        # Simple measurement operator
        A = x -> x[rows]
        At = y -> begin
                z = zeros(N)
                z[rows] .= y
                return z
        end

        # Measurements
        b = A(x_exact)

        # --- Sparse Transform (Gabor placeholder) ---
        # Using STFT as surrogate
        nfft = 256
        hop = 64
        win = hanning(128)

        function stft_matrix(x)
                S = stft(x, win, hop, nfft)
                return S
        end

        S_orig = stft_matrix(x_exact)

        # --- Placeholder Reconstruction (zero-filling) ---
        x_rec = At(b)  # naive reconstruction

        # --- Visualization ---
        t_vec = t
        plot(t_vec, x_exact, label="Original Bat Call")
        plot!(t_vec, x_rec, label="Recovered (Zero-fill)")
        xlabel!("Time (s)"); ylabel!("Amplitude")
        title!("Bat Echolocation Signal - Time Domain")

        # Plot STFT magnitude
        S_rec = abs.(S_orig)
        p2 = heatmap(abs.(S_rec), xlabel="Time Frames", ylabel="Frequency Bins",
                     title="STFT Magnitude (Synthetic Bat Call)", color=:hot)
        display(p2)
end

bat_signal_tfa()

