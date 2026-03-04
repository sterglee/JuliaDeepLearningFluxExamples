using WAV
using MultivariateStats
using LinearAlgebra
using Random
using Statistics

# Explicitly import the types and functions to avoid UndefVarErrors
import MultivariateStats: ICA, PCA, fit, predict

Random.seed!(42)

function load_mono(filename)
    if !isfile(filename)
        error("File not found: $filename. Please ensure your WAV files are in the same folder.")
    end
    y, fs = WAV.wavread(filename)
    # Handle stereo vs mono: if 2D, take the first channel (column)
    y_m = size(y, 2) > 1 ? vec(y[:, 1]) : vec(y)
    return y_m, Int(fs)
end

function main()
    println("--- Starting ICA Audio Separation ---")

    # 1. Load WAV files
    # Make sure voice1.wav, voice2.wav, and music.wav exist in your directory
    try
        x1_m, fs1 = load_mono("voice1.wav")
        x2_m, fs2 = load_mono("voice2.wav")
        x3_m, fs3 = load_mono("music.wav")

        fs = fs1 # Assuming same sample rate for all

        # 2. Truncate all signals to the shortest length for matrix operations
        min_len = minimum([length(x1_m), length(x2_m), length(x3_m)])
        # Use views or slicing to align lengths
        S = vcat(x1_m[1:min_len]', x2_m[1:min_len]', x3_m[1:min_len]')

        # 3. Mixing signals
        # We simulate 3 microphones picking up a mix of the 3 sources
        println("Mixing signals into 3 virtual microphones...")
        A = randn(3, 3)
        Y = A * S

        # 4. Run ICA (Independent Component Analysis)
        # ICA tries to find the 'un-mixing' matrix to recover independent sources
        println("Running ICA (FastICA algorithm)...")
        # Note: MultivariateStats uses fit(ICA, data, k)
        ica_model = fit(ICA, Y, 3; maxiter=500, tol=1e-6)
        icasig = predict(ica_model, Y)

        # 5. Run PCA (Principal Component Analysis) for comparison
        # PCA finds directions of max variance, which rarely separates audio cleanly
        println("Running PCA for comparison...")
            pca_model = fit(PCA, Y; maxoutdim=3)
            pcasig = predict(pca_model, Y)

            # 6. Normalize and Save
            # Audio must be between -1.0 and 1.0 to avoid clipping
            normalize_audio(sig) = sig ./ (maximum(abs.(sig)) + 1e-8)

            println("Saving recovered files...")
            for i in 1:3
                # Extract row i and convert to 1D vector
                rec_ica = normalize_audio(vec(icasig[i, :]))
                rec_pca = normalize_audio(vec(pcasig[i, :]))

                WAV.wavwrite(rec_ica, "recovered_ica_$i.wav", Fs=fs)
                WAV.wavwrite(rec_pca, "recovered_pca_$i.wav", Fs=fs)
            end

            println("✔ Success! Check your directory for recovered_*.wav files.")

                catch e
                println("An error occurred: ", e)
            end
        end

        main()

