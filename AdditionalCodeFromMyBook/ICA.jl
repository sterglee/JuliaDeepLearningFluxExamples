using Downloads, CSV, DataFrames
using MultivariateStats, Statistics, LinearAlgebra
using Plots, DelimitedFiles

# =========================
# 1. Load EEG dataset
# =========================
url = "https://datahub.io/core/eeg-eye-state/r/eeg-eye-state.csv"
file = "eeg_eye_state.csv"

if !isfile(file)
    Downloads.download(url, file)
end

df = CSV.read(file, DataFrame)
X = Matrix(df[:, 1:end-1])'

# =========================
# 2. Standardization
# =========================
μ = mean(X, dims=2)
σ = std(X, dims=2)

X_std = (X .- μ) ./ σ

# =========================
# 3. ICA
# =========================
model = fit(ICA, X_std, size(X,1); maxiter=1000, tol=1e-5)

ICs = MultivariateStats.transform(model, X_std)

# =========================
# 4. Artifact detection
# =========================
kurtosis(v) = mean(((v .- mean(v)) ./ std(v)).^4) - 3

k_vals = [kurtosis(ICs[i,:]) for i in 1:size(ICs,1)]
artifact_idx = sortperm(k_vals, rev=true)[1:2]

println("Artifacts: ", artifact_idx)

# =========================
# 5. CLEAN + CORRECT RECONSTRUCTION
# =========================
ICs_clean = copy(ICs)
ICs_clean[artifact_idx, :] .= 0.0

# 🔥 IMPORTANT FIX
X_rec = model.W \ ICs_clean
X_clean = X_rec .* σ .+ μ

# =========================
# 6. Plot
# =========================
t = 1:size(X,2)

p = plot(t, X[1,:], label="Raw EEG", alpha=0.5)
plot!(p, t, X_clean[1,:], label="Clean EEG", lw=2)

display(p)

# =========================
# 7. Save
# =========================
mkpath("results")
savefig(p, "results/eeg_ica.png")

println("Done.")

