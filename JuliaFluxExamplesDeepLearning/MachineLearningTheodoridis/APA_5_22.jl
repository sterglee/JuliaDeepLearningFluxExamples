using LinearAlgebra
using Statistics
using Plots

function run_apa_simulation()
    # Parameters
    L = 60          # Dimension of unknown vector
    N = 3500        # Number of Data points
    IterNo = 100    # Number of Monte Carlo iterations
    noisevar = 0.01

    # Pre-allocate MSE matrices (N x IterNo)
    MSE1 = zeros(N, IterNo) # APA q=30
    MSE2 = zeros(N, IterNo) # APA q=10
    MSE3 = zeros(N, IterNo) # LMS
    MSE4 = zeros(N, IterNo) # NLMS

    theta = randn(L) # The "True" system parameters

    for It in 1:IterNo
        # Generate correlated input using a convolution matrix approach
        xc = randn(N + L - 1)
        xc /= std(xc)

        # Constructing the input matrix X (L x N)
        # In Julia, we can use a Toeplitz-like approach or manual slicing
        X = zeros(L, N)
        for i in 1:N
            X[:, i] = xc[i .+ (L-1:-1:0)]
        end

        # Generate target signal with noise
        noise = randn(N) * sqrt(noisevar)
        y = (X' * theta) + noise

        # --- 1. APA (q=30) ---
        w = zeros(L); mu = 0.1; delta = 0.001; q = 30
        for i in q:N
            indices = i:-1:i-q+1
            yvec = y[indices]
            Xq = X[:, indices]' # (q x L)

            e = yvec - Xq * w
            e_instant = y[i] - dot(w, X[:, i])

            # APA Update: w = w + mu * Xq' * inv(delta*I + Xq*Xq') * e
            # Using \ (backslash) is numerically more stable than inv()
            w += mu * Xq' * ((delta * I + Xq * Xq') \ e)
            MSE1[i, It] = e_instant^2
        end

        # --- 2. APA (q=10) ---
        w = zeros(L); q = 10
        for i in q:N
            indices = i:-1:i-q+1
            yvec = y[indices]
            Xq = X[:, indices]'

            e = yvec - Xq * w
            e_instant = y[i] - dot(w, X[:, i])
            w += mu * Xq' * ((delta * I + Xq * Xq') \ e)
            MSE2[i, It] = e_instant^2
        end

        # --- 3. NLMS ---
        w = zeros(L); mu_nlms = 0.35; delta_nlms = 0.001
        for i in 1:N
            xi = X[:, i]
            e = y[i] - dot(w, xi)
            mun = mu_nlms / (delta_nlms + dot(xi, xi))
            w += mun * e * xi
            MSE4[i, It] = e^2
        end

        # --- 4. LMS ---
        w = zeros(L); mu_lms = 0.025
        for i in 1:N
            xi = X[:, i]
            e = y[i] - dot(w, xi)
            w += mu_lms * e * xi
            MSE3[i, It] = e^2
        end
    end

    # Average over iterations
    MSEav1 = mean(MSE1, dims=2)
    MSEav2 = mean(MSE2, dims=2)
    MSEav3 = mean(MSE3, dims=2)
    MSEav4 = mean(MSE4, dims=2)

    # Plotting
    p = plot(10log10.(MSEav1), label="APA (q=30)", color=:red)
    plot!(10log10.(MSEav2), label="APA (q=10)", color=:green)
    plot!(10log10.(MSEav3), label="LMS", color=:blue)
    plot!(10log10.(MSEav4), label="NLMS", color=:black)
    xlabel!("Iterations")
    ylabel!("MSE (dB)")
    display(p)
end

run_apa_simulation()
