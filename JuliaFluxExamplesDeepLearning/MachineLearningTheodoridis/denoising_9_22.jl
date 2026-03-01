using LinearAlgebra, Statistics, Random, ProgressMeter, Plots

# --- 1. The Core OMP Solver ---
function omp_solver(Φ, y, s)
    n, m = size(Φ)
    residual = copy(y)
    selected = Int[]
    θ = zeros(m)
    for i in 1:s
        # Correlation step
        proj = Φ' * residual
        _, pos = findmax(abs.(proj))
        push!(selected, pos)
        # Orthogonal projection (Least Squares)
        Φ_s = Φ[:, selected]
        θ_s = Φ_s \ y
        residual .= y .- Φ_s * θ_s
        θ[selected] .= θ_s
        if norm(residual) < 1e-5 break end
    end
    return θ
end

# --- 2. The Main Function ---
function call_denoising_example()
    # Parameters from Exercise 9.22
    N = 128           # Image size (scaled down for speed)
    blocksize = 12    # patch size
    sigma = 20.0      # noise level
    sparsity_s = 5    # number of atoms to use

    # 1. Create Synthetic "Boats" Image (simple geometric shapes)
    Random.seed!(42)
    x_exact = zeros(N, N)
    x_exact[30:90, 30:90] .= 180.0 # A square "hull"
    x_exact[10:40, 50:60] .= 240.0 # A "mast"

    # 2. Add Noise
    x_noise = x_exact .+ randn(N, N) .* sigma

    # 3. Build the 2D DCT Dictionary
    K_atoms = 14^2
    n_range = 0:blocksize-1
    freqs = 0:sqrt(K_atoms)-1
    DCT_1d = [cos(i * j * π / sqrt(K_atoms)) for i in n_range, j in freqs]
        # Normalize atoms
        for j in 1:size(DCT_1d, 2) DCT_1d[:, j] ./= norm(DCT_1d[:, j]) end
        Dict_fixed = kron(DCT_1d, DCT_1d)

        # 4. Sliding Window Denoising
        yout = zeros(N, N)
        weight = zeros(N, N)

        println("Processing patches...")
        @showprogress for j in 1:(N - blocksize + 1), i in 1:(N - blocksize + 1)
            patch = vec(x_noise[i:i+blocksize-1, j:j+blocksize-1])
            θ = omp_solver(Dict_fixed, patch, sparsity_s)

            reconst = reshape(Dict_fixed * θ, blocksize, blocksize)
            yout[i:i+blocksize-1, j:j+blocksize-1] .+= reconst
            weight[i:i+blocksize-1, j:j+blocksize-1] .+= 1.0
        end

        x_recovered = yout ./ weight

        # 5. Calculate PSNR Improvement
        psnr(orig, proc) = 20log10(255 / sqrt(mean((orig .- proc).^2)))
        println("Noisy PSNR: $(round(psnr(x_exact, x_noise), digits=2)) dB")
        println("Recovered PSNR: $(round(psnr(x_exact, x_recovered), digits=2)) dB")

        # Plot results
        p1 = heatmap(x_exact, title="Exact", c=:grays, yflip=true)
        p2 = heatmap(x_noise, title="Noisy", c=:grays, yflip=true)
        p3 = heatmap(x_recovered, title="Denoised", c=:grays, yflip=true)
        display(plot(p1, p2, p3, layout=(1,3), size=(900, 300)))
    end

    # EXECUTE
    call_denoising_example()


