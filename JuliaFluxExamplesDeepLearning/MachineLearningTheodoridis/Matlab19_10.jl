using Images, MultivariateStats, LinearAlgebra, Statistics, Printf

function eigenface_reconstruction(dir_path::String)
    # 1. Load Images
    search_path = joinpath(dir_path, "faces")
    files = filter(f -> occursin(r"\.(jpg|png|pgm|bmp)$"i, f), readdir(search_path))

    if isempty(files)
        error("No images found in $search_path")
    end

    # Load first image to get dimensions
    img1 = load(joinpath(search_path, files[1]))
    h, w = size(img1)
    d = h * w
    n_images = length(files)

    # Build data matrix X (d x n)
    X = zeros(Float64, d, n_images)
    for (i, file) in enumerate(files)
        img = load(joinpath(search_path, file))
        X[:, i] = Float64.(vec(Gray.(img)))
    end

    # 2. Select a random test face and remove it from the training set
    pick_idx = rand(1:n_images)
    test_img_vec = X[:, pick_idx]

    # Training set (exclude the test image)
    X_train = X[:, 1:end .!= pick_idx]

    # 3. Perform PCA (using MultivariateStats for convenience)
    # We fit PCA on the training data
    println("Computing PCA/Eigenfaces...")
    pca_model = fit(PCA, X_train; method=:svd)

    # Extract the Mean and Eigenvectors (basis)
    μ = mean(pca_model)
    V = projection(pca_model) # Each column is an Eigenface

    # 4. Reconstruct the test face
    # Center the test image
    centered_test = test_img_vec - μ

    # Project into "Face Space" (Weights)
    weights = V' * centered_test

    # Reconstruction helper
    function reconstruct(n_comp)
        # Sum of (Weight_i * Eigenface_i) + Mean
        rec = V[:, 1:n_comp] * weights[1:n_comp] + μ
        return reshape(clamp.(rec, 0, 1), h, w)
    end

    # 5. Generate Outputs
    println("Generating reconstructions...")
    orig = reshape(test_img_vec, h, w)
    rec_300 = reconstruct(min(300, size(V, 2)))
    rec_all = reconstruct(size(V, 2))

    # Save or Display (using FileIO/Images)
    save("original_face.png", colorview(Gray, orig))
    save("reconstruction_300.png", colorview(Gray, rec_300))
    save("reconstruction_full.png", colorview(Gray, rec_all))

    println("✔ Done! Images saved to current directory.")
end

