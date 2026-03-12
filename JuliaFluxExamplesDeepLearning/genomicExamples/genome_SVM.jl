

#Accuracy: 100.0%
 #   Actual: 1.0, Predicted: 1.0
# 0.186387 seconds (4.26 M allocations: 225.880 MiB, 9.45% gc time, 1 lock conflict)


using HTTP, LIBSVM, Statistics, Random, Flux

# ------------------------------------------------------------
# 1. DATA PREPROCESSING
# ------------------------------------------------------------
function load_and_encode()
    println("Fetching dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data"
    response = HTTP.get(url)
    raw_lines = split(strip(String(response.body)), "\n")

    alphabet = ['a', 'g', 'c', 't']
    X, Y = [], []

    for line in raw_lines
        parts = split(line, ",")
        if length(parts) < 3 continue end

        # Label: + is 1 (Promoter), - is 2 (Non-Promoter)
        label = strip(parts[1]) == "+" ? 1.0 : 2.0
        seq = lowercase(replace(strip(parts[3]), r"\s+" => ""))
        if length(seq) != 57 continue end

        # One-hot encode and flatten (4 * 57 = 228 features)
        encoded = Float64.(Flux.onehotbatch(collect(seq), alphabet))
        push!(X, vec(encoded))
        push!(Y, label)
    end

    # Convert to Matrix: (Features x Samples) -> (Samples x Features)
    return reduce(hcat, X)', Vector{Float64}(Y)
end

# ------------------------------------------------------------
# 2. TRAINING THE SVM
# ------------------------------------------------------------
function main()
    Random.seed!(42)
    X, Y = load_and_encode()

    # Split into Train (80%) and Test (20%)
    n = size(X, 1)
    idx = shuffle(1:n)
    train_idx = idx[1:Int(floor(0.8*n))]
    test_idx = idx[Int(floor(0.8*n))+1:end]

    X_train, Y_train = X[train_idx, :], Y[train_idx]
    X_test, Y_test = X[test_idx, :], Y[test_idx]

    println("Training SVM with RBF Kernel...")

    # LIBSVM fit
    # kernel: Radial Basis Function (RBF) is standard for DNA
    model = svmtrain(X_train', Y_train;
                     kernel=LIBSVM.Kernel.RadialBasis,
                     cost=10.0,
                     gamma=0.01)

    # 3. EVALUATION
    Y_pred, _ = svmpredict(model, X_test')

    accuracy = mean(Y_pred .== Y_test) * 100
    println("\n--- Results ---")
    println("Accuracy: $(round(accuracy, digits=2))%")

    # Sample Prediction
    sample_idx = 1
    println("Actual: $(Y_test[sample_idx]), Predicted: $(Y_pred[sample_idx])")
end

@time main()

