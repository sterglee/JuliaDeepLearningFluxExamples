using DecisionTree
using Statistics
using Printf
using Random

function protein_folding_prediction_synthetic()
    println("DEMO: Synthetic CART decision tree for protein folding prediction")
        println("Dataset: 27-class synthetic protein folding data\n")

        # --- 1. Parameters ---
        n_samples_train = 500       # training samples
        n_samples_test  = 200       # test samples
        n_features = 10             # number of features per sample
        n_classes  = 27             # number of folding classes

        # --- 2. Generate Synthetic Training Data ---
        Random.seed!(0)
        train_X = randn(n_features, n_samples_train) + rand(1:n_classes, 1)'  # small class offsets
        train_y = rand(1:n_classes, n_samples_train)

        # --- 3. Generate Synthetic Test Data ---
        test_X = randn(n_features, n_samples_test) + rand(1:n_classes, 1)'
        test_y = rand(1:n_classes, n_samples_test)

        # --- 4. Train the CART Model ---
        # min_samples_split=5, max_depth=-1 (unlimited), min_purity_increase=0.7
        model = build_tree(train_X', train_y;
                           min_samples_split=5,
                           max_depth=-1,
                           min_purity_increase=0.7)

        # --- 5. Evaluate Performance ---
        predictions = apply_tree(model, test_X')
        accuracy = mean(predictions .== test_y)
        @printf("Final classification accuracy (synthetic data): %.2f%%\n", accuracy * 100)

        # --- 6. Visualization / Tree Summary ---
        println("\nTree structure (first 3 levels):")
        print_tree(model, 3)
    end

    # Run the synthetic prediction
    protein_folding_prediction_synthetic()

