using LIBSVM
using Statistics, Random, Printf

# --- 1. CONFIGURATION ---
const SEQ_LEN = 250
const VOCAB_SIZE = 5 

# --- 2. DATA PREPARATION ---
function get_svm_data()
    println("Generating genomic dataset for SVM...")
    X = rand(1:VOCAB_SIZE, SEQ_LEN, 5000)
    Y = rand(1:2, 5000)
    
    train_x, train_y = X[:, 1:4000], Y[1:4000]
    test_x, test_y = X[:, 4001:5000], Y[4001:5000]
    
    # LIBSVM expects (Features x Samples)
    return Float64.(train_x), train_y, Float64.(test_x), test_y
end

# --- 3. MANUAL CONFUSION MATRIX ---
# This replaces the missing MLUtils function
function get_metrics(y_pred, y_true)
    # Class 1 = Promoter (Positive), Class 2 = Non-Promoter (Negative)
    tp = sum((y_pred .== 1) .& (y_true .== 1))
    tn = sum((y_pred .== 2) .& (y_true .== 2))
    fp = sum((y_pred .== 1) .& (y_true .== 2))
    fn = sum((y_pred .== 2) .& (y_true .== 1))
    
    return tp, tn, fp, fn
end

# --- 4. SVM MODEL CONSTRUCTION ---
function run_svm()
    Random.seed!(42)
    xt, yt, xv, yv = get_svm_data()

    println("Training SVM (RBF Kernel)...")
    model = svmtrain(xt, yt; 
        kernel = Kernel.RadialBasis, 
        gamma = 1.0/SEQ_LEN, 
        cost = 1.0
    )

    println("Evaluating SVM...")
    (predictions, _) = svmpredict(model, xv)
    
    acc = mean(predictions .== yv)
    @printf("Final SVM Accuracy: %.2f%%\n", acc * 100)
    
    return predictions, yv
end

# --- 5. BIOLOGICAL COMPARISON ---
function compare_metrics(y_pred, y_true)
    tp, tn, fp, fn = get_metrics(y_pred, y_true)
    
    # Avoid division by zero
    sens = tp / max(1, (tp + fn))
    spec = tn / max(1, (tn + fp))
    
    println("--- Biological Metrics ---")
    @printf("True Positives:  %d\n", tp)
    @printf("True Negatives:  %d\n", tn)
    @printf("Sensitivity:     %.2f%%\n", sens * 100)
    @printf("Specificity:     %.2f%%\n", spec * 100)
end

# Main Execution
preds, targets = run_svm()
compare_metrics(preds, targets)

