using Statistics
using Plots
using LinearAlgebra

# --- 1. Data and Calculations ---
true_labels = ["normal", "normal", "cancer", "normal", "cancer", "cancer", "normal", "normal", "cancer", "normal"]
predictions = ["normal", "cancer", "cancer", "normal", "cancer", "normal", "normal", "normal", "cancer", "cancer"]

function get_confusion_stats(true_labels, predictions, pos)
    TP = sum((true_labels .== pos) .& (predictions .== pos))
    FP = sum((true_labels .!= pos) .& (predictions .== pos))
    TN = sum((true_labels .!= pos) .& (predictions .!= pos))
    FN = sum((true_labels .== pos) .& (predictions .!= pos))
    return [TP FP; FN TN]
end

conf_matrix = get_confusion_stats(true_labels, predictions, "cancer")
tp, fp, fn, tn = Int.(conf_matrix)

# Metrics
accuracy  = (tp + tn) / sum(conf_matrix)
precision = tp / (tp + fp)
recall    = tp / (tp + fn)

# --- 2. Visualization ---
x_labs = ["Pred: Cancer", "Pred: Normal"]
y_labs = ["True: Cancer", "True: Normal"]

# Create the heatmap
# We use 'digits=2' in the title string formatting
p = heatmap(x_labs, y_labs, conf_matrix,
            aspect_ratio=:equal,
            c=:blues,
            xlabel="Model Prediction",
            ylabel="Actual Reality",
            yflip=true,
            title="Confusion Matrix\n(Acc: $(round(accuracy, digits=2)), Prec: $(round(precision, digits=2)), Rec: $(round(recall, digits=2)))")

# Add text annotations
# Coordinates are 1-based indices for the heatmap cells
annotate!(p, [
    (1, 1, text("TP: $tp", :white, :bold, 12)),
    (2, 1, text("FP: $fp", :black, :bold, 12)),
    (1, 2, text("FN: $fn", :black, :bold, 12)),
    (2, 2, text("TN: $tn", :black, :bold, 12))
    ])

# --- 3. Save to PNG ---
savefig(p, "confusion_matrix_final.png")
println("Successfully saved plot to confusion_matrix_final.png")
display(p)

