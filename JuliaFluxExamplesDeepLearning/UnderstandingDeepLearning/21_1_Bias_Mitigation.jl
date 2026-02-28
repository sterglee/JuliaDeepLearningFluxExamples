using Flux
using Distributions
using Statistics
using Plots

# --- 1. Data Generation (Synthetic Credit Scores) ---
# Replicating the two populations (Blue and Yellow) from the notebook
function generate_scores(n_samples=1000)
    # Group 0 (Blue): Generally lower credit scores in this synthetic example
    scores_p0 = rand(Beta(2, 5), n_samples) .* 100
    # Group 1 (Yellow): Generally higher credit scores
    scores_p1 = rand(Beta(5, 2), n_samples) .* 100

    # True labels (1 = will repay, 0 = will default)
    # Repayment probability is tied to the score
    repay_p0 = rand.(Bernoulli.(clamp.(scores_p0 ./ 100, 0, 1)))
    repay_p1 = rand.(Bernoulli.(clamp.(scores_p1 ./ 100, 0, 1)))

    return (scores_p0, repay_p0), (scores_p1, repay_p1)
end

# --- 2. Fairness Metrics ---
function compute_metrics(scores, labels, threshold)
    predictions = scores .>= threshold

    tp = sum(predictions .& labels)
    fp = sum(predictions .& .!labels)
    tn = sum(.!predictions .& .!labels)
    fn = sum(.!predictions .& labels)

    tpr = tp / (tp + fn) # True Positive Rate (Opportunity)
    fpr = fp / (fp + tn) # False Positive Rate
    selection_rate = mean(predictions) # Demographic Parity metric

    return (tpr=tpr, fpr=fpr, selection_rate=selection_rate)
end

# --- 3. Post-Processing for Fairness ---
# Finding the "Fair" thresholds for both groups
function find_fair_thresholds(data_p0, data_p1; target_tpr=0.8)
    scores_0, labels_0 = data_p0
    scores_1, labels_1 = data_p1

    # Search for thresholds that satisfy Equal Opportunity (Same TPR for both groups)
    thresholds = 0:1:100

    best_t0, best_t1 = 50, 50
    min_diff = Inf

    for t0 in thresholds, t1 in thresholds
        m0 = compute_metrics(scores_0, labels_0, t0)
        m1 = compute_metrics(scores_1, labels_1, t1)

        # Criterion: Minimize difference in TPR while staying near target performance
        diff = abs(m0.tpr - m1.tpr)
        if diff < min_diff && m0.tpr > 0.5 # Ensure some utility
            min_diff = diff
            best_t0, best_t1 = t0, t1
        end
    end

    return best_t0, best_t1
end

# --- 4. Execution & Visualization ---
(p0_data, p1_data) = generate_scores()

# Initial "Unfair" threshold (same for everyone)
t_global = 50
m0_init = compute_metrics(p0_data..., t_global)
m1_init = compute_metrics(p1_data..., t_global)

println("Initial Metrics (Threshold=$t_global):")
println("Group Blue TPR: $(m0_init.tpr), Group Yellow TPR: $(m1_init.tpr)")

# Mitigate Bias
t0, t1 = find_fair_thresholds(p0_data, p1_data)
m0_fair = compute_metrics(p0_data..., t0)
m1_fair = compute_metrics(p1_data..., t1)

println("\nFair Metrics (Threshold Blue=$t0, Threshold Yellow=$t1):")
println("Group Blue TPR: $(m0_fair.tpr), Group Yellow TPR: $(m1_fair.tpr)")

# --- 5. Plotting the Trade-offs ---


p = plot(title="Bias Mitigation: Equal Opportunity", xlabel="False Positive Rate", ylabel="True Positive Rate")
# Plot ROC-like points for both groups
scatter!(p, [m0_init.fpr], [m0_init.tpr], label="Blue (Initial)", markersize=8)
scatter!(p, [m1_init.fpr], [m1_init.tpr], label="Yellow (Initial)", markersize=8)
scatter!(p, [m0_fair.fpr], [m0_fair.tpr], label="Blue (Mitigated)", shape=:star, markersize=10)
scatter!(p, [m1_fair.fpr], [m1_fair.tpr], label="Yellow (Mitigated)", shape=:star, markersize=10)
display(p)

