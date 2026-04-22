using MultivariateStats
using Statistics

# 1. Δημιουργία τεχνητών δεδομένων (10 μεταβλητές, 100 παρατηρήσεις)
# Σημείωση: Στη Julia, οι παρατηρήσεις είναι συνήθως στήλες (column-major)
d = 10  # αριθμός χαρακτηριστικών (features)
n = 100 # αριθμός δειγμάτων
X = randn(d, n)

# 2. Εκπαίδευση του μοντέλου Factor Analysis
# outdim: Ο αριθμός των λανθανόντων παραγόντων που θέλουμε να εξάγουμε
model = fit(FactorAnalysis, X; maxoutdim=3, method=:cm)

# 3. Ανάκτηση βασικών ιδιοτήτων του μοντέλου
W = loadings(model)      # Ο πίνακας παραγοντικών φορτίσεων (loadings)
μ = mean(model)          # Ο μέσος όρος των δεδομένων εισόδου
ψ = var(model)           # Η ειδική διακύμανση (specific variance) κάθε μεταβλητής

println("Διαστάσεις πίνακα φορτίσεων: ", size(W))

# 4. Μετασχηματισμός (Inference)
# Μετατρέπουμε τα αρχικά δεδομένα d-διαστάσεων στις p-διαστάσεις των παραγόντων
Z = predict(model, X)

# 5. Ανακατασκευή (Reconstruction)
# Προσπάθεια ανακατασκευής των αρχικών δεδομένων από τους παράγοντες
X_hat = reconstruct(model, Z)

println("Σφάλμα ανακατασκευής (μέση τιμή): ", mean(abs.(X - X_hat)))

