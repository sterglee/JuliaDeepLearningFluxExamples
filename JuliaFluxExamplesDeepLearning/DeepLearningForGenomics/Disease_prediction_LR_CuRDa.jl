using CSV
using DataFrames
using StatsPlots
using MLJ
using MLJLinearModels
using ZipFile
using Plots

# --- 1. Data Collection & Preprocessing ---

function load_zipped_csv(path)
    if !isfile(path)
        @warn "File not found: $path"
        return DataFrame()
    end

    z = ZipFile.Reader(path)
    f = z.files[1]

    # Avoid special string types
    df = DataFrame(CSV.File(read(f), stringtype=String))

    close(z)
    return df
end

paths = [
    "lung/GSE87340.csv.zip",
    "lung/GSE60052.csv.zip",
    "lung/GSE40419.csv.zip",
    "lung/GSE37764.csv.zip"
]

dfs = [load_zipped_csv(p) for p in paths]
filter!(d -> !isempty(d), dfs)

lung_1_4 = reduce(vcat, dfs)

# Clean labels
lung_1_4.class = strip.(string.(lung_1_4.class))

# --- 2. Model Preparation ---

# Target variable
y = coerce(lung_1_4.class, Multiclass)

# Feature matrix
X = DataFrames.select(lung_1_4, Not([:class, :ID]))

# Partition indices (robust MLJ method)
train_inds, test_inds = partition(eachindex(y), 0.75, rng=42, stratify=y)

X_train = X[train_inds, :]
X_test  = X[test_inds, :]

y_train = y[train_inds]
y_test  = y[test_inds]

# --- 3. Model Training Pipeline ---

LogReg = @load LogisticClassifier pkg=MLJLinearModels verbosity=0

model = LogReg()

pipe = Standardizer() |> model

mach = machine(pipe, X_train, y_train)

fit!(mach)

# --- 4. Evaluation ---

y_pred = predict_mode(mach, X_test)

acc = accuracy(y_pred, y_test)

println("Final Test Accuracy: $(round(acc, digits=4))")

# --- 5. Confusion Matrix ---

cm = confusion_matrix(y_pred, y_test)

labels = levels(y)

heatmap(
    labels,
    labels,
    cm.mat,
    title="Confusion Matrix (Acc: $(round(acc, digits=2)))",
    xlabel="Predicted",
    ylabel="Actual",
    aspect_ratio=:equal,
    color=:viridis
)

