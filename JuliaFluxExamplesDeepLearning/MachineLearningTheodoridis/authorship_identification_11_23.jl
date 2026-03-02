using LibSVM
using Random
using Statistics
using LinearAlgebra
using TextAnalysis
using SparseArrays

# -----------------------------
# 1. Set paths
# -----------------------------
baseDir = "/full/path/to/"
train_dir = joinpath(baseDir, "data/train/")

# -----------------------------
# 2. Bag-of-Words Extraction
# -----------------------------
docs = String[]
labels = Int32[]

# Load documents from 4 authors
for (author_id, folder) in enumerate(["author1", "author2", "author3", "author4"])
    folder_path = joinpath(train_dir, folder)
    files = readdir(folder_path)
    for file in files
        push!(docs, read(joinpath(folder_path, file), String))
        push!(labels, Int32(author_id))
    end
end

# Create Corpus and vectorize
corpus = Corpus(docs)
vectorizer = CountVectorizer()
X_sparse = fit_transform!(vectorizer, corpus)       # returns SparseMatrixCSC
X = Matrix{Float64}(X_sparse)                        # convert to dense matrix

# -----------------------------
# 3. SVM Cross-Validation
# -----------------------------
function libsvm_cv(labels::Vector{Int32}, data::Matrix{Float64};
                   kfolds::Int=5, gamma::Float64=0.5, C::Float64=1.0)
    N = length(labels)
    indices = collect(1:N)
    shuffle!(indices)
    fold_size = ceil(Int, N / kfolds)
    accuracies = Float64[]

    for fold in 1:kfolds
        test_idx = indices[((fold-1)*fold_size+1):min(fold*fold_size, N)]
        train_idx = setdiff(indices, test_idx)
        model = svmtrain(data[train_idx, :], labels[train_idx];
                         kernel=RBFKernel(), gamma=gamma, C=C)
        pred = svmpredict(model, data[test_idx, :])
        acc = mean(pred .== labels[test_idx])
        push!(accuracies, acc)
    end
    return mean(accuracies)
end

# -----------------------------
# 4. Run cross-validation
# -----------------------------
Random.seed!(123)
acc = libsvm_cv(labels, X)
println("Cross-validated accuracy (Bag-of-Words SVM): ", round(acc*100, digits=2), "%")


