using Plots
using DelimitedFiles

# 1. Load the data from the text file
# The file contains: [Epoch] [Loss] [Accuracy]
data = readdlm("resultsProteinPredictionGRU.txt")

# 2. Extract columns
epochs = data[:, 1]
loss   = data[:, 2]
accuracy = data[:, 3]

# 3. Create a dual-axis plot
p1 = plot(epochs, loss,
          label="Loss",
          linecolor=:red,
          linewidth=2,
          ylabel="Loss",
          title="GRU Training Metrics",
          grid=true)

p2 = plot(epochs, accuracy,
          label="Accuracy %",
          linecolor=:blue,
          linewidth=2,
          ylabel="Accuracy (%)",
          xlabel="Epoch")

# Combine the plots into a layout (Subplots)
plot(p1, p2, layout=(2,1), size=(800, 600))

# Save the plot
savefig("gru_genomic_training_plot.png")

