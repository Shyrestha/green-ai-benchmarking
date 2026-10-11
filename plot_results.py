
import os
import pandas as pd
import matplotlib.pyplot as plt


# 1. Load the saved benchmark results
input_path = "results/metrics/baseline_results.csv"

results_df = pd.read_csv(input_path)

print("Results loaded successfully!")
print(results_df)


# 2. Create the output folder
os.makedirs("results/plots", exist_ok=True)


# 3. Create the accuracy comparison graph
plt.figure(figsize=(8, 5))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"]
)

plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy")

plt.ylim(0, 1.05)
plt.xticks(rotation=15)
plt.tight_layout()

accuracy_path = "results/plots/accuracy_comparison.png"

plt.savefig(accuracy_path, dpi=300)
plt.show()

print("Accuracy graph saved to:", accuracy_path)


# 4. Create the training time comparison graph
plt.figure(figsize=(8, 5))

plt.bar(
    results_df["Model"],
    results_df["Training Time (seconds)"]
)

plt.title("Model Training Time Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Training Time (seconds)")

plt.xticks(rotation=15)
plt.tight_layout()

training_path = "results/plots/training_time_comparison.png"

plt.savefig(training_path, dpi=300)
plt.show()

print("Training time graph saved to:", training_path)