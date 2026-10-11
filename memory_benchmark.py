
import os
import gc
import psutil
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# 1. Identify the current Python process
process = psutil.Process(os.getpid())


# 2. Load the dataset
data = load_breast_cancer()

X = data.data
y = data.target


# 3. Split the dataset consistently
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Function to create fresh models
def create_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, random_state=42)
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "SVM": make_pipeline(
            StandardScaler(),
            SVC()
        )
    }


# 5. Experiment settings
number_of_runs = 5
results = []


# 6. Run the experiment repeatedly
for run in range(1, number_of_runs + 1):

    print(f"\nRun {run}")
    models = create_models()

    for model_name, model in models.items():

        # Encourage cleanup of unreachable Python objects
        gc.collect()

        # Measure process RSS before training
        memory_before = (
            process.memory_info().rss / (1024 * 1024)
        )

        # Train the model
        model.fit(X_train, y_train)

        # Measure process RSS after training
        memory_after = (
            process.memory_info().rss / (1024 * 1024)
        )

        # Calculate observed RSS difference
        memory_change = memory_after - memory_before

        # Store the measurements
        results.append({
            "Run": run,
            "Model": model_name,
            "Memory Before (MB)": memory_before,
            "Memory After (MB)": memory_after,
            "RSS Change (MB)": memory_change
        })

        print(
            f"{model_name}: "
            f"Before={memory_before:.2f} MB, "
            f"After={memory_after:.2f} MB, "
            f"Change={memory_change:+.2f} MB"
        )

        # Release the model reference before the next model
        del model


# 7. Convert results to a DataFrame
results_df = pd.DataFrame(results)


# 8. Calculate summary statistics
summary_df = results_df.groupby("Model").agg(
    Average_RSS_Change=("RSS Change (MB)", "mean"),
    RSS_Change_Std=("RSS Change (MB)", "std"),
    Average_Memory_Before=("Memory Before (MB)", "mean"),
    Average_Memory_After=("Memory After (MB)", "mean")
).reset_index()


# 9. Create output directory
os.makedirs("results/metrics", exist_ok=True)


# 10. Save detailed results and summary
results_df.to_csv(
    "results/metrics/memory_benchmark_results.csv",
    index=False
)

summary_df.to_csv(
    "results/metrics/memory_benchmark_summary.csv",
    index=False
)


# 11. Display the results
print("\n" + "=" * 65)
print("DETAILED MEMORY BENCHMARK RESULTS")
print("=" * 65)
print(results_df.round(3).to_string(index=False))

print("\n" + "=" * 65)
print("MEMORY BENCHMARK SUMMARY")
print("=" * 65)
print(summary_df.round(3).to_string(index=False))

print("\nFiles saved successfully!")
print("results/metrics/memory_benchmark_results.csv")
print("results/metrics/memory_benchmark_summary.csv")
