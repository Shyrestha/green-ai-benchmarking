
import time
import os
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset loaded successfully!")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])


# ==========================================
# 2. SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 3. CREATE MODEL FUNCTION
# ==========================================

def create_models():

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                random_state=42
            )
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

    return models


# ==========================================
# 4. EXPERIMENT SETTINGS
# ==========================================

number_of_runs = 5

all_results = []


# ==========================================
# 5. REPEAT THE EXPERIMENT
# ==========================================

for run in range(1, number_of_runs + 1):

    print("\n====================================")
    print("RUN:", run)
    print("====================================")

    models = create_models()

    for model_name, model in models.items():

        print("\nModel:", model_name)

        # ------------------------------
        # Measure training time
        # ------------------------------

        start_training = time.perf_counter()

        model.fit(X_train, y_train)

        end_training = time.perf_counter()

        training_time = end_training - start_training

        # ------------------------------
        # Measure prediction time
        # ------------------------------

        start_prediction = time.perf_counter()

        predictions = model.predict(X_test)

        end_prediction = time.perf_counter()

        prediction_time = end_prediction - start_prediction

        # ------------------------------
        # Calculate accuracy
        # ------------------------------

        accuracy = accuracy_score(y_test, predictions)

        print("Accuracy:", round(accuracy, 4))
        print(
            "Training time:",
            round(training_time, 6),
            "seconds"
        )
        print(
            "Prediction time:",
            round(prediction_time, 6),
            "seconds"
        )

        # ------------------------------
        # Store results
        # ------------------------------

        all_results.append({
            "Run": run,
            "Model": model_name,
            "Accuracy": accuracy,
            "Training Time (seconds)": training_time,
            "Prediction Time (seconds)": prediction_time
        })


# ==========================================
# 6. CONVERT RESULTS TO DATAFRAME
# ==========================================

results_df = pd.DataFrame(all_results)


# ==========================================
# 7. CALCULATE SUMMARY STATISTICS
# ==========================================

summary_df = (
    results_df
    .groupby("Model")
    .agg(
        Average_Accuracy=("Accuracy", "mean"),
        Average_Training_Time=("Training Time (seconds)", "mean"),
        Training_Time_Std=("Training Time (seconds)", "std"),
        Average_Prediction_Time=("Prediction Time (seconds)", "mean"),
        Prediction_Time_Std=("Prediction Time (seconds)", "std")
    )
    .reset_index()
)


# ==========================================
# 8. CREATE OUTPUT FOLDER
# ==========================================

os.makedirs("results/metrics", exist_ok=True)


# ==========================================
# 9. SAVE DETAILED RESULTS
# ==========================================

detailed_path = (
    "results/metrics/repeated_benchmark_results.csv"
)

results_df.to_csv(detailed_path, index=False)


# ==========================================
# 10. SAVE SUMMARY RESULTS
# ==========================================

summary_path = (
    "results/metrics/repeated_benchmark_summary.csv"
)

summary_df.to_csv(summary_path, index=False)


# ==========================================
# 11. DISPLAY SUMMARY
# ==========================================

print("\n====================================")
print("REPEATED BENCHMARK SUMMARY")
print("====================================")

print(summary_df.to_string(index=False))

print("\nDetailed results saved to:", detailed_path)
print("Summary results saved to:", summary_path)