
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


# 1. Load the dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset loaded successfully!")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])


# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 3. Define the models
models = {
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


# 4. Create an empty list to store results
results = []


# 5. Train and evaluate each model
for model_name, model in models.items():

    print("\n------------------------------")
    print("Model:", model_name)
    print("------------------------------")

    # Measure training time
    start_training = time.perf_counter()

    model.fit(X_train, y_train)

    end_training = time.perf_counter()

    training_time = end_training - start_training

    # Measure prediction time
    start_prediction = time.perf_counter()

    predictions = model.predict(X_test)

    end_prediction = time.perf_counter()

    prediction_time = end_prediction - start_prediction

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", round(accuracy, 4))
    print("Training time:", round(training_time, 6), "seconds")
    print("Prediction time:", round(prediction_time, 6), "seconds")

    # Store the model's results
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Training Time (seconds)": training_time,
        "Prediction Time (seconds)": prediction_time
    })


# 6. Convert results into a Pandas DataFrame
results_df = pd.DataFrame(results)


# 7. Create the output folder if it does not exist
os.makedirs("results/metrics", exist_ok=True)


# 8. Save results to a CSV file
output_path = "results/metrics/baseline_results.csv"

results_df.to_csv(output_path, index=False)


# 9. Display the final table
print("\n==============================")
print("FINAL BENCHMARK RESULTS")
print("==============================")

print(results_df.to_string(index=False))

print("\nResults saved to:", output_path)