
import pandas as pd
from pathlib import Path


# ============================================================
# GREEN AI TRADE-OFF ANALYSIS
# ============================================================

RESULTS_DIR = Path("results/metrics")

input_file = RESULTS_DIR / "green_ai_master_results.csv"
output_file = RESULTS_DIR / "green_ai_tradeoff_analysis.csv"

df = pd.read_csv(input_file)


# ============================================================
# 1. ACCURACY IN PERCENTAGE
# ============================================================

df["Accuracy_Percent"] = df["Accuracy"] * 100


# ============================================================
# 2. RELATIVE TRAINING AND PREDICTION TIME
# ============================================================

fastest_training = df["Training_Time_s"].min()
fastest_prediction = df["Prediction_Time_s"].min()

df["Training_Time_Relative_To_Fastest"] = (
    df["Training_Time_s"] / fastest_training
)

df["Prediction_Time_Relative_To_Fastest"] = (
    df["Prediction_Time_s"] / fastest_prediction
)


# ============================================================
# 3. RELATIVE RAW RAPL COUNTER INCREASE
# ============================================================

lowest_energy_counter = (
    df["RAPL_Energy_Delta_Raw_Per_Fit"].min()
)

df["RAPL_Counter_Relative_To_Lowest"] = (
    df["RAPL_Energy_Delta_Raw_Per_Fit"]
    / lowest_energy_counter
)


# ============================================================
# 4. ACCURACY DIFFERENCE FROM THE BEST OBSERVED ACCURACY
# ============================================================

best_accuracy = df["Accuracy"].max()

df["Accuracy_Gap_Percentage_Points"] = (
    best_accuracy - df["Accuracy"]
) * 100


# ============================================================
# 5. ACCURACY PER SECOND OF TRAINING
# ============================================================

# This is a simple descriptive ratio, not a universal
# scientific efficiency score.

df["Accuracy_Per_Training_Second"] = (
    df["Accuracy_Percent"] / df["Training_Time_s"]
)


# ============================================================
# 6. PRINT KEY FINDINGS
# ============================================================

print("\n" + "=" * 75)
print("GREEN AI TRADE-OFF ANALYSIS")
print("=" * 75)

print("\n1. MODEL METRICS AND RELATIVE COSTS\n")

columns_to_display = [
    "Model",
    "Accuracy_Percent",
    "Training_Time_s",
    "Training_Time_Relative_To_Fastest",
    "Prediction_Time_s",
    "Prediction_Time_Relative_To_Fastest",
    "Peak_RSS_Increase_MB",
    "RAPL_Counter_Relative_To_Lowest",
    "Accuracy_Gap_Percentage_Points",
]

print(
    df[columns_to_display].round(4).to_string(index=False)
)


# ============================================================
# 7. AUTOMATIC PAIRWISE COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("PAIRWISE COMPARISONS")
print("=" * 75)

models = df.set_index("Model")

rf = models.loc["Random Forest"]
svm = models.loc["SVM"]
lr = models.loc["Logistic Regression"]

print(
    "\nRandom Forest / SVM training-time ratio: "
    f"{rf['Training_Time_s'] / svm['Training_Time_s']:.2f}x"
)

print(
    "Random Forest / SVM prediction-time ratio: "
    f"{rf['Prediction_Time_s'] / svm['Prediction_Time_s']:.2f}x"
)

print(
    "Random Forest / SVM raw RAPL counter ratio: "
    f"{rf['RAPL_Energy_Delta_Raw_Per_Fit'] / svm['RAPL_Energy_Delta_Raw_Per_Fit']:.2f}x"
)

print(
    "\nSVM accuracy advantage over Random Forest: "
    f"{(svm['Accuracy'] - rf['Accuracy']) * 100:.2f} percentage points"
)

print(
    "Logistic Regression / SVM training-time ratio: "
    f"{lr['Training_Time_s'] / svm['Training_Time_s']:.2f}x"
)


# ============================================================
# 8. SAVE ANALYSIS
# ============================================================

df.to_csv(output_file, index=False)

print("\n" + "=" * 75)
print("ANALYSIS SAVED")
print("=" * 75)
print(f"File: {output_file}")
