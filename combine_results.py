import pandas as pd
from pathlib import Path


# ============================================================
# GREEN AI MASTER RESULTS
# ============================================================

RESULTS_DIR = Path("results/metrics")


# ============================================================
# 1. LOAD BENCHMARK RESULTS
# ============================================================

benchmark = pd.read_csv(
    RESULTS_DIR / "repeated_benchmark_summary.csv"
)

memory = pd.read_csv(
    RESULTS_DIR / "peak_memory_benchmark_summary.csv"
)

energy = pd.read_csv(
    RESULTS_DIR / "energy_benchmark_summary.csv"
)


# ============================================================
# 2. SELECT REQUIRED COLUMNS
# ============================================================

benchmark = benchmark[
    [
        "Model",
        "Average_Accuracy",
        "Average_Training_Time",
        "Training_Time_Std",
        "Average_Prediction_Time",
        "Prediction_Time_Std"
    ]
]

memory = memory[
    [
        "Model",
        "Average_Peak_Memory",
        "Peak_Memory_Std",
        "Average_Peak_Increase",
        "Peak_Increase_Std"
    ]
]

energy = energy[
    [
        "Model",
        "Average_Energy_Per_Fit_Raw",
        "Energy_Per_Fit_Std"
    ]
]


# ============================================================
# 3. MERGE ALL RESULTS
# ============================================================

master = benchmark.merge(
    memory,
    on="Model",
    how="inner"
)

master = master.merge(
    energy,
    on="Model",
    how="inner"
)


# ============================================================
# 4. RENAME COLUMNS
# ============================================================

master = master.rename(
    columns={
        "Average_Accuracy":
            "Accuracy",

        "Average_Training_Time":
            "Training_Time_s",

        "Average_Prediction_Time":
            "Prediction_Time_s",

        "Average_Peak_Memory":
            "Peak_RSS_MB",

        "Average_Peak_Increase":
            "Peak_RSS_Increase_MB",

        "Average_Energy_Per_Fit_Raw":
            "RAPL_Energy_Delta_Raw_Per_Fit"
    }
)


# ============================================================
# 5. SAVE MASTER CSV
# ============================================================

output_file = (
    RESULTS_DIR / "green_ai_master_results.csv"
)

master.to_csv(
    output_file,
    index=False
)


# ============================================================
# 6. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 80)
print("GREEN AI MASTER RESULTS")
print("=" * 80)

print(master.to_string(index=False))

print("\n" + "=" * 80)
print(f"Saved to: {output_file}")
print("=" * 80)