import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# LOAD MASTER RESULTS
# ============================================================

RESULTS_DIR = Path("results/metrics")
PLOTS_DIR = Path("results/plots")

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

file_path = RESULTS_DIR / "green_ai_master_results.csv"

df = pd.read_csv(file_path)

print("\n" + "=" * 70)
print("GREEN AI VISUALIZATION")
print("=" * 70)

print(df.to_string(index=False))


# ============================================================
# GRAPH 1 — ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    df["Model"],
    df["Accuracy"] * 100
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.title("Model Accuracy Comparison")
plt.ylim(90, 100)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "green_ai_accuracy.png",
    dpi=300
)

plt.show()

plt.close()


# ============================================================
# GRAPH 2 — TRAINING TIME
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    df["Model"],
    df["Training_Time_s"]
)

plt.ylabel("Training Time (seconds)")
plt.xlabel("Model")
plt.title("Training Time Comparison")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "green_ai_training_time.png",
    dpi=300
)

plt.show()

plt.close()


# ============================================================
# GRAPH 3 — PEAK MEMORY INCREASE
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    df["Model"],
    df["Peak_RSS_Increase_MB"]
)

plt.ylabel("Peak RSS Increase (MB)")
plt.xlabel("Model")
plt.title("Observed Peak Memory Increase")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "green_ai_peak_memory.png",
    dpi=300
)

plt.show()

plt.close()


# ============================================================
# GRAPH 4 — RAPL ENERGY COUNTER
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    df["Model"],
    df["RAPL_Energy_Delta_Raw_Per_Fit"]
)

plt.ylabel("Raw RAPL Counter Increase / Fit")
plt.xlabel("Model")
plt.title("Observed RAPL Package-Energy Counter Increase")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "green_ai_energy_raw.png",
    dpi=300
)

plt.show()

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("ALL GREEN AI PLOTS CREATED")
print("=" * 70)

print(f"Saved in: {PLOTS_DIR}")