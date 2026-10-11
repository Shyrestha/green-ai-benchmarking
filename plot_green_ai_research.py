
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PHASE 9: RESEARCH-QUALITY GREEN AI VISUALIZATIONS
# ============================================================

METRICS_DIR = Path("results/metrics")
PLOTS_DIR = Path("results/plots")

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

data_file = METRICS_DIR / "green_ai_master_results.csv"
df = pd.read_csv(data_file)

print("\nGREEN AI RESEARCH VISUALIZATIONS")
print("=" * 65)
print(f"Models loaded: {len(df)}")
print(f"Output directory: {PLOTS_DIR}")


# ============================================================
# GRAPH 1: TRAINING AND PREDICTION TIME
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

x = range(len(df))
width = 0.36

ax.bar(
    [i - width / 2 for i in x],
    df["Training_Time_s"],
    width,
    label="Training time"
)

ax.bar(
    [i + width / 2 for i in x],
    df["Prediction_Time_s"],
    width,
    label="Prediction time"
)

ax.set_xticks(list(x))
ax.set_xticklabels(df["Model"])
ax.set_ylabel("Time (seconds)")
ax.set_title("Training and Prediction Time by Model")
ax.set_yscale("log")
ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(
    PLOTS_DIR / "research_timing_comparison.png",
    dpi=300
)
plt.close(fig)


# ============================================================
# GRAPH 2: ACCURACY VERSUS TRAINING TIME
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    df["Training_Time_s"],
    df["Accuracy"] * 100,
    s=100
)

for _, row in df.iterrows():
    ax.annotate(
        row["Model"],
        (
            row["Training_Time_s"],
            row["Accuracy"] * 100
        ),
        xytext=(6, 6),
        textcoords="offset points"
    )

ax.set_xlabel("Average training time (seconds)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Predictive Accuracy versus Training Cost")
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(
    PLOTS_DIR / "research_accuracy_vs_training.png",
    dpi=300
)
plt.close(fig)


# ============================================================
# GRAPH 3: OBSERVED PEAK RSS INCREASE
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

memory_std = df["Peak_Increase_Std"]

ax.bar(
    df["Model"],
    df["Peak_RSS_Increase_MB"],
    yerr=memory_std,
    capsize=5
)

ax.set_ylabel("Observed peak RSS increase (MB)")
ax.set_title("Observed Peak Process Memory Increase")
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(
    PLOTS_DIR / "research_peak_memory.png",
    dpi=300
)
plt.close(fig)


# ============================================================
# GRAPH 4: RAW RAPL PACKAGE-ENERGY COUNTER INCREASE
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

energy_mean = df["RAPL_Energy_Delta_Raw_Per_Fit"]
energy_std = df["Energy_Per_Fit_Std"]

ax.bar(
    df["Model"],
    energy_mean,
    yerr=energy_std,
    capsize=5
)

ax.set_ylabel("Raw counter increase per fit (raw units)")
ax.set_title("Observed RAPL Package-Energy Counter Increase")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(
    PLOTS_DIR / "research_rapl_counter.png",
    dpi=300
)
plt.close(fig)


# ============================================================
# GRAPH 5: ACCURACY COMPARISON
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    df["Model"],
    df["Accuracy"] * 100
)

ax.bar_label(bars, fmt="%.2f%%", padding=3)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Predictive Accuracy Comparison")
ax.set_ylim(90, 100)
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(
    PLOTS_DIR / "research_accuracy.png",
    dpi=300
)
plt.close(fig)


# ============================================================
# FINISH
# ============================================================

print("\nAll research plots created successfully:")

for filename in [
    "research_timing_comparison.png",
    "research_accuracy_vs_training.png",
    "research_peak_memory.png",
    "research_rapl_counter.png",
    "research_accuracy.png"
]:
    print(f" - {PLOTS_DIR / filename}")

print("\nNote:")
print("- RAPL values are raw counter units, not joules.")
print("- RSS measures process memory, not model-only memory.")
print("- Accuracy has no error bars because the current summary")
print("  uses one fixed train/test split.")
