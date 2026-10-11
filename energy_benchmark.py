import subprocess
import time
import multiprocessing
import gc

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# ============================================================
# CONFIGURATION
# ============================================================

ENERGY_COUNTER = r"\Energy Meter(RAPL_Package0_PKG)\Energy"

N_TRIALS = 5
N_REPETITIONS = 50

RANDOM_STATE = 42


# ============================================================
# RAPL ENERGY READER
# ============================================================

def get_energy_counter():
    """
    Read the Windows RAPL package energy counter.

    IMPORTANT:
    We store the raw CookedValue returned by Windows.
    We do not assume its physical unit here.
    """

    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            f"(Get-Counter '{ENERGY_COUNTER}')"
            ".CounterSamples[0].CookedValue"
        )
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True
    )

    return float(result.stdout.strip())


# ============================================================
# DATASET
# ============================================================

def load_data():

    data = load_breast_cancer()

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# MODEL CREATION
# ============================================================

def create_model(model_name):

    if model_name == "Logistic Regression":

        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    random_state=RANDOM_STATE
                )
            )
        ])

    elif model_name == "Random Forest":

        return RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    elif model_name == "SVM":

        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                SVC()
            )
        ])

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# CHILD PROCESS
# ============================================================

def run_model_workload(
    model_name,
    n_repetitions,
    result_queue
):
    """
    Runs the actual model training workload.

    A fresh model is created for every repetition so that
    every fit is an independent experiment.
    """

    try:

        X_train, X_test, y_train, y_test = load_data()

        # ----------------------------------------------------
        # Warm-up
        # ----------------------------------------------------

        warmup_model = create_model(model_name)
        warmup_model.fit(X_train, y_train)

        del warmup_model
        gc.collect()

        # ----------------------------------------------------
        # Tell parent process that preparation is complete
        # ----------------------------------------------------

        result_queue.put({
            "status": "READY"
        })

        # Small stabilization period
        time.sleep(1)

        # ----------------------------------------------------
        # Start workload
        # ----------------------------------------------------

        start_time = time.perf_counter()

        for _ in range(n_repetitions):

            model = create_model(model_name)

            model.fit(
                X_train,
                y_train
            )

            del model

        end_time = time.perf_counter()

        total_training_time = end_time - start_time

        # ----------------------------------------------------
        # Send result back to parent
        # ----------------------------------------------------

        result_queue.put({
            "status": "DONE",
            "training_time": total_training_time
        })

    except Exception as e:

        result_queue.put({
            "status": "ERROR",
            "error": str(e)
        })


# ============================================================
# RUN ONE ENERGY TRIAL
# ============================================================

def run_energy_trial(model_name, trial_number):

    print(
        f"\nTrial {trial_number}/{N_TRIALS} "
        f"| {model_name}"
    )

    queue = multiprocessing.Queue()

    process = multiprocessing.Process(
        target=run_model_workload,
        args=(
            model_name,
            N_REPETITIONS,
            queue
        )
    )

    process.start()

    # --------------------------------------------------------
    # Wait until child process has completed preparation
    # --------------------------------------------------------

    message = queue.get()

    if message["status"] == "ERROR":

        process.join()

        raise RuntimeError(
            f"{model_name} failed: "
            f"{message['error']}"
        )

    if message["status"] != "READY":

        process.join()

        raise RuntimeError(
            f"Unexpected message: {message}"
        )

    # --------------------------------------------------------
    # Energy BEFORE workload
    # --------------------------------------------------------

    energy_before = get_energy_counter()

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Tell child process to continue
    # --------------------------------------------------------

    # The child is currently sleeping for 1 second.
    # Give it a moment so the measurement begins after
    # the baseline energy reading.

    time.sleep(0.1)

    # --------------------------------------------------------
    # Wait for workload completion
    # --------------------------------------------------------

    result = queue.get()

    end_time = time.perf_counter()

    energy_after = get_energy_counter()

    process.join()

    # --------------------------------------------------------
    # Validate result
    # --------------------------------------------------------

    if result["status"] != "DONE":

        raise RuntimeError(
            f"{model_name} failed: "
            f"{result.get('error', 'Unknown error')}"
        )

    # --------------------------------------------------------
    # Calculate energy difference
    # --------------------------------------------------------

    energy_delta = energy_after - energy_before

    total_time = result["training_time"]

    energy_per_fit = (
        energy_delta / N_REPETITIONS
    )

    time_per_fit = (
        total_time / N_REPETITIONS
    )

    print(
        f"  Energy before : {energy_before}"
    )

    print(
        f"  Energy after  : {energy_after}"
    )

    print(
        f"  Raw energy Δ  : {energy_delta}"
    )

    print(
        f"  Total time    : {total_time:.6f} s"
    )

    print(
        f"  Energy/fit    : {energy_per_fit}"
    )

    print(
        f"  Time/fit      : {time_per_fit:.6f} s"
    )

    return {
        "Model": model_name,
        "Trial": trial_number,
        "Repetitions": N_REPETITIONS,
        "Energy_Before_Raw": energy_before,
        "Energy_After_Raw": energy_after,
        "Energy_Delta_Raw": energy_delta,
        "Energy_Per_Fit_Raw": energy_per_fit,
        "Total_Training_Time_s": total_time,
        "Training_Time_Per_Fit_s": time_per_fit
    }


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():

    print("=" * 70)
    print(" GREEN AI — RAPL ENERGY BENCHMARK")
    print("=" * 70)

    print("\nConfiguration:")
    print(f"Trials       : {N_TRIALS}")
    print(f"Repetitions  : {N_REPETITIONS}")
    print(f"Energy       : {ENERGY_COUNTER}")

    print("\nIMPORTANT:")
    print(
        "Energy values are stored as RAW Windows "
        "counter values."
    )

    print(
        "No physical unit conversion is assumed."
    )

    models = [
        "Logistic Regression",
        "Random Forest",
        "SVM"
    ]

    all_results = []

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    for model_name in models:

        print("\n" + "=" * 70)
        print(f"MODEL: {model_name}")
        print("=" * 70)

        for trial in range(1, N_TRIALS + 1):

            result = run_energy_trial(
                model_name,
                trial
            )

            all_results.append(result)

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    results_df.to_csv(
        "results/metrics/energy_benchmark_results.csv",
        index=False
    )

    # --------------------------------------------------------
    # Create summary
    # --------------------------------------------------------

    summary = (
        results_df
        .groupby("Model")
        .agg(
            Average_Energy_Delta_Raw=(
                "Energy_Delta_Raw",
                "mean"
            ),

            Energy_Delta_Std=(
                "Energy_Delta_Raw",
                "std"
            ),

            Average_Energy_Per_Fit_Raw=(
                "Energy_Per_Fit_Raw",
                "mean"
            ),

            Energy_Per_Fit_Std=(
                "Energy_Per_Fit_Raw",
                "std"
            ),

            Average_Training_Time_s=(
                "Training_Time_Per_Fit_s",
                "mean"
            ),

            Training_Time_Std=(
                "Training_Time_Per_Fit_s",
                "std"
            )
        )
        .reset_index()
    )

    summary.to_csv(
        "results/metrics/energy_benchmark_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # Display summary
    # --------------------------------------------------------

    print("\n\n")
    print("=" * 70)
    print(" ENERGY BENCHMARK SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print("\nFiles created:")

    print(
        "  results/metrics/"
        "energy_benchmark_results.csv"
    )

    print(
        "  results/metrics/"
        "energy_benchmark_summary.csv"
    )

    print("\n" + "=" * 70)
    print(" BENCHMARK COMPLETE")
    print("=" * 70)


# ============================================================
# WINDOWS ENTRY POINT
# ============================================================

if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()