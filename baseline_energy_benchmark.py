import subprocess
import time
import multiprocessing
import gc

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

# Baseline workload will be repeated for approximately
# the same duration as the model workload.
BASELINE_SLEEP_BUFFER = 0.05


# ============================================================
# RAPL ENERGY READER
# ============================================================

def get_energy_counter():

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
            (
                "scaler",
                StandardScaler()
            ),
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
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                SVC()
            )
        ])

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )


# ============================================================
# BASELINE PROCESS
# ============================================================

def run_baseline_workload(
    duration,
    result_queue
):
    """
    Performs no ML training.

    The process simply remains active for approximately
    the same duration as the model workload.

    This gives us a background/package-energy reference.
    """

    try:

        start = time.perf_counter()

        while (
            time.perf_counter() - start
            < duration
        ):

            # Small amount of Python activity so that the
            # process behaves similarly to an active workload.
            _ = 1 + 1

        elapsed = (
            time.perf_counter() - start
        )

        result_queue.put({
            "status": "DONE",
            "duration": elapsed
        })

    except Exception as e:

        result_queue.put({
            "status": "ERROR",
            "error": str(e)
        })


# ============================================================
# MODEL WORKLOAD PROCESS
# ============================================================

def run_model_workload(
    model_name,
    n_repetitions,
    result_queue
):

    try:

        X_train, X_test, y_train, y_test = load_data()

        # ----------------------------------------------------
        # Warm-up
        # ----------------------------------------------------

        warmup_model = create_model(
            model_name
        )

        warmup_model.fit(
            X_train,
            y_train
        )

        del warmup_model
        gc.collect()

        # ----------------------------------------------------
        # Tell parent preparation is complete
        # ----------------------------------------------------

        result_queue.put({
            "status": "READY"
        })

        # Give the system a short stabilization period.
        time.sleep(1)

        # ----------------------------------------------------
        # MODEL TRAINING
        # ----------------------------------------------------

        start = time.perf_counter()

        for _ in range(n_repetitions):

            model = create_model(
                model_name
            )

            model.fit(
                X_train,
                y_train
            )

            del model

        end = time.perf_counter()

        total_time = end - start

        result_queue.put({
            "status": "DONE",
            "training_time": total_time
        })

    except Exception as e:

        result_queue.put({
            "status": "ERROR",
            "error": str(e)
        })


# ============================================================
# MEASURE BASELINE
# ============================================================

def measure_baseline(duration):

    queue = multiprocessing.Queue()

    process = multiprocessing.Process(
        target=run_baseline_workload,
        args=(
            duration,
            queue
        )
    )

    process.start()

    # Small delay to make sure the process is running.
    time.sleep(0.1)

    energy_before = get_energy_counter()

    result = queue.get()

    energy_after = get_energy_counter()

    process.join()

    if result["status"] != "DONE":

        raise RuntimeError(
            result.get(
                "error",
                "Baseline process failed."
            )
        )

    energy_delta = (
        energy_after -
        energy_before
    )

    return {
        "energy_before": energy_before,
        "energy_after": energy_after,
        "energy_delta": energy_delta,
        "duration": result["duration"]
    }


# ============================================================
# MEASURE MODEL WORKLOAD
# ============================================================

def measure_model(
    model_name,
    n_repetitions
):

    queue = multiprocessing.Queue()

    process = multiprocessing.Process(
        target=run_model_workload,
        args=(
            model_name,
            n_repetitions,
            queue
        )
    )

    process.start()

    # --------------------------------------------------------
    # Wait for model preparation
    # --------------------------------------------------------

    ready_message = queue.get()

    if ready_message["status"] != "READY":

        process.join()

        raise RuntimeError(
            ready_message.get(
                "error",
                "Model preparation failed."
            )
        )

    # --------------------------------------------------------
    # Read energy immediately before workload
    # --------------------------------------------------------

    energy_before = get_energy_counter()

    # --------------------------------------------------------
    # Wait for model workload
    # --------------------------------------------------------

    result = queue.get()

    energy_after = get_energy_counter()

    process.join()

    if result["status"] != "DONE":

        raise RuntimeError(
            result.get(
                "error",
                "Model process failed."
            )
        )

    energy_delta = (
        energy_after -
        energy_before
    )

    training_time = result[
        "training_time"
    ]

    return {
        "energy_before": energy_before,
        "energy_after": energy_after,
        "energy_delta": energy_delta,
        "training_time": training_time
    }


# ============================================================
# RUN ONE TRIAL
# ============================================================

def run_trial(
    model_name,
    trial_number
):

    print(
        f"\nTrial {trial_number}/{N_TRIALS}"
        f" | {model_name}"
    )

    # --------------------------------------------------------
    # First run the model workload.
    #
    # We need its duration so that the baseline measurement
    # can use approximately the same time window.
    # --------------------------------------------------------

    model_result = measure_model(
        model_name,
        N_REPETITIONS
    )

    model_duration = (
        model_result["training_time"]
        + BASELINE_SLEEP_BUFFER
    )

    # --------------------------------------------------------
    # Baseline measurement
    # --------------------------------------------------------

    baseline_result = measure_baseline(
        model_duration
    )

    # --------------------------------------------------------
    # Values
    # --------------------------------------------------------

    workload_energy = (
        model_result["energy_delta"]
    )

    baseline_energy = (
        baseline_result["energy_delta"]
    )

    corrected_energy = (
        workload_energy -
        baseline_energy
    )

    corrected_energy_per_fit = (
        corrected_energy /
        N_REPETITIONS
    )

    energy_per_fit = (
        workload_energy /
        N_REPETITIONS
    )

    time_per_fit = (
        model_result["training_time"] /
        N_REPETITIONS
    )

    print(
        f"  Workload energy Δ : "
        f"{workload_energy}"
    )

    print(
        f"  Baseline energy Δ : "
        f"{baseline_energy}"
    )

    print(
        f"  Corrected energy  : "
        f"{corrected_energy}"
    )

    print(
        f"  Corrected / fit   : "
        f"{corrected_energy_per_fit}"
    )

    print(
        f"  Time / fit        : "
        f"{time_per_fit:.6f} s"
    )

    return {

        "Model": model_name,

        "Trial": trial_number,

        "Repetitions": N_REPETITIONS,

        "Workload_Energy_Delta_Raw":
            workload_energy,

        "Baseline_Energy_Delta_Raw":
            baseline_energy,

        "Corrected_Energy_Raw":
            corrected_energy,

        "Raw_Energy_Per_Fit":
            energy_per_fit,

        "Corrected_Energy_Per_Fit":
            corrected_energy_per_fit,

        "Model_Training_Time_s":
            model_result["training_time"],

        "Training_Time_Per_Fit_s":
            time_per_fit,

        "Baseline_Duration_s":
            baseline_result["duration"]
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        " GREEN AI — BASELINE-CORRECTED ENERGY BENCHMARK"
    )
    print("=" * 70)

    print("\nConfiguration:")

    print(
        f"Trials       : {N_TRIALS}"
    )

    print(
        f"Repetitions  : {N_REPETITIONS}"
    )

    print(
        f"Energy       : {ENERGY_COUNTER}"
    )

    print("\nEnergy values are stored as RAW")
    print("Windows RAPL counter values.")

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
    # Run all experiments
    # --------------------------------------------------------

    for model_name in models:

        print("\n" + "=" * 70)

        print(
            f"MODEL: {model_name}"
        )

        print("=" * 70)

        for trial in range(
            1,
            N_TRIALS + 1
        ):

            result = run_trial(
                model_name,
                trial
            )

            all_results.append(
                result
            )

    # --------------------------------------------------------
    # Detailed results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    results_df.to_csv(
        "results/metrics/"
        "baseline_corrected_energy_results.csv",
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = (
        results_df
        .groupby("Model")
        .agg(

            Average_Workload_Energy_Raw=(
                "Workload_Energy_Delta_Raw",
                "mean"
            ),

            Workload_Energy_Std=(
                "Workload_Energy_Delta_Raw",
                "std"
            ),

            Average_Baseline_Energy_Raw=(
                "Baseline_Energy_Delta_Raw",
                "mean"
            ),

            Baseline_Energy_Std=(
                "Baseline_Energy_Delta_Raw",
                "std"
            ),

            Average_Corrected_Energy_Raw=(
                "Corrected_Energy_Raw",
                "mean"
            ),

            Corrected_Energy_Std=(
                "Corrected_Energy_Raw",
                "std"
            ),

            Average_Corrected_Energy_Per_Fit=(
                "Corrected_Energy_Per_Fit",
                "mean"
            ),

            Corrected_Energy_Per_Fit_Std=(
                "Corrected_Energy_Per_Fit",
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
        "results/metrics/"
        "baseline_corrected_energy_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n\n")

    print("=" * 70)
    print(
        " BASELINE-CORRECTED ENERGY SUMMARY"
    )
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print("\nFiles created:")

    print(
        "  results/metrics/"
        "baseline_corrected_energy_results.csv"
    )

    print(
        "  results/metrics/"
        "baseline_corrected_energy_summary.csv"
    )

    print("\n" + "=" * 70)

    print(
        " BENCHMARK COMPLETE"
    )

    print("=" * 70)


# ============================================================
# WINDOWS ENTRY POINT
# ============================================================

if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()