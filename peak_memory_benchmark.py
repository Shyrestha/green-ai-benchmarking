import os
import time
import gc
import multiprocessing
import psutil
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# ---------------------------------------------------------
# CREATE MODEL
# ---------------------------------------------------------

def create_model(model_name):

    if model_name == "Logistic Regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                random_state=42
            )
        )

    elif model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

    elif model_name == "SVM":
        return make_pipeline(
            StandardScaler(),
            SVC()
        )

    else:
        raise ValueError("Unknown model")


# ---------------------------------------------------------
# CHILD PROCESS
# ---------------------------------------------------------

def train_model(model_name, connection):

    # Load dataset inside the isolated process
    data = load_breast_cancer()

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create a fresh model
    model = create_model(model_name)

    # Send signal to parent process:
    # dataset and model are ready
    connection.send("READY")

    # Give parent process a moment to record baseline memory
    time.sleep(0.5)

    # Start training
    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    end_time = time.perf_counter()

    training_time = end_time - start_time

    # Tell parent that training has finished
    connection.send({
        "status": "DONE",
        "training_time": training_time
    })

    connection.close()


# ---------------------------------------------------------
# RUN ONE MODEL
# ---------------------------------------------------------

def benchmark_model(model_name):

    parent_connection, child_connection = multiprocessing.Pipe()

    process = multiprocessing.Process(
        target=train_model,
        args=(model_name, child_connection)
    )

    process.start()

    child_process = psutil.Process(process.pid)

    # Wait until the child process is ready
    message = parent_connection.recv()

    if message != "READY":
        raise RuntimeError("Child process did not become ready.")

    # Record baseline memory
    baseline_memory = (
        child_process.memory_info().rss
        / (1024 * 1024)
    )

    peak_memory = baseline_memory

    training_finished = False
    training_time = None

    # Monitor memory while training
    while process.is_alive():

        try:
            current_memory = (
                child_process.memory_info().rss
                / (1024 * 1024)
            )

            peak_memory = max(
                peak_memory,
                current_memory
            )

        except psutil.NoSuchProcess:
            break

        # Check whether training has finished
        if parent_connection.poll():

            result = parent_connection.recv()

            if isinstance(result, dict):
                training_finished = True
                training_time = result["training_time"]

        time.sleep(0.005)

    # Make sure final message is received
    while parent_connection.poll():

        result = parent_connection.recv()

        if isinstance(result, dict):
            training_finished = True
            training_time = result["training_time"]

    process.join()

    peak_memory_increase = (
        peak_memory - baseline_memory
    )

    return {
        "Model": model_name,
        "Baseline Memory (MB)": baseline_memory,
        "Peak Memory (MB)": peak_memory,
        "Peak Memory Increase (MB)": peak_memory_increase,
        "Training Time (s)": training_time
    }


# ---------------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------------

def main():

    models = [
        "Logistic Regression",
        "Random Forest",
        "SVM"
    ]

    number_of_runs = 5

    results = []

    print("\n==========================================")
    print(" ISOLATED PEAK MEMORY BENCHMARK")
    print("==========================================")

    for run in range(1, number_of_runs + 1):

        print(f"\n========== RUN {run} ==========")

        for model_name in models:

            print(f"\nTesting {model_name}...")

            gc.collect()

            result = benchmark_model(model_name)

            result["Run"] = run

            results.append(result)

            print(
                f"Baseline Memory : "
                f"{result['Baseline Memory (MB)']:.2f} MB"
            )

            print(
                f"Peak Memory     : "
                f"{result['Peak Memory (MB)']:.2f} MB"
            )

            print(
                f"Peak Increase   : "
                f"{result['Peak Memory Increase (MB)']:+.2f} MB"
            )

            print(
                f"Training Time   : "
                f"{result['Training Time (s)']:.6f} s"
            )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate summary
    summary_df = results_df.groupby("Model").agg(
        Average_Peak_Memory=(
            "Peak Memory (MB)",
            "mean"
        ),

        Peak_Memory_Std=(
            "Peak Memory (MB)",
            "std"
        ),

        Average_Peak_Increase=(
            "Peak Memory Increase (MB)",
            "mean"
        ),

        Peak_Increase_Std=(
            "Peak Memory Increase (MB)",
            "std"
        ),

        Average_Training_Time=(
            "Training Time (s)",
            "mean"
        ),

        Training_Time_Std=(
            "Training Time (s)",
            "std"
        )
    ).reset_index()

    # Create output directory
    os.makedirs(
        "results/metrics",
        exist_ok=True
    )

    # Save detailed results
    results_df.to_csv(
        "results/metrics/"
        "peak_memory_benchmark_results.csv",
        index=False
    )

    # Save summary
    summary_df.to_csv(
        "results/metrics/"
        "peak_memory_benchmark_summary.csv",
        index=False
    )

    print("\n==========================================")
    print(" SUMMARY")
    print("==========================================")

    print(summary_df.to_string(index=False))

    print("\nResults saved to:")

    print(
        "results/metrics/"
        "peak_memory_benchmark_results.csv"
    )

    print(
        "results/metrics/"
        "peak_memory_benchmark_summary.csv"
    )


# ---------------------------------------------------------
# WINDOWS SAFE ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()