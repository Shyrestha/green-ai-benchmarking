import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METRICS_DIR = ROOT / "results" / "metrics"

SAFE_STEPS = [
    ("Combine benchmark results", "combine_results.py"),
    ("Analyse model trade-offs", "green_ai_analysis.py"),
    ("Generate research plots", "plot_green_ai_research.py"),
]

FULL_STEPS = [
    ("Repeated timing and accuracy benchmark", "repeated_benchmark.py"),
    ("Peak memory benchmark", "peak_memory_benchmark.py"),
    ("RAPL energy-counter benchmark", "energy_benchmark.py"),
] + SAFE_STEPS

REQUIRED_FILES = [
    METRICS_DIR / "repeated_benchmark_summary.csv",
    METRICS_DIR / "peak_memory_benchmark_summary.csv",
    METRICS_DIR / "energy_benchmark_summary.csv",
]


def run_script(description, script_name):
    script_path = ROOT / script_name

    if not script_path.is_file():
        raise FileNotFoundError(f"Script not found: {script_name}")

    print("\n" + "=" * 65, flush=True)
    print(f"STEP: {description}", flush=True)
    print(f"Running: {script_name}", flush=True)
    print("=" * 65, flush=True)

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=ROOT,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Green AI benchmarking workflow"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Rerun timing, peak-memory and RAPL counter benchmarks",
    )
    args = parser.parse_args()

    if args.full:
        steps = FULL_STEPS
        print("\nFULL BENCHMARK MODE")
        print("This will rerun experiments and overwrite their result CSVs.")
        print("RAPL output is a raw counter measurement, not verified joules.")
    else:
        steps = SAFE_STEPS
        print("\nSAFE MODE: reusing existing benchmark measurements.")
        print("No timing, memory or energy experiments will be rerun.")

    print(f"Project folder: {ROOT}")

    if not args.full:
        missing = [p for p in REQUIRED_FILES if not p.is_file()]
        if missing:
            print("\nCannot continue. Missing required summary files:")
            for path in missing:
                print(f" - {path.relative_to(ROOT)}")
            return 1

    try:
        for description, script_name in steps:
            run_script(description, script_name)

    except subprocess.CalledProcessError as error:
        failed_script = Path(error.cmd[-1]).name
        print(f"\nERROR: {failed_script} failed.")
        print(f"Exit code: {error.returncode}")
        print("Workflow stopped; inspect the error above before continuing.")
        return error.returncode or 1

    except FileNotFoundError as error:
        print(f"\nERROR: {error}")
        return 1

    print("\n" + "=" * 65)
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print(f"Master results: {METRICS_DIR / 'green_ai_master_results.csv'}")
    print(f"Trade-off analysis: {METRICS_DIR / 'green_ai_tradeoff_analysis.csv'}")
    print(f"Plots folder: {ROOT / 'results' / 'plots'}")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
