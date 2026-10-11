## 🌱 Green AI Benchmarking Framework

A Python-based machine learning benchmarking project that compares predictive performance, training time, prediction time, process memory usage, and raw CPU energy-counter measurements across different machine learning models.

## 📌 Project Overview

Machine learning models can achieve similar predictive performance while requiring different amounts of computational resources.

This project investigates the trade-offs between model accuracy and computational cost to help understand how machine learning choices relate to Green AI.

## 🎯 Objectives

- Compare machine learning models using the same dataset and train-test split.
- Measure model accuracy and execution time.
- Examine process memory usage during model training.
- Collect raw RAPL energy-counter measurements where supported.
- Analyse trade-offs between predictive performance and computational cost.
- Generate CSV reports and visualizations for further analysis.

## 🧠 Models Compared

The project evaluates three classification algorithms:

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Support Vector Machine (SVM)**

## 📊 Dataset

The project uses the **Breast Cancer Wisconsin Diagnostic dataset** available through scikit-learn.

- Samples: 569
- Features: 30
- Task: Binary classification

The dataset is divided into training and testing sets using a stratified split, with an 80:20 ratio and a fixed random state of 42.

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- psutil
- Windows Performance Monitoring / RAPL counters where available

## ⚙️ Setup and Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Green-ai-benchmark
```

Replace the placeholder with your actual GitHub repository URL after uploading the project.

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use an appropriate execution-policy setting for your environment.

### 4. Install dependencies

```bash
python -m pip install numpy pandas scikit-learn matplotlib psutil
```

## ▶️ Running the Project

### Safe mode — analyse existing benchmark results

```bash
python run_green_ai.py
```

Safe mode reuses existing benchmark measurements, combines the results, analyses model trade-offs, and generates research visualizations. It does not rerun the timing, memory, or energy-counter experiments.

### Full benchmark mode — rerun experiments

```bash
python run_green_ai.py --full
```

Full mode reruns the timing, memory, and raw energy-counter benchmarks before rebuilding the combined analysis and plots.

**Important:** Full mode can overwrite existing benchmark CSV files. Back up results you want to preserve before running it.

## 📁 Project Structure

```text
Green-ai-benchmark/
├── run_green_ai.py
├── repeated_benchmark.py
├── peak_memory_benchmark.py
├── energy_benchmark.py
├── combine_results.py
├── green_ai_analysis.py
├── plot_green_ai_research.py
├── plot_results.py
├── results/
│   ├── metrics/
│   └── plots/
└── README.md
```

Other experiment and diagnostic scripts are also included in the project.

## 📈 Results and Visualizations

The workflow generates CSV files and plots to help compare the evaluated models.

Examples of the generated visualizations include:

- Model training-time comparison
- Accuracy versus training time
- Peak process memory comparison
- Raw RAPL counter comparison
- Model accuracy comparison

Combined metrics are stored in `results/metrics/`, while visualizations are stored in `results/plots/`.

## 🔬 Measurement Methodology and Limitations

This project is an experimental benchmark, and its measurements should be interpreted carefully.

- **Accuracy:** The reported accuracy uses one fixed train-test split. Results may differ with other splits or datasets.
- **Execution time:** Training and prediction times are measured experimentally and may vary with system load and software conditions.
- **Memory:** The reported peak resident set size (RSS) includes the Python process, loaded libraries, dataset, model, and temporary allocations. It is not an exact measurement of memory used by the model alone. Peak RSS is sampled, so brief peaks may be missed.
- **Energy-counter measurements:** The recorded RAPL values are raw counter deltas per fit. They have not been validated or converted into physical energy units such as joules. Package-level measurements may include operating-system and background activity.
- **Comparability:** Timing, memory, and energy-counter measurements are collected in separate experiments. They should not be interpreted as perfectly synchronized measurements of one execution.
- **Scope:** Findings apply to the selected dataset, model configurations, software environment, and test machine. They should not be generalized to all machine learning workloads.

## 🚀 Future Improvements

- Evaluate additional datasets and machine learning algorithms.
- Repeat accuracy evaluation across multiple train-test splits.
- Improve measurement isolation and quantify system-background effects.
- Validate energy measurement units and methodology.
- Add confidence intervals and more detailed statistical analysis.
- Automate experiment reports and improve reproducibility.

## 🌍 Why Green AI?

Green AI emphasizes not only what machine learning models can achieve, but also the computational resources required to achieve it.

By measuring predictive performance alongside execution time, memory usage, and energy-related indicators, this project explores how model selection can support more resource-aware machine learning.

## 👩‍💻 Author

Developed as a machine learning and Green AI benchmarking project using Python.
