# Decision Tree Classifier Implemented from Scratch
## Z5007 Programming and Data Structures - Milestone 3

**Author:** [Student Name]
**Course:** Z5007 Programming and Data Structures
**Submission Date:** January 19, 2026

---

## 1. Project Overview

The objective of this project is to implement a **Classification and Regression Tree (CART)** algorithm entirely from scratch, using **custom data structures** (Hash Table, Queue, Tree Nodes) and avoiding any machine learning libraries for the core logic.

### Key Features & Constraints
- **100% Custom Implementation**: The decision tree training, split finding, and prediction logic are written from scratch.
- **No ML Libraries for Training**: `scikit-learn`, `tensorflow`, or `pytorch` are **strictly forbidden** for the model implementation.
- **Data Handling**: `numpy` and `pandas` are used solely for data loading and matrix manipulation, as permitted.
- **Benchmarking**: `scikit-learn` is included **only** for the purpose of validating results and creating a baseline comparison.
- **Optimization**: Features an efficient $O(N \log N)$ split-finding algorithm using linear scans and incremental Gini updates.

---

## 2. Requirements

This project requires **Python 3.8+**.

### Dependencies
The project relies on the following packages for *data manipulation* and *visualization*:

- **numpy** (>=1.21.0): For efficient numerical array operations.
- **pandas**: For loading and handling CSV data.
- **matplotlib** (>=3.5.0): For generating plots.
- **seaborn** (>=0.11.0): For enhanced visualization styling.
- **scikit-learn** (>=1.0.0): **Used ONLY for benchmarking and metrics calculation (not training).**

---

## 3. Setup and Installation

### Step 1: Clone or Extract Project
Ensure you have the project source code in a directory (e.g., `DSA-Decision-Tree-Classifier`).

### Step 2: Create a Virtual Environment (Recommended)
It is best practice to run this project in an isolated environment.

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
Install all required libraries using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## 4. How to Run

To train the model, evaluate performance, and generate results, run the main entry point:

```bash
python main.py
```

### Pipeline Steps
The script performs the following fully automated pipeline:
1.  **Loading Data**: Loads the Adult Income dataset from `data/`.
2.  **Cleaning & Encoding**: Handles missing values and encodes categorical features.
3.  **Training**: Builds the Decision Tree using custom `TreeNode` and `HashTable` structures with optimizations.
4.  **Evaluation**: Calculates Accuracy and RMSE on the test set.
5.  **Benchmarking**: (Optional) Trains a standard Sklearn model on the exact same data to compare performance.
6.  **Visualization**: Generates Confusion Matrix and Tree Structure plots in the `outputs/` directory.

---

## 5. Example Output

Upon successful execution, the terminal will display the following summary:

```text
STEP 4: TRAINING DECISION TREE
============================================================
  Training on 32561 samples, 108 features
  max_depth=10, min_samples_split=2
  Building tree...
  Done! Created 311 nodes in 18.4s

STEP 5: EVALUATING MODEL
============================================================
RESULTS
----------------------------------------
Training Accuracy: 85.78%
Test Accuracy:     85.60%
Test RMSE:         0.3795
Tree Depth:        10
----------------------------------------

Confusion Matrix:
[[11687   748]
 [ 1596  2250]]

============================================================
SKLEARN BENCHMARK COMPARISON
(For validation only - custom tree is fully independent)
============================================================
Custom Decision Tree Accuracy:  85.60%
Sklearn Decision Tree Accuracy: 85.50%
Agreement Rate:                 96.19%
```

---

## 6. Project Structure

The codebase is organized modularly to separate Data Structures, Algorithms, and Evaluation.

```text
DSA-Decision-Tree-Classifier/
├── main.py                     # MAIN ENTRY POINT - Run this file
├── requirements.txt            # List of dependencies
├── README.md                   # Project documentation
│
├── custom_ds/                  # Custom Data Structures (Grading Criterion 2)
│   ├── hash_table.py           # Custom HashTable (replaces dict)
│   ├── queue_ds.py             # Custom FIFO Queue (for BFS)
│   └── tree_node.py            # Node structure for the Tree
│
├── custom_ml/                  # Core Algorithm Implementation (Grading Criterion 1)
│   ├── decision_tree.py        # The CART Algorithm logic (Recursive build)
│   ├── splitter.py             # Optimization logic for finding best splits
│   └── gini.py                 # Mathematical Functions (Gini Index)
│
├── processing/                 # Data Pipeline
│   ├── data_loader.py          # CSV loading utils
│   ├── cleaner.py              # Missing value handling
│   └── encoder.py              # Categorical to Numerical encoding
│
├── evaluation/                 # Metrics and Plots
│   ├── metrics.py              # Accuracy, RMSE, Confusion Matrix code
│   ├── visualization.py        # Plotting functions (matplotlib)
│   └── benchmark.py            # Sklearn comparison wrapper
│
├── data/                       # Dataset Directory
│   ├── adult.csv               # Primary dataset
│   └── adult.data / .test      # Raw UCI format (if applicable)
│
├── outputs/                    # Generated Artifacts
│   ├── metrics/                # Text report of results
│   ├── plots/                  # Confusion Matrix & Tree PNGs
│   └── trees/                  # Text representation of tree structure
│
└── tests/                      # Unit Tests (Grading Criterion 4)
    ├── test_data_structures.py # Tests for Stack, Queue, HashTable
    ├── test_decision_tree.py   # Integration tests
    └── test_splitter.py        # Tests for split optimization
```

---

## 7. Troubleshooting

### 1. "Python is not recognized as an internal or external command"
**Cause:** Python is not added to your system PATH.
**Fix:** Reinstall Python and check the box "Add Python to PATH" during installation, or try running `py main.py` or `python3 main.py`.

### 2. "ModuleNotFoundError: No module named 'numpy'/'pandas'..."
**Cause:** Dependencies are not installed in the current environment.
**Fix:** Ensure your virtual environment is active and run `pip install -r requirements.txt`.

### 3. "FileNotFoundError: No dataset file found in data/"
**Cause:** The `adult.csv` or `adult.data` file is missing.
**Fix:** Download the "Adult" dataset from the UCI Machine Learning Repository and place the files in the `data/` folder as structured above.

### 4. "ImportError: cannot import name..."
**Cause:** Circular imports or corrupted `.pyc` files.
**Fix:** Delete all `__pycache__` folders and run the script again.

---
**License:** Educational Use Only.
