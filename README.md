# Decision Tree Classifier from Scratch

A complete implementation of a **CART-style Decision Tree Classifier** using custom data structures. This project demonstrates understanding of both machine learning algorithms and fundamental data structures.

## 🌟 Features

- **100% From Scratch**: No sklearn, TensorFlow, PyTorch, or XGBoost for training
- **Custom Data Structures**:
  - `TreeNode`: Custom tree node for decision tree structure
  - `HashTable`: Custom hash table with chaining (replaces Python dict)
  - `Queue`: Custom FIFO queue for BFS traversal
- **CART Algorithm**: Binary splits using Gini impurity
- **Mixed Feature Types**: Supports both numerical and categorical features
- **Pre-Pruning**: Configurable max_depth, min_samples_split, min_impurity_decrease

## 📚 Terminology Clarification

To ensure technical accuracy (as per Course Requirements):

- **Algorithm**: The procedural logic that builds the model. In this project, it is the **CART (Classification and Regression Trees)** algorithm. It uses a recursive approach to build a binary tree.  
  *Location: `custom_ml/decision_tree.py`*

- **Mathematical Functions**: The tools used by the algorithm to make decisions. These are NOT the algorithm itself.
  - **Gini Impurity**: Measures the likelihood of incorrect classification.
  - **Information Gain**: measures the reduction in impurity.  
  *Location: `custom_ml/gini.py`*

- **Evaluation Metrics**: The results used to measure the performance of the trained model.
  - **Accuracy**: Overall correctness.
  - **RMSE**: Root Mean Squared Error (added for completeness).
  - **Confusion Matrix**: Detailed breakdown of predictions.  
  *Location: `evaluation/metrics.py`*

## 📁 Project Structure

```
decision_tree_from_scratch/
│
├── main.py                     # Entry point
│
├── data/
│   ├── adult.data              # Training dataset
│   └── adult.test              # Test dataset
│
├── custom_ds/                  # Custom Data Structures
│   ├── tree_node.py            # TreeNode class
│   ├── hash_table.py           # Custom hash table
│   └── queue_ds.py             # Custom queue (BFS)
│
├── custom_ml/                  # Custom ML Components
│   ├── gini.py                 # Gini impurity
│   ├── splitter.py             # Best split logic
│   ├── decision_tree.py        # CART tree builder
│   └── predictor.py            # Tree traversal
│
├── preprocessing/              # Data Preprocessing
│   ├── data_loader.py          # Load Adult dataset
│   ├── cleaner.py              # Handle missing values
│   └── encoder.py              # Encode features
│
├── evaluation/                 # Model Evaluation
│   ├── metrics.py              # Accuracy, F1, etc.
│   └── benchmark.py            # Optional sklearn comparison
│
├── outputs/                    # Generated outputs
│   ├── logs/
│   ├── metrics/
│   └── trees/
│
├── utils/
│   └── config.py               # Hyperparameters
│
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Download the Dataset

Download the **Adult Income Dataset** from UCI ML Repository:
- [adult.data](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data)
- [adult.test](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test)

Place both files in the `data/` directory.

### 2. Install Dependencies (Optional)

```bash
pip install -r requirements.txt
```

> Note: The core implementation has **no external dependencies**. Sklearn is only used for optional benchmarking.

### 3. Run the Classifier

```bash
python main.py
```

## 📊 Expected Output

```
╔══════════════════════════════════════════════════════════════╗
║    DECISION TREE CLASSIFIER - IMPLEMENTED FROM SCRATCH       ║
╚══════════════════════════════════════════════════════════════╝

STEP 1: LOADING DATA
Loaded 32561 training samples from adult.data
Loaded 16281 test samples from adult.test

STEP 4: TRAINING DECISION TREE
Training completed in X.XX seconds
  Depth: 10
  Total Nodes: XXXX
  Leaf Nodes: XXXX

STEP 5: EVALUATING MODEL
Training Accuracy: ~85%
Test Accuracy:     ~83%
```

## ⚙️ Configuration

Edit `utils/config.py` to adjust hyperparameters:

```python
MAX_DEPTH = 10              # Maximum tree depth
MIN_SAMPLES_SPLIT = 2       # Minimum samples to split
MIN_IMPURITY_DECREASE = 0.001  # Minimum impurity decrease
```

## 🧠 Algorithm Details

### Split Criterion: Gini Impurity

```
Gini(S) = 1 - Σ(p_i)²
```

Where `p_i` is the probability of class `i` in set `S`.

### Split Types

- **Numerical Features**: `value <= threshold` vs `value > threshold`
- **Categorical Features**: `value ∈ S` vs `value ∉ S`

### Pre-Pruning

The tree uses pre-pruning to prevent overfitting:
1. **Max Depth**: Stop splitting at maximum depth
2. **Min Samples Split**: Require minimum samples to split
3. **Min Impurity Decrease**: Require minimum improvement

## 📈 Dataset: Adult Income

- **Task**: Predict if income exceeds $50K/year
- **Samples**: ~32K training, ~16K test
- **Features**: 14 (mix of numerical and categorical)
- **Classes**: `<=50K`, `>50K`

## 🔧 Custom Data Structures

### HashTable

Custom hash table using separate chaining:
- `put(key, value)`: Insert/update
- `get(key)`: Retrieve value
- `increment(key)`: For counting
- Dynamic resizing at 75% load factor

### Queue

Custom FIFO queue using linked list:
- `enqueue(item)`: Add to back
- `dequeue()`: Remove from front
- O(1) operations

### TreeNode

Custom tree node storing:
- Split information (feature, threshold, category set)
- Left/right child references
- Leaf predictions
- Metadata (depth, samples, impurity)

## 📝 License

This project is for educational purposes.

## 👨‍💻 Author

Academic project demonstrating Decision Tree implementation from scratch.
