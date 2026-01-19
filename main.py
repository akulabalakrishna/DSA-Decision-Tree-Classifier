"""
Decision Tree Classifier from Scratch - Main entry point.
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from custom_ds.tree_node import TreeNode
from custom_ds.hash_table import HashTable
from custom_ds.queue_ds import Queue

from custom_ml.decision_tree import DecisionTreeClassifier
from custom_ml.gini import gini_impurity, majority_class

from preprocessing.data_loader import (
    load_adult_data, 
    train_test_split, 
    get_dataset_info,
    ADULT_FEATURE_NAMES,
    ADULT_FEATURE_TYPES
)
from preprocessing.cleaner import clean_data
from preprocessing.encoder import DataEncoder

from evaluation.metrics import accuracy, classification_report, print_confusion_matrix, root_mean_squared_error
from evaluation.benchmark import compare_with_sklearn

from utils.config import Config


def print_banner():
    banner = """
    +==============================================================+
    |    DECISION TREE CLASSIFIER - IMPLEMENTED FROM SCRATCH       |
    |                                                              |
    |    Using Custom Data Structures:                             |
    |    * TreeNode   * HashTable   * Queue                        |
    |                                                              |
    |    Split Criterion: Gini Impurity                            |
    |    Algorithm: CART (Binary Splits)                           |
    +==============================================================+
    """
    print(banner)


def load_and_preprocess_data():
    print("\n" + "=" * 60)
    print("STEP 1: LOADING DATA")
    print("=" * 60)
    
    data_path = os.path.join(PROJECT_ROOT, Config.DATA_DIR)
    
    csv_file = os.path.join(data_path, "adult.csv")
    train_file = os.path.join(data_path, "adult.data")
    
    if not os.path.exists(csv_file) and not os.path.exists(train_file):
        print(f"\nERROR: No dataset file found in {data_path}")
        print("\nPlease add one of the following:")
        print("  Option 1: adult.csv (single file with headers)")
        print("  Option 2: adult.data + adult.test (UCI format)")
        print(f"\nPlace files in: {data_path}")
        sys.exit(1)
    
    dataset = load_adult_data(data_path)
    
    train_data = dataset['train_data']
    train_labels = dataset['train_labels']
    test_data = dataset['test_data']
    test_labels = dataset['test_labels']
    feature_types = dataset['feature_types']
    feature_names = dataset['feature_names']
    
    print("\n" + "=" * 60)
    print("STEP 2: CLEANING DATA")
    print("=" * 60)
    
    print("\nCleaning training data:")
    train_data, train_labels = clean_data(train_data, train_labels)
    
    print("\nCleaning test data:")
    test_data, test_labels = clean_data(test_data, test_labels)
    
    print("\n" + "=" * 60)
    print("STEP 3: ENCODING FEATURES")
    print("=" * 60)
    
    encoder = DataEncoder(feature_types)
    
    train_data = encoder.fit_transform(train_data, train_labels)
    test_data = encoder.transform(test_data)
    
    encoder.describe()
    
    print("\nTraining Set:")
    get_dataset_info(train_data, train_labels, feature_names)
    
    print("\nTest Set:")
    get_dataset_info(test_data, test_labels, feature_names)
    
    return {
        'train_data': train_data,
        'train_labels': train_labels,
        'test_data': test_data,
        'test_labels': test_labels,
        'feature_types': feature_types,
        'feature_names': feature_names,
        'encoder': encoder
    }


def train_decision_tree(train_data, train_labels, feature_types, feature_names):
    print("\n" + "=" * 60)
    print("STEP 4: TRAINING DECISION TREE")
    print("=" * 60)
    
    Config.print_config()
    
    model = DecisionTreeClassifier(
        max_depth=Config.MAX_DEPTH,
        min_samples_split=Config.MIN_SAMPLES_SPLIT,
        min_impurity_decrease=Config.MIN_IMPURITY_DECREASE
    )
    
    print("Training custom Decision Tree...")
    start_time = time.time()
    
    model.fit(train_data, train_labels, feature_types, feature_names)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    print(f"\nTree Statistics:")
    print(f"  Depth:       {model.get_depth()}")
    print(f"  Total Nodes: {model.count_nodes()}")
    print(f"  Leaf Nodes:  {model.count_leaves()}")
    
    return model


def evaluate_model(model, train_data, train_labels, test_data, test_labels):
    print("\n" + "=" * 60)
    print("STEP 5: EVALUATING MODEL")
    print("=" * 60)
    
    print("\nMaking predictions on training set...")
    train_predictions = model.predict(train_data)
    train_accuracy = accuracy(train_labels, train_predictions)
    
    print("Making predictions on test set...")
    test_predictions = model.predict(test_data)
    test_accuracy = accuracy(test_labels, test_predictions)
    
    # Calculate RMSE if applicable (converted to float)
    test_rmse = root_mean_squared_error(test_labels, test_predictions)
    
    print("\n" + "-" * 40)
    print("RESULTS")
    print("-" * 40)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy:     {test_accuracy * 100:.2f}%")
    print(f"Test RMSE:         {test_rmse:.4f}")
    print(f"Tree Depth:        {model.get_depth()}")
    print("-" * 40)
    
    print(classification_report(test_labels, test_predictions))
    print_confusion_matrix(test_labels, test_predictions)
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_predictions': train_predictions,
        'test_predictions': test_predictions,
        'tree_depth': model.get_depth()
    }


def save_results(results, model, feature_names):
    print("\n" + "=" * 60)
    print("STEP 6: SAVING RESULTS")
    print("=" * 60)
    
    Config.ensure_output_dirs(PROJECT_ROOT)
    
    metrics_path = os.path.join(PROJECT_ROOT, Config.METRICS_DIR, Config.RESULTS_FILE)
    
    with open(metrics_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DECISION TREE CLASSIFIER RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Model: Custom Decision Tree (CART-style)\n")
        f.write("Split Criterion: Gini Impurity\n\n")
        
        f.write("Hyperparameters:\n")
        f.write(f"  max_depth:             {Config.MAX_DEPTH}\n")
        f.write(f"  min_samples_split:     {Config.MIN_SAMPLES_SPLIT}\n")
        f.write(f"  min_impurity_decrease: {Config.MIN_IMPURITY_DECREASE}\n\n")
        
        f.write("Results:\n")
        f.write(f"  Training Accuracy: {results['train_accuracy'] * 100:.2f}%\n")
        f.write(f"  Test Accuracy:     {results['test_accuracy'] * 100:.2f}%\n")
        f.write(f"  Tree Depth:        {results['tree_depth']}\n")
        f.write(f"  Total Nodes:       {model.count_nodes()}\n")
        f.write(f"  Leaf Nodes:        {model.count_leaves()}\n")
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"Results saved to: {metrics_path}")
    
    tree_path = os.path.join(PROJECT_ROOT, Config.TREES_DIR, Config.TREE_FILE)
    
    with open(tree_path, 'w') as f:
        f.write("DECISION TREE STRUCTURE\n")
        f.write("=" * 60 + "\n")
        f.write(f"(Showing first 5 levels)\n\n")
        f.write(model.get_tree_string(max_depth=5))
    
    print(f"Tree structure saved to: {tree_path}")


def main():
    print_banner()
    
    data = load_and_preprocess_data()
    
    model = train_decision_tree(
        data['train_data'],
        data['train_labels'],
        data['feature_types'],
        data['feature_names']
    )
    
    results = evaluate_model(
        model,
        data['train_data'],
        data['train_labels'],
        data['test_data'],
        data['test_labels']
    )
    
    save_results(results, model, data['feature_names'])
    
    if Config.RUN_SKLEARN_BENCHMARK:
        try:
            compare_with_sklearn(
                data['train_data'],
                data['train_labels'],
                data['test_data'],
                data['test_labels'],
                results['test_predictions'],
                data['feature_types']
            )
        except Exception as e:
            print(f"\nSklearn benchmark skipped: {e}")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\nCheck outputs in: {os.path.join(PROJECT_ROOT, Config.OUTPUT_DIR)}")
print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
