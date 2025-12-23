"""
Metrics - Evaluation metrics for classification.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_ds.hash_table import HashTable


def accuracy(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    if len(y_true) == 0:
        return 0.0
    
    correct = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1
    
    return correct / len(y_true)


def confusion_matrix(y_true, y_pred, labels=None):
    if labels is None:
        label_set = HashTable()
        for label in y_true:
            label_set.put(label, True)
        for label in y_pred:
            label_set.put(label, True)
        labels = sorted(label_set.keys())
    
    n_labels = len(labels)
    
    label_to_idx = HashTable()
    for i, label in enumerate(labels):
        label_to_idx.put(label, i)
    
    matrix = [[0] * n_labels for _ in range(n_labels)]
    
    for true, pred in zip(y_true, y_pred):
        true_idx = label_to_idx.get(true, -1)
        pred_idx = label_to_idx.get(pred, -1)
        
        if true_idx >= 0 and pred_idx >= 0:
            matrix[true_idx][pred_idx] += 1
    
    return {
        'matrix': matrix,
        'labels': labels
    }


def precision_recall_f1(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels)
    matrix = cm['matrix']
    labels = cm['labels']
    n_labels = len(labels)
    
    metrics = {}
    
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(n_labels)) - tp
        fn = sum(matrix[i][j] for j in range(n_labels)) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': sum(matrix[i])
        }
    
    total_support = len(y_true)
    avg_precision = sum(m['precision'] * m['support'] for m in metrics.values()) / total_support if total_support > 0 else 0
    avg_recall = sum(m['recall'] * m['support'] for m in metrics.values()) / total_support if total_support > 0 else 0
    avg_f1 = sum(m['f1'] * m['support'] for m in metrics.values()) / total_support if total_support > 0 else 0
    
    metrics['weighted_avg'] = {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'support': total_support
    }
    
    return metrics


def classification_report(y_true, y_pred, labels=None):
    metrics = precision_recall_f1(y_true, y_pred, labels)
    
    lines = []
    lines.append("")
    lines.append("Classification Report:")
    lines.append("-" * 60)
    lines.append(f"{'Label':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    lines.append("-" * 60)
    
    for label, m in metrics.items():
        if label == 'weighted_avg':
            continue
        lines.append(f"{str(label):<20} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f} {m['support']:>10}")
    
    lines.append("-" * 60)
    m = metrics['weighted_avg']
    lines.append(f"{'Weighted Avg':<20} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f} {m['support']:>10}")
    lines.append("")
    
    return "\n".join(lines)


def print_confusion_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels)
    matrix = cm['matrix']
    labels = cm['labels']
    
    max_label_len = max(len(str(l)) for l in labels)
    max_label_len = max(max_label_len, 10)
    
    print("\nConfusion Matrix:")
    print("-" * (max_label_len + 5 + len(labels) * 8))
    
    header = " " * (max_label_len + 2) + "Predicted"
    print(header)
    header = " " * (max_label_len + 2) + " ".join(f"{str(l)[:7]:>7}" for l in labels)
    print(header)
    
    for i, label in enumerate(labels):
        row_str = f"{str(label):<{max_label_len}} | "
        row_str += " ".join(f"{matrix[i][j]:>7}" for j in range(len(labels)))
        print(row_str)
    
    print("-" * (max_label_len + 5 + len(labels) * 8))
