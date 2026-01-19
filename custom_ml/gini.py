"""
Gini Impurity - Mathematical Functions for split quality.

TERMINOLOGY CLARIFICATION:
These are "Mathematical Functions" (tools), NOT the Algorithm.
- They are used to MEASURE the purity of a node.
- The Algorithm (Decision Tree) uses these tools to make decisions.
- Gini Impurity formula: 1 - sum(p_i^2)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_ds.hash_table import HashTable, count_items


def gini_impurity(labels):
    if len(labels) == 0:
        return 0.0
    
    counts = count_items(labels)
    total = len(labels)
    
    sum_squared = 0.0
    for count in counts.values():
        probability = count / total
        sum_squared += probability * probability
    
    return 1.0 - sum_squared


def weighted_gini(left_labels, right_labels):
    n_left = len(left_labels)
    n_right = len(right_labels)
    n_total = n_left + n_right
    
    if n_total == 0:
        return 0.0
    
    gini_left = gini_impurity(left_labels)
    gini_right = gini_impurity(right_labels)
    
    weighted = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
    
    return weighted


def information_gain(parent_labels, left_labels, right_labels):
    parent_gini = gini_impurity(parent_labels)
    children_gini = weighted_gini(left_labels, right_labels)
    
    return parent_gini - children_gini


def class_distribution(labels):
    return count_items(labels)


def majority_class(labels):
    if len(labels) == 0:
        return None
    
    counts = count_items(labels)
    
    max_count = -1
    majority = None
    
    for class_label, count in counts.items():
        if count > max_count:
            max_count = count
            majority = class_label
    
    return majority
