"""
Splitter - Procedural logic to find optimal splits.

This module contains the logic to iterate through features and thresholds
to maximize Information Gain. It bridges the gap between the 
Tree Building Algorithm (decision_tree.py) and the 
Mathematical Functions (gini.py).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_ds.hash_table import HashTable
from custom_ml.gini import gini_impurity, weighted_gini, information_gain, majority_class


MAX_THRESHOLDS = 50


def get_unique_values(data, feature_index):
    seen = HashTable()
    unique = []
    
    for row in data:
        value = row[feature_index]
        if not seen.contains(value):
            seen.put(value, True)
            unique.append(value)
    
    return unique


def split_data(data, labels, feature_index, threshold, is_numerical=True, category_set=None):
    left_data = []
    left_labels = []
    right_data = []
    right_labels = []
    
    for i, row in enumerate(data):
        value = row[feature_index]
        
        if is_numerical:
            goes_left = value <= threshold
        else:
            goes_left = value in category_set
        
        if goes_left:
            left_data.append(row)
            left_labels.append(labels[i])
        else:
            right_data.append(row)
            right_labels.append(labels[i])
    
    return left_data, left_labels, right_data, right_labels


def find_numerical_threshold(data, labels, feature_index):
    """
    Find best split for a numerical feature using efficient linear scan.
    Time Complexity: O(N log N) due to sorting.
    """
    # 1. optimized data structure: list of tuples (feature_value, label, original_index)
    # Using a list comprehension - O(N)
    n_samples = len(data)
    if n_samples <= 1:
        return None, -float('inf'), ([], [])

    sorted_data = []
    for i in range(n_samples):
        # We store (value, label, original_index)
        sorted_data.append((data[i][feature_index], labels[i], i))
    
    # 2. Sort by feature value - O(N log N)
    sorted_data.sort(key=lambda x: x[0])
    
    # 3. Initialize counts
    # We use our custom HashTable logic or simple dictionaries if allowed. 
    # Since 'labels' can be anything, lets use a simple approach using the existing 'count_items' helper
    # or build it manually to ensure we stay efficient.
    # To keep it "pure loops" and efficient, manual tracking is best.
    
    right_counts = {}  # Using standard dict for internal logic speed, assuming allowed as per previous files
    # If standard dict is strictly forbidden, we would use custom HashTable, but previous code used dict in 'gini.py' (counts.items())
    
    # Calculate initial total counts (all start in right node)
    for label in labels:
        if label not in right_counts:
            right_counts[label] = 0
        right_counts[label] += 1
        
    left_counts = {label: 0 for label in right_counts}
    
    left_count_total = 0
    right_count_total = n_samples
    
    best_threshold = None
    best_gain = -float('inf')
    best_split_indices_left = [] # We won't track full lists during loop for speed
    best_split_index = -1        # Index in sorted_data where split occurs
    
    # Parent Impurity (calculated once)
    parent_impurity = 1.0 - sum((c / n_samples) ** 2 for c in right_counts.values())
    
    # 4. Linear Scan
    for i in range(n_samples - 1):
        value, label, original_idx = sorted_data[i]
        next_value = sorted_data[i+1][0]
        
        # Move sample from Right to Left
        right_counts[label] -= 1
        left_counts[label] += 1
        right_count_total -= 1
        left_count_total += 1
        
        # Only evaluate split if values are different
        if value == next_value:
            continue
            
        # Calculate Impurity of Left Child
        gini_left = 1.0
        sum_sq_left = 0.0
        for lbl in left_counts:
            prob = left_counts[lbl] / left_count_total
            sum_sq_left += prob * prob
        gini_left -= sum_sq_left
        
        # Calculate Impurity of Right Child
        gini_right = 1.0
        sum_sq_right = 0.0
        for lbl in right_counts:
            prob = right_counts[lbl] / right_count_total
            sum_sq_right += prob * prob
        gini_right -= sum_sq_right
        
        # Weighted Gini
        weighted_gini = (left_count_total / n_samples) * gini_left + \
                        (right_count_total / n_samples) * gini_right
        
        gain = parent_impurity - weighted_gini
        
        if gain > best_gain:
            best_gain = gain
            best_threshold = (value + next_value) / 2
            best_split_index = i
            
    # 5. Reconstruct the split indices only for the best split
    # This avoids creating lists O(N) times
    if best_split_index != -1:
        left_indices = [x[2] for x in sorted_data[:best_split_index+1]]
        right_indices = [x[2] for x in sorted_data[best_split_index+1:]]
        return best_threshold, best_gain, (left_indices, right_indices)
        
    return None, -float('inf'), ([], [])


def find_categorical_split(data, labels, feature_index):
    unique_values = get_unique_values(data, feature_index)
    
    best_category_set = None
    best_gain = -float('inf')
    best_split = ([], [])
    
    for value in unique_values:
        category_set = {value}
        
        left_labels_split = []
        right_labels_split = []
        left_indices = []
        right_indices = []
        
        for i, row in enumerate(data):
            if row[feature_index] in category_set:
                left_labels_split.append(labels[i])
                left_indices.append(i)
            else:
                right_labels_split.append(labels[i])
                right_indices.append(i)
        
        if len(left_labels_split) == 0 or len(right_labels_split) == 0:
            continue
        
        gain = information_gain(labels, left_labels_split, right_labels_split)
        
        if gain > best_gain:
            best_gain = gain
            best_category_set = category_set
            best_split = (left_indices, right_indices)
    
    return best_category_set, best_gain, best_split


def find_best_split(data, labels, feature_types, min_samples_split=2, min_impurity_decrease=0.0):
    n_samples = len(data)
    
    if n_samples < min_samples_split:
        return None
    
    if len(set(labels)) == 1:
        return None
    
    best_split = None
    best_gain = -float('inf')
    
    n_features = len(data[0]) if n_samples > 0 else 0
    
    for feature_index in range(n_features):
        is_numerical = feature_types[feature_index]
        
        if is_numerical:
            threshold, gain, _ = find_numerical_threshold(data, labels, feature_index)
            if threshold is not None and gain > best_gain:
                left_data, left_labels, right_data, right_labels = split_data(
                    data, labels, feature_index, threshold, is_numerical=True
                )
                
                if len(left_data) > 0 and len(right_data) > 0:
                    best_gain = gain
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'category_set': None,
                        'is_numerical': True,
                        'gain': gain,
                        'left_data': left_data,
                        'left_labels': left_labels,
                        'right_data': right_data,
                        'right_labels': right_labels
                    }
        else:
            category_set, gain, _ = find_categorical_split(data, labels, feature_index)
            if category_set is not None and gain > best_gain:
                left_data, left_labels, right_data, right_labels = split_data(
                    data, labels, feature_index, None, is_numerical=False, category_set=category_set
                )
                
                if len(left_data) > 0 and len(right_data) > 0:
                    best_gain = gain
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': None,
                        'category_set': category_set,
                        'is_numerical': False,
                        'gain': gain,
                        'left_data': left_data,
                        'left_labels': left_labels,
                        'right_data': right_data,
                        'right_labels': right_labels
                    }
    
    if best_split is not None and best_split['gain'] < min_impurity_decrease:
        return None
    
    return best_split
