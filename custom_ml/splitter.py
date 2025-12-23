"""
Splitter - Find the best split for a decision tree node.
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


def sample_thresholds(thresholds, max_samples=MAX_THRESHOLDS):
    if len(thresholds) <= max_samples:
        return thresholds
    
    step = len(thresholds) / max_samples
    sampled = []
    for i in range(max_samples):
        idx = int(i * step)
        sampled.append(thresholds[idx])
    
    return sampled


def find_numerical_threshold(data, labels, feature_index):
    values = [(data[i][feature_index], labels[i], i) for i in range(len(data))]
    values.sort(key=lambda x: x[0])
    
    best_threshold = None
    best_gain = -float('inf')
    best_split = ([], [])
    
    thresholds = []
    for i in range(len(values) - 1):
        if values[i][0] != values[i + 1][0]:
            midpoint = (values[i][0] + values[i + 1][0]) / 2
            thresholds.append(midpoint)
    
    thresholds = sample_thresholds(thresholds)
    
    for threshold in thresholds:
        left_labels_split = []
        right_labels_split = []
        left_indices = []
        right_indices = []
        
        for value, label, idx in values:
            if value <= threshold:
                left_labels_split.append(label)
                left_indices.append(idx)
            else:
                right_labels_split.append(label)
                right_indices.append(idx)
        
        if len(left_labels_split) == 0 or len(right_labels_split) == 0:
            continue
        
        gain = information_gain(labels, left_labels_split, right_labels_split)
        
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
            best_split = (left_indices, right_indices)
    
    return best_threshold, best_gain, best_split


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
