"""
Cleaner - Handle missing values and text cleanup.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_ds.hash_table import HashTable, count_items


def find_mode(values):
    counts = HashTable()
    
    for value in values:
        if value != '?':
            counts.increment(value)
    
    max_count = -1
    mode = None
    
    for val, count in counts.items():
        if count > max_count:
            max_count = count
            mode = val
    
    return mode


def get_column(data, col_idx):
    return [row[col_idx] for row in data]


def handle_missing_values(data, missing_marker='?'):
    if len(data) == 0:
        return data
    
    n_cols = len(data[0])
    
    modes = []
    for col_idx in range(n_cols):
        column_values = get_column(data, col_idx)
        mode = find_mode(column_values)
        modes.append(mode)
    
    missing_count = 0
    for row in data:
        for col_idx in range(n_cols):
            if row[col_idx] == missing_marker:
                if modes[col_idx] is not None:
                    row[col_idx] = modes[col_idx]
                    missing_count += 1
    
    if missing_count > 0:
        print(f"  Replaced {missing_count} missing values with mode")
    
    return data


def clean_whitespace(data):
    for row in data:
        for i in range(len(row)):
            if isinstance(row[i], str):
                row[i] = row[i].strip()
    return data


def normalize_labels(labels):
    for i in range(len(labels)):
        if isinstance(labels[i], str):
            labels[i] = labels[i].strip().rstrip('.')
    return labels


def clean_data(data, labels, missing_marker='?'):
    print("Cleaning data...")
    
    data = clean_whitespace(data)
    labels = normalize_labels(labels)
    data = handle_missing_values(data, missing_marker)
    
    print("  Data cleaning complete")
    
    return data, labels


def count_missing(data, missing_marker='?'):
    if len(data) == 0:
        return {}
    
    n_cols = len(data[0])
    missing_counts = HashTable()
    
    for row in data:
        for col_idx in range(n_cols):
            if row[col_idx] == missing_marker:
                missing_counts.increment(col_idx)
    
    return dict(missing_counts.items())


def remove_rows_with_missing(data, labels, missing_marker='?'):
    filtered_data = []
    filtered_labels = []
    
    for i, row in enumerate(data):
        has_missing = False
        for value in row:
            if value == missing_marker:
                has_missing = True
                break
        
        if not has_missing:
            filtered_data.append(row)
            filtered_labels.append(labels[i])
    
    removed = len(data) - len(filtered_data)
    if removed > 0:
        print(f"  Removed {removed} rows with missing values")
    
    return filtered_data, filtered_labels
