"""
Encoder - Encode categorical features for decision tree processing.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_ds.hash_table import HashTable


class DataEncoder:
    
    def __init__(self, feature_types):
        self.feature_types = feature_types
        self.n_features = len(feature_types)
        self.category_maps = {}
        self.reverse_maps = {}
        self.label_map = HashTable()
        self.reverse_label_map = []
        self.is_fitted = False
    
    def fit(self, data, labels=None):
        for col_idx in range(self.n_features):
            if not self.feature_types[col_idx]:
                cat_map = HashTable()
                reverse_map = []
                
                for row in data:
                    value = row[col_idx]
                    if not cat_map.contains(value):
                        code = len(reverse_map)
                        cat_map.put(value, code)
                        reverse_map.append(value)
                
                self.category_maps[col_idx] = cat_map
                self.reverse_maps[col_idx] = reverse_map
        
        if labels is not None:
            for label in labels:
                if not self.label_map.contains(label):
                    code = len(self.reverse_label_map)
                    self.label_map.put(label, code)
                    self.reverse_label_map.append(label)
        
        self.is_fitted = True
        return self
    
    def transform(self, data, encode_categorical=False):
        transformed = []
        
        for row in data:
            new_row = []
            for col_idx in range(self.n_features):
                value = row[col_idx]
                
                if self.feature_types[col_idx]:
                    try:
                        new_row.append(float(value))
                    except (ValueError, TypeError):
                        new_row.append(0.0)
                else:
                    if encode_categorical:
                        cat_map = self.category_maps.get(col_idx)
                        if cat_map and cat_map.contains(value):
                            new_row.append(cat_map.get(value))
                        else:
                            new_row.append(-1)
                    else:
                        new_row.append(value)
            
            transformed.append(new_row)
        
        return transformed
    
    def fit_transform(self, data, labels=None, encode_categorical=False):
        self.fit(data, labels)
        return self.transform(data, encode_categorical)
    
    def transform_labels(self, labels):
        return [self.label_map.get(label, -1) for label in labels]
    
    def inverse_transform_labels(self, encoded_labels):
        return [self.reverse_label_map[code] if 0 <= code < len(self.reverse_label_map) else None 
                for code in encoded_labels]
    
    def get_category_values(self, feature_index):
        return self.reverse_maps.get(feature_index, [])
    
    def get_label_values(self):
        return self.reverse_label_map
    
    def describe(self):
        print("\nEncoder Information:")
        print(f"  Features: {self.n_features}")
        
        n_numerical = sum(1 for t in self.feature_types if t)
        n_categorical = self.n_features - n_numerical
        
        print(f"  Numerical features: {n_numerical}")
        print(f"  Categorical features: {n_categorical}")
        
        if self.reverse_label_map:
            print(f"  Labels: {self.reverse_label_map}")
        
        print("\n  Categorical feature categories:")
        for col_idx in sorted(self.reverse_maps.keys()):
            n_cats = len(self.reverse_maps[col_idx])
            print(f"    Feature {col_idx}: {n_cats} categories")


def encode_data(data, labels, feature_types):
    encoder = DataEncoder(feature_types)
    encoded_data = encoder.fit_transform(data, labels, encode_categorical=False)
    
    return encoded_data, labels, encoder
