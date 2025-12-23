"""
Decision Tree Classifier - CART-style recursive tree builder.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_ds.tree_node import TreeNode
from custom_ds.hash_table import count_items
from custom_ds.queue_ds import Queue
from custom_ml.gini import gini_impurity, majority_class
from custom_ml.splitter import find_best_split


class DecisionTreeClassifier:
    
    def __init__(self, max_depth=10, min_samples_split=2, min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        
        self.root = None
        self.feature_types = None
        self.feature_names = None
        self.n_classes = 0
        self.classes = None
        self.tree_depth = 0
        
        self._node_count = 0
        self._start_time = None
        self._last_progress_time = 0
    
    def fit(self, X, y, feature_types=None, feature_names=None):
        n_samples = len(X)
        n_features = len(X[0]) if n_samples > 0 else 0
        
        if feature_types is None:
            self.feature_types = []
            for j in range(n_features):
                is_num = True
                for i in range(min(n_samples, 100)):
                    try:
                        float(X[i][j])
                    except (ValueError, TypeError):
                        is_num = False
                        break
                self.feature_types.append(is_num)
        else:
            self.feature_types = feature_types
        
        self.feature_names = feature_names
        
        class_counts = count_items(y)
        self.classes = class_counts.keys()
        self.n_classes = len(self.classes)
        
        print(f"  Training on {n_samples} samples, {n_features} features")
        print(f"  max_depth={self.max_depth}, min_samples_split={self.min_samples_split}")
        print(f"  Building tree...")
        
        self._node_count = 0
        self._start_time = time.time()
        self._last_progress_time = 0
        
        self.root = TreeNode()
        self._build_tree(self.root, X, y, depth=0)
        
        elapsed = time.time() - self._start_time
        print(f"  Done! Created {self._node_count} nodes in {elapsed:.1f}s")
        
        self.tree_depth = self._get_tree_depth(self.root)
        
        return self
    
    def _build_tree(self, node, X, y, depth):
        n_samples = len(X)
        self._node_count += 1
        
        elapsed = time.time() - self._start_time
        if elapsed - self._last_progress_time >= 10:
            print(f"    Progress: {self._node_count} nodes, depth {depth}, {n_samples} samples, {elapsed:.1f}s elapsed")
            self._last_progress_time = elapsed
        
        node.depth = depth
        node.n_samples = n_samples
        node.impurity = gini_impurity(y)
        node.class_distribution = count_items(y)
        
        should_stop = False
        
        if depth >= self.max_depth:
            should_stop = True
        
        if n_samples < self.min_samples_split:
            should_stop = True
        
        if len(set(y)) == 1:
            should_stop = True
        
        if n_samples == 0:
            should_stop = True
        
        if should_stop:
            prediction = majority_class(y)
            node.make_leaf(prediction, n_samples, node.impurity, node.class_distribution)
            return
        
        best_split = find_best_split(
            X, y, 
            self.feature_types,
            self.min_samples_split,
            self.min_impurity_decrease
        )
        
        if best_split is None:
            prediction = majority_class(y)
            node.make_leaf(prediction, n_samples, node.impurity, node.class_distribution)
            return
        
        left_child, right_child = node.make_split(
            feature_index=best_split['feature_index'],
            threshold=best_split['threshold'],
            category_set=best_split['category_set'],
            is_numerical=best_split['is_numerical'],
            n_samples=n_samples,
            impurity=node.impurity
        )
        
        self._build_tree(left_child, best_split['left_data'], best_split['left_labels'], depth + 1)
        self._build_tree(right_child, best_split['right_data'], best_split['right_labels'], depth + 1)
    
    def predict(self, X):
        predictions = []
        for sample in X:
            pred = self._predict_single(self.root, sample)
            predictions.append(pred)
        return predictions
    
    def _predict_single(self, node, sample):
        if node.is_leaf:
            return node.prediction
        
        feature_value = sample[node.feature_index]
        
        if node.go_left(feature_value):
            return self._predict_single(node.left, sample)
        else:
            return self._predict_single(node.right, sample)
    
    def _get_tree_depth(self, node):
        if node is None or node.is_leaf:
            return 0
        
        left_depth = self._get_tree_depth(node.left) if node.left else 0
        right_depth = self._get_tree_depth(node.right) if node.right else 0
        
        return 1 + max(left_depth, right_depth)
    
    def get_depth(self):
        return self.tree_depth
    
    def print_tree(self, node=None, indent="", feature_names=None):
        if node is None:
            node = self.root
        
        if feature_names is None:
            feature_names = self.feature_names
        
        if node.is_leaf:
            print(f"{indent}Predict: {node.prediction} (samples: {node.n_samples})")
            return
        
        if feature_names and node.feature_index < len(feature_names):
            feat_name = feature_names[node.feature_index]
        else:
            feat_name = f"feature[{node.feature_index}]"
        
        if node.is_numerical:
            print(f"{indent}if {feat_name} <= {node.threshold:.4f}:")
        else:
            print(f"{indent}if {feat_name} in {node.category_set}:")
        
        print(f"{indent}  [Left branch]")
        self.print_tree(node.left, indent + "    ", feature_names)
        
        print(f"{indent}  [Right branch]")
        self.print_tree(node.right, indent + "    ", feature_names)
    
    def get_tree_string(self, max_depth=3):
        lines = []
        self._tree_to_string(self.root, "", lines, max_depth, 0)
        return "\n".join(lines)
    
    def _tree_to_string(self, node, indent, lines, max_depth, current_depth):
        if node is None:
            return
        
        if current_depth > max_depth:
            lines.append(f"{indent}...")
            return
        
        if node.is_leaf:
            lines.append(f"{indent}Predict: {node.prediction} (n={node.n_samples})")
            return
        
        feat_name = f"X[{node.feature_index}]"
        if self.feature_names and node.feature_index < len(self.feature_names):
            feat_name = self.feature_names[node.feature_index]
        
        if node.is_numerical:
            lines.append(f"{indent}[{feat_name} <= {node.threshold:.2f}]")
        else:
            lines.append(f"{indent}[{feat_name} in {node.category_set}]")
        
        self._tree_to_string(node.left, indent + "  L: ", lines, max_depth, current_depth + 1)
        self._tree_to_string(node.right, indent + "  R: ", lines, max_depth, current_depth + 1)
    
    def count_nodes(self):
        if self.root is None:
            return 0
        
        queue = Queue()
        queue.enqueue(self.root)
        count = 0
        
        while not queue.is_empty():
            node = queue.dequeue()
            count += 1
            
            if not node.is_leaf:
                if node.left:
                    queue.enqueue(node.left)
                if node.right:
                    queue.enqueue(node.right)
        
        return count
    
    def count_leaves(self):
        if self.root is None:
            return 0
        
        queue = Queue()
        queue.enqueue(self.root)
        count = 0
        
        while not queue.is_empty():
            node = queue.dequeue()
            
            if node.is_leaf:
                count += 1
            else:
                if node.left:
                    queue.enqueue(node.left)
                if node.right:
                    queue.enqueue(node.right)
        
        return count
