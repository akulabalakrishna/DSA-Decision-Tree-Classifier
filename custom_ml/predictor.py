"""
Predictor - Tree traversal and prediction utilities.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_ds.queue_ds import Queue, Stack


def predict_single(root, sample):
    node = root
    
    while not node.is_leaf:
        feature_value = sample[node.feature_index]
        
        if node.go_left(feature_value):
            node = node.left
        else:
            node = node.right
    
    return node.prediction


def predict_batch(root, samples):
    return [predict_single(root, sample) for sample in samples]


def predict_with_probability(root, sample):
    node = root
    
    while not node.is_leaf:
        feature_value = sample[node.feature_index]
        
        if node.go_left(feature_value):
            node = node.left
        else:
            node = node.right
    
    distribution = {}
    if node.class_distribution:
        total = sum(node.class_distribution.values())
        for label, count in node.class_distribution.items():
            distribution[label] = count / total if total > 0 else 0
    
    return node.prediction, distribution


def get_leaf_path(root, sample):
    path = []
    node = root
    
    while not node.is_leaf:
        feature_value = sample[node.feature_index]
        
        if node.go_left(feature_value):
            path.append((node, 'left'))
            node = node.left
        else:
            path.append((node, 'right'))
            node = node.right
    
    path.append((node, 'leaf'))
    return path


def get_feature_importance(root, n_features):
    importance = [0.0] * n_features
    total_samples = root.n_samples if root else 0
    
    if root is None or total_samples == 0:
        return importance
    
    queue = Queue()
    queue.enqueue(root)
    
    while not queue.is_empty():
        node = queue.dequeue()
        
        if not node.is_leaf:
            left_weight = node.left.n_samples / total_samples if node.left else 0
            right_weight = node.right.n_samples / total_samples if node.right else 0
            node_weight = node.n_samples / total_samples
            
            left_impurity = node.left.impurity if node.left else 0
            right_impurity = node.right.impurity if node.right else 0
            
            impurity_decrease = (node_weight * node.impurity - 
                                left_weight * left_impurity - 
                                right_weight * right_impurity)
            
            importance[node.feature_index] += impurity_decrease
            
            if node.left:
                queue.enqueue(node.left)
            if node.right:
                queue.enqueue(node.right)
    
    total = sum(importance)
    if total > 0:
        importance = [imp / total for imp in importance]
    
    return importance


def tree_to_rules(root, feature_names=None, current_rules=None):
    if current_rules is None:
        current_rules = []
    
    rules = []
    
    stack = Stack()
    stack.push((root, []))
    
    while not stack.is_empty():
        node, conditions = stack.pop()
        
        if node.is_leaf:
            if conditions:
                rule = " AND ".join(conditions)
                rules.append(f"IF {rule} THEN {node.prediction}")
            else:
                rules.append(f"DEFAULT: {node.prediction}")
        else:
            feat_name = feature_names[node.feature_index] if feature_names else f"X[{node.feature_index}]"
            
            if node.is_numerical:
                left_cond = f"{feat_name} <= {node.threshold:.2f}"
                right_cond = f"{feat_name} > {node.threshold:.2f}"
            else:
                left_cond = f"{feat_name} in {node.category_set}"
                right_cond = f"{feat_name} not in {node.category_set}"
            
            stack.push((node.right, conditions + [right_cond]))
            stack.push((node.left, conditions + [left_cond]))
    
    return rules
