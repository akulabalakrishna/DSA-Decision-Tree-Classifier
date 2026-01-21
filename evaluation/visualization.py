"""
Visualization - Functions to generate plots for model evaluation.
"""

import os
import math
import matplotlib.pyplot as plt
import numpy as np

from evaluation.metrics import confusion_matrix
from custom_ds.hash_table import HashTable


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix", save_path=None):
    """
    Generate and save a confusion matrix heatmap using Matplotlib.
    """
    cm_data = confusion_matrix(y_true, y_pred, labels)
    matrix = np.array(cm_data['matrix'])
    labels = cm_data['labels']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Show all ticks and label them with the respective list entries
    ax.set(xticks=np.arange(matrix.shape[1]),
           yticks=np.arange(matrix.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = np.max(matrix) / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i][j], 'd'),
                    ha="center", va="center",
                    color="white" if matrix[i][j] > thresh else "black")
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()


def plot_metrics(metrics_dict, save_path=None):
    """
    Plot bar chart for key metrics.
    metrics_dict: {'Train Accuracy': 0.85, 'Test Accuracy': 0.82, 'Test RMSE': 0.45}
    """
    # Filter out non-numeric or incompatible types
    plot_data = {k: v for k, v in metrics_dict.items() if isinstance(v, (int, float))}
    
    if not plot_data:
        return

    names = list(plot_data.keys())
    values = list(plot_data.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=['#4c72b0', '#55a868', '#c44e52', '#8172b3'])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}',
                 ha='center', va='bottom')
    
    plt.title('Model Performance Metrics')
    plt.ylim(0, 1.1)  # Assuming metrics like accuracy are 0-1. RMSE might exceed, but OK for now.
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Metrics plot saved to: {save_path}")
    else:
        plt.show()


def visualize_tree(model, save_path=None):
    """
    Visualizes the decision tree structure using Matplotlib.
    This recursively draws nodes and edges.
    """
    if model.root is None:
        print("Empty tree, nothing to visualize.")
        return
        
    depth = model.get_depth()
    
    # Dynamic figure size based on depth/width estimate
    fig_width = max(12, depth * 2) 
    fig_height = max(8, depth * 1.5)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_axis_off()
    
    # Helper for layout
    def get_width(node):
        if node.is_leaf:
            return 1
        return get_width(node.left) + get_width(node.right)
    
    total_width = get_width(model.root)
    
    def plot_node(node, x, y, width, level):
        # Draw current node
        if node.is_leaf:
            text = f"Class:\n{node.prediction}"
            color = '#e0f3db' # Light green for leaves
            box_style = 'round,pad=0.5'
        else:
            if model.feature_names and node.feature_index < len(model.feature_names):
                feature_name = model.feature_names[node.feature_index]
            else:
                feature_name = f"idx_{node.feature_index}"
            
            # Handle float vs int/string values for display
            if isinstance(node.threshold, float):
                thresh_str = f"{node.threshold:.2f}"
            else:
                thresh_str = str(node.threshold)
                
            text = f"{feature_name}\n<= {thresh_str}?"
            color = '#deebf7' # Light blue for internal nodes
            box_style = 'round4,pad=0.5'
            
        ax.text(x, y, text, ha='center', va='center', 
                bbox=dict(facecolor=color, edgecolor='#333333', boxstyle=box_style),
                fontsize=9)
        
        if not node.is_leaf:
            # Child coordinates
            left_width = get_width(node.left)
            right_width = get_width(node.right)
            
            # Calculate child positions
            # The logic is: split the available width for this node proportionally to children's width
            # Scale x offset by how deep we are to avoid overlap
            
            x_left = x - (right_width / (left_width + right_width)) * (width / 2)
            x_right = x + (left_width / (left_width + right_width)) * (width / 2)
            
            y_next = y - 1  # Move down by 1 unit
            
            # Draw edges
            ax.plot([x, x_left], [y, y_next], 'k-', lw=1, alpha=0.6)
            ax.plot([x, x_right], [y, y_next], 'k-', lw=1, alpha=0.6)
            
            # Add Yes/No labels
            mid_x_left = (x + x_left) / 2
            mid_y_left = (y + y_next) / 2
            ax.text(mid_x_left, mid_y_left, 'Yes', fontsize=8, color='green', ha='right')
            
            mid_x_right = (x + x_right) / 2
            mid_y_right = (y + y_next) / 2
            ax.text(mid_x_right, mid_y_right, 'No', fontsize=8, color='red', ha='left')
            
            # Recursive calls
            # Reduce width for children
            plot_node(node.left, x_left, y_next, width / 2, level + 1)
            plot_node(node.right, x_right, y_next, width / 2, level + 1)

    # Initial call
    plot_node(model.root, 0, 0, total_width, 0)
    
    # Adjust view limit
    ax.set_ylim(-depth - 0.5, 0.5)
    ax.set_xlim(-(total_width/2) - 1, (total_width/2) + 1)
    
    plt.title(f"Decision Tree Visualization (Max Depth: {model.get_depth()})")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Tree visualization saved to: {save_path}")
    else:
        plt.show()
