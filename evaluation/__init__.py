"""
Evaluation Module
Contains metrics and benchmarking utilities.
"""

from .metrics import accuracy, confusion_matrix, classification_report
from .benchmark import compare_with_sklearn

__all__ = [
    'accuracy',
    'confusion_matrix', 
    'classification_report',
    'compare_with_sklearn'
]
