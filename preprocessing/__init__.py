"""
Preprocessing Module
Contains data loading, cleaning, and encoding utilities.
"""

from .data_loader import load_adult_data, load_data_file
from .cleaner import clean_data, handle_missing_values
from .encoder import DataEncoder

__all__ = [
    'load_adult_data',
    'load_data_file',
    'clean_data',
    'handle_missing_values',
    'DataEncoder'
]
