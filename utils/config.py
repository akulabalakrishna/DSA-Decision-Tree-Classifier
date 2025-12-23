"""
Config - Hyperparameters and configuration settings.
"""

import os


class Config:
    
    MAX_DEPTH = 8
    MIN_SAMPLES_SPLIT = 20
    MIN_IMPURITY_DECREASE = 0.001
    
    TEST_SPLIT_RATIO = 0.2
    RANDOM_SEED = 42
    
    USE_PROVIDED_SPLIT = True
    
    DATA_DIR = "data"
    OUTPUT_DIR = "outputs"
    LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
    METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
    TREES_DIR = os.path.join(OUTPUT_DIR, "trees")
    
    RESULTS_FILE = "results.txt"
    TREE_FILE = "tree_structure.txt"
    
    RUN_SKLEARN_BENCHMARK = True
    
    @classmethod
    def get_hyperparameters(cls):
        return {
            'max_depth': cls.MAX_DEPTH,
            'min_samples_split': cls.MIN_SAMPLES_SPLIT,
            'min_impurity_decrease': cls.MIN_IMPURITY_DECREASE
        }
    
    @classmethod
    def print_config(cls):
        print("\n" + "=" * 50)
        print("CONFIGURATION")
        print("=" * 50)
        print("\nDecision Tree Hyperparameters:")
        print(f"  max_depth:             {cls.MAX_DEPTH}")
        print(f"  min_samples_split:     {cls.MIN_SAMPLES_SPLIT}")
        print(f"  min_impurity_decrease: {cls.MIN_IMPURITY_DECREASE}")
        print("\nData Settings:")
        print(f"  test_split_ratio:      {cls.TEST_SPLIT_RATIO}")
        print(f"  random_seed:           {cls.RANDOM_SEED}")
        print(f"  use_provided_split:    {cls.USE_PROVIDED_SPLIT}")
        print("\nPaths:")
        print(f"  data_dir:              {cls.DATA_DIR}")
        print(f"  output_dir:            {cls.OUTPUT_DIR}")
        print("=" * 50 + "\n")
    
    @classmethod
    def ensure_output_dirs(cls, base_path):
        dirs = [
            os.path.join(base_path, cls.OUTPUT_DIR),
            os.path.join(base_path, cls.LOGS_DIR),
            os.path.join(base_path, cls.METRICS_DIR),
            os.path.join(base_path, cls.TREES_DIR)
        ]
        
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"Created directory: {dir_path}")
