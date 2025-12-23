"""
Data Loader - Load and parse the Adult Income dataset.
"""

import os


ADULT_FEATURE_NAMES = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
]

ADULT_FEATURE_TYPES = [
    True,   # age
    False,  # workclass
    True,   # fnlwgt
    False,  # education
    True,   # education-num
    False,  # marital-status
    False,  # occupation
    False,  # relationship
    False,  # race
    False,  # sex
    True,   # capital-gain
    True,   # capital-loss
    True,   # hours-per-week
    False   # native-country
]


def parse_csv_line(line):
    values = []
    current = ""
    in_quotes = False
    
    for char in line:
        if char == '"':
            in_quotes = not in_quotes
        elif char == ',' and not in_quotes:
            values.append(current.strip().strip('"'))
            current = ""
        else:
            current += char
    
    values.append(current.strip().strip('"'))
    
    return values


def load_csv_file(filepath, has_header=True):
    data = []
    labels = []
    feature_names = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = True
        for line in f:
            line = line.strip()
            
            if not line:
                continue
            
            parts = parse_csv_line(line)
            
            if first_line and has_header:
                feature_names = parts[:-1]
                first_line = False
                continue
            
            first_line = False
            
            if len(parts) < 15:
                continue
            
            features = parts[:14]
            target = parts[14].strip().rstrip('.')
            
            data.append(features)
            labels.append(target)
    
    return data, labels, feature_names


def load_data_file(filepath):
    data = []
    labels = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line or line.startswith('|'):
                continue
            
            parts = line.split(',')
            
            if len(parts) < 15:
                continue
            
            cleaned_parts = [p.strip() for p in parts]
            features = cleaned_parts[:14]
            target = cleaned_parts[14].rstrip('.')
            
            data.append(features)
            labels.append(target)
    
    return data, labels


def load_adult_data(data_dir):
    csv_path = os.path.join(data_dir, 'adult.csv')
    train_path = os.path.join(data_dir, 'adult.data')
    test_path = os.path.join(data_dir, 'adult.test')
    
    if os.path.exists(csv_path):
        print(f"Found adult.csv - loading and splitting 80/20...")
        all_data, all_labels, csv_feature_names = load_csv_file(csv_path, has_header=True)
        print(f"Loaded {len(all_data)} total samples from adult.csv")
        
        feature_names = csv_feature_names if csv_feature_names else ADULT_FEATURE_NAMES
        
        train_data, train_labels, test_data, test_labels = train_test_split(
            all_data, all_labels, test_ratio=0.2, random_seed=42
        )
        
        print(f"Split into {len(train_data)} training and {len(test_data)} test samples")
        
        return {
            'train_data': train_data,
            'train_labels': train_labels,
            'test_data': test_data,
            'test_labels': test_labels,
            'feature_names': ADULT_FEATURE_NAMES,
            'feature_types': ADULT_FEATURE_TYPES
        }
    
    train_data, train_labels = [], []
    test_data, test_labels = [], []
    
    if os.path.exists(train_path):
        train_data, train_labels = load_data_file(train_path)
        print(f"Loaded {len(train_data)} training samples from adult.data")
    else:
        print(f"Warning: {train_path} not found")
    
    if os.path.exists(test_path):
        test_data, test_labels = load_data_file(test_path)
        print(f"Loaded {len(test_data)} test samples from adult.test")
    else:
        print(f"Warning: {test_path} not found")
    
    if len(train_data) == 0 and len(test_data) == 0:
        raise FileNotFoundError(
            f"No dataset files found in {data_dir}. "
            f"Please add adult.csv OR (adult.data and adult.test)"
        )
    
    return {
        'train_data': train_data,
        'train_labels': train_labels,
        'test_data': test_data,
        'test_labels': test_labels,
        'feature_names': ADULT_FEATURE_NAMES,
        'feature_types': ADULT_FEATURE_TYPES
    }


def merge_datasets(data1, labels1, data2, labels2):
    merged_data = data1 + data2
    merged_labels = labels1 + labels2
    return merged_data, merged_labels


def train_test_split(data, labels, test_ratio=0.2, random_seed=42):
    n = len(data)
    
    indices = list(range(n))
    
    a, c, m = 1103515245, 12345, 2**31
    state = random_seed
    
    for i in range(n - 1, 0, -1):
        state = (a * state + c) % m
        j = state % (i + 1)
        indices[i], indices[j] = indices[j], indices[i]
    
    split_idx = int(n * (1 - test_ratio))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_data = [data[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    return train_data, train_labels, test_data, test_labels


def get_dataset_info(data, labels, feature_names=None):
    n_samples = len(data)
    n_features = len(data[0]) if n_samples > 0 else 0
    
    print(f"\nDataset Information:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    
    from custom_ds.hash_table import count_items
    class_counts = count_items(labels)
    
    print(f"  Classes: {len(class_counts.keys())}")
    print(f"  Class distribution:")
    for label, count in class_counts.items():
        pct = 100 * count / n_samples if n_samples > 0 else 0
        print(f"    {label}: {count} ({pct:.1f}%)")
