"""
Benchmark - Optional comparison with sklearn Decision Tree.
"""


def compare_with_sklearn(X_train, y_train, X_test, y_test, custom_predictions, feature_types):
    try:
        from sklearn.tree import DecisionTreeClassifier as SklearnDT
        from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
        import numpy as np
    except ImportError:
        print("sklearn not available for benchmarking")
        return None
    
    print("\n" + "=" * 60)
    print("SKLEARN BENCHMARK COMPARISON")
    print("(For validation only - custom tree is fully independent)")
    print("=" * 60)
    
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    X_train_encoded = []
    X_test_encoded = []
    encoders = {}
    
    for i in range(len(feature_types)):
        if feature_types[i]:
            train_col = [float(row[i]) for row in X_train]
            test_col = [float(row[i]) for row in X_test]
        else:
            enc = LabelEncoder()
            train_col = [row[i] for row in X_train]
            test_col = [row[i] for row in X_test]
            
            all_vals = train_col + test_col
            enc.fit(all_vals)
            
            train_col = enc.transform(train_col).tolist()
            test_col = enc.transform(test_col).tolist()
            encoders[i] = enc
        
        X_train_encoded.append(train_col)
        X_test_encoded.append(test_col)
    
    X_train_np = np.array(X_train_encoded).T
    X_test_np = np.array(X_test_encoded).T
    
    sklearn_tree = SklearnDT(max_depth=10, min_samples_split=2)
    sklearn_tree.fit(X_train_np, y_train_encoded)
    
    sklearn_pred_encoded = sklearn_tree.predict(X_test_np)
    sklearn_predictions = le.inverse_transform(sklearn_pred_encoded)
    
    custom_correct = sum(1 for t, p in zip(y_test, custom_predictions) if t == p)
    custom_accuracy = custom_correct / len(y_test)
    
    sklearn_correct = sum(1 for t, p in zip(y_test, sklearn_predictions) if t == p)
    sklearn_accuracy = sklearn_correct / len(y_test)
    
    agreement = sum(1 for c, s in zip(custom_predictions, sklearn_predictions) if c == s)
    agreement_rate = agreement / len(y_test)
    
    print(f"\nCustom Decision Tree Accuracy:  {custom_accuracy * 100:.2f}%")
    print(f"Sklearn Decision Tree Accuracy: {sklearn_accuracy * 100:.2f}%")
    print(f"Agreement Rate:                 {agreement_rate * 100:.2f}%")
    print(f"Sklearn Tree Depth:             {sklearn_tree.get_depth()}")
    print("=" * 60)
    
    return {
        'custom_accuracy': custom_accuracy,
        'sklearn_accuracy': sklearn_accuracy,
        'agreement_rate': agreement_rate,
        'sklearn_depth': sklearn_tree.get_depth()
    }
