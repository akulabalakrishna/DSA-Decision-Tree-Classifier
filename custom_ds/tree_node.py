"""
TreeNode - Custom node class for Decision Tree.
"""


class TreeNode:
    
    def __init__(self):
        self.is_leaf = False
        self.feature_index = None
        self.threshold = None
        self.category_set = None
        self.is_numerical = True
        self.left = None
        self.right = None
        self.prediction = None
        self.depth = 0
        self.n_samples = 0
        self.impurity = 0.0
        self.class_distribution = None
    
    def make_leaf(self, prediction, n_samples=0, impurity=0.0, class_distribution=None):
        self.is_leaf = True
        self.prediction = prediction
        self.n_samples = n_samples
        self.impurity = impurity
        self.class_distribution = class_distribution
        self.feature_index = None
        self.threshold = None
        self.category_set = None
        self.left = None
        self.right = None
    
    def make_split(self, feature_index, threshold=None, category_set=None, 
                   is_numerical=True, n_samples=0, impurity=0.0):
        self.is_leaf = False
        self.feature_index = feature_index
        self.is_numerical = is_numerical
        self.n_samples = n_samples
        self.impurity = impurity
        
        if is_numerical:
            self.threshold = threshold
            self.category_set = None
        else:
            self.threshold = None
            self.category_set = category_set
        
        self.left = TreeNode()
        self.right = TreeNode()
        self.left.depth = self.depth + 1
        self.right.depth = self.depth + 1
        
        return self.left, self.right
    
    def go_left(self, feature_value):
        if self.is_numerical:
            return feature_value <= self.threshold
        else:
            return feature_value in self.category_set
    
    def __repr__(self):
        if self.is_leaf:
            return f"LeafNode(prediction={self.prediction}, n_samples={self.n_samples})"
        else:
            if self.is_numerical:
                return f"SplitNode(feature={self.feature_index}, threshold={self.threshold})"
            else:
                return f"SplitNode(feature={self.feature_index}, categories={self.category_set})"
