import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_ml.decision_tree import DecisionTreeClassifier

class TestDecisionTree(unittest.TestCase):
    def test_simple_split(self):
        # Easy case: x < 5 is 0, x >= 5 is 1
        X = [[1], [2], [3], [6], [7], [8]]
        y = [0, 0, 0, 1, 1, 1]
        
        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X, y)
        preds = clf.predict(X)
        
        self.assertEqual(preds, y)

    def test_xor(self):
        # XOR Problem (requires splits on both axes)
        X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y = [0, 1, 1, 0]
        
        clf = DecisionTreeClassifier(max_depth=5)
        clf.fit(X, y)
        preds = clf.predict(X)
        self.assertEqual(preds, y)

if __name__ == '__main__':
    unittest.main()
