import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_ml.splitter import find_numerical_threshold

class TestSplitter(unittest.TestCase):
    def test_perfect_split(self):
        # Feature: [1, 2, 8, 9] with labels [0, 0, 1, 1]
        # Should split between 2 and 8
        data = [[1.0], [2.0], [8.0], [9.0]]
        labels = [0, 0, 1, 1]
        
        threshold, gain, (left, right) = find_numerical_threshold(data, labels, 0)
        
        self.assertIsNotNone(threshold)
        self.assertTrue(2.0 <= threshold <= 8.0)
        self.assertGreater(gain, 0.0)
        self.assertEqual(len(left), 2)
        self.assertEqual(len(right), 2)

    def test_no_split_possible(self):
        # Heterogeneous but all same feature value
        data = [[1.0], [1.0], [1.0], [1.0]]
        labels = [0, 0, 1, 1]
        
        threshold, gain, splits = find_numerical_threshold(data, labels, 0)
        # Should be None or gain <= 0
        if threshold is not None:
             self.assertTrue(gain <= 0 or len(splits[0])==0 or len(splits[1])==0)
        else:
             self.assertIsNone(threshold)

    def test_mixed_split(self):
        # [1, 2, 3, 4] -> [0, 1, 0, 1] ... terrible split but should run
        data = [[1.0], [2.0], [3.0], [4.0]]
        labels = [0, 1, 0, 1]
        
        threshold, gain, splits = find_numerical_threshold(data, labels, 0)
        
        self.assertIsNotNone(threshold)

if __name__ == '__main__':
    unittest.main()
