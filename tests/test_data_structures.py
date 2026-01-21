import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_ds.hash_table import HashTable
from custom_ds.queue_ds import Queue
from custom_ds.tree_node import TreeNode

class TestHashTable(unittest.TestCase):
    def setUp(self):
        self.ht = HashTable(capacity=10)

    def test_put_get(self):
        self.ht.put("key1", "value1")
        self.assertEqual(self.ht.get("key1"), "value1")
        self.ht.put("key1", "value2") # Update
        self.assertEqual(self.ht.get("key1"), "value2")

    def test_collision_handling(self):
        # Force collision if we knew the hash function, but here we just test general robustness
        keys = [f"key{i}" for i in range(20)]
        for k in keys:
            self.ht.put(k, k)
        
        for k in keys:
            self.assertEqual(self.ht.get(k), k)

    def test_contains(self):
        self.ht.put("exists", True)
        self.assertTrue(self.ht.contains("exists"))
        self.assertFalse(self.ht.contains("does_not_exist"))

    def test_resize(self):
        initial_capacity = self.ht.capacity
        # Trigger resize
        for i in range(100):
            self.ht.put(f"k{i}", i)
        
        self.assertTrue(self.ht.capacity > initial_capacity)
        self.assertEqual(self.ht.size, 100)
        self.assertEqual(self.ht.get("k50"), 50)

class TestQueue(unittest.TestCase):
    def setUp(self):
        self.queue = Queue()

    def test_enqueue_dequeue(self):
        self.queue.enqueue(1)
        self.queue.enqueue(2)
        self.assertEqual(self.queue.dequeue(), 1)
        self.assertEqual(self.queue.dequeue(), 2)
        self.assertTrue(self.queue.is_empty())

    def test_peek(self):
        self.queue.enqueue("a")
        self.assertEqual(self.queue.peek(), "a")
        self.queue.dequeue()
        with self.assertRaises(IndexError):
            self.queue.peek()

class TestTreeNode(unittest.TestCase):
    def test_split_logic(self):
        node = TreeNode()
        # Test split creation logic if applicable
        # The TreeNode in this project seems to handle 'make_split' which returns children
        left, right = node.make_split(feature_index=0, threshold=0.5, category_set=None, is_numerical=True, n_samples=10, impurity=0.5)
        
        self.assertIsNotNone(node.left)
        self.assertIsNotNone(node.right)
        self.assertEqual(node.left, left)
        self.assertEqual(node.right, right)
        self.assertEqual(node.threshold, 0.5)
        self.assertTrue(node.is_numerical)

if __name__ == '__main__':
    unittest.main()
