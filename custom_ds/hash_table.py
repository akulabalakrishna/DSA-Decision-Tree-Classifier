"""
HashTable - Custom hash table implementation using chaining.
"""


class HashNode:
    
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None


class HashTable:
    
    def __init__(self, capacity=101):
        self.capacity = capacity
        self.buckets = [None] * capacity
        self.size = 0
    
    def _hash(self, key):
        return abs(hash(key)) % self.capacity
    
    def put(self, key, value):
        index = self._hash(key)
        node = self.buckets[index]
        
        while node is not None:
            if node.key == key:
                node.value = value
                return
            node = node.next
        
        new_node = HashNode(key, value)
        new_node.next = self.buckets[index]
        self.buckets[index] = new_node
        self.size += 1
        
        if self.size / self.capacity > 0.75:
            self._resize()
    
    def get(self, key, default=None):
        index = self._hash(key)
        node = self.buckets[index]
        
        while node is not None:
            if node.key == key:
                return node.value
            node = node.next
        
        return default
    
    def contains(self, key):
        return self.get(key, default=None) is not None or self._key_exists(key)
    
    def _key_exists(self, key):
        index = self._hash(key)
        node = self.buckets[index]
        
        while node is not None:
            if node.key == key:
                return True
            node = node.next
        return False
    
    def increment(self, key, amount=1):
        current = self.get(key, 0)
        self.put(key, current + amount)
    
    def delete(self, key):
        index = self._hash(key)
        node = self.buckets[index]
        prev = None
        
        while node is not None:
            if node.key == key:
                if prev is None:
                    self.buckets[index] = node.next
                else:
                    prev.next = node.next
                self.size -= 1
                return True
            prev = node
            node = node.next
        
        return False
    
    def keys(self):
        result = []
        for bucket in self.buckets:
            node = bucket
            while node is not None:
                result.append(node.key)
                node = node.next
        return result
    
    def values(self):
        result = []
        for bucket in self.buckets:
            node = bucket
            while node is not None:
                result.append(node.value)
                node = node.next
        return result
    
    def items(self):
        result = []
        for bucket in self.buckets:
            node = bucket
            while node is not None:
                result.append((node.key, node.value))
                node = node.next
        return result
    
    def _resize(self):
        old_buckets = self.buckets
        self.capacity = self.capacity * 2 + 1
        self.buckets = [None] * self.capacity
        self.size = 0
        
        for bucket in old_buckets:
            node = bucket
            while node is not None:
                self.put(node.key, node.value)
                node = node.next
    
    def __len__(self):
        return self.size
    
    def __repr__(self):
        items = self.items()
        return f"HashTable({dict(items)})"
    
    def clear(self):
        self.buckets = [None] * self.capacity
        self.size = 0


def count_items(items):
    counts = HashTable()
    for item in items:
        counts.increment(item)
    return counts
