"""
Queue - Custom queue implementation for BFS traversal.
"""


class QueueNode:
    
    def __init__(self, data):
        self.data = data
        self.next = None


class Queue:
    
    def __init__(self):
        self.front = None
        self.rear = None
        self._size = 0
    
    def enqueue(self, item):
        new_node = QueueNode(item)
        
        if self.rear is None:
            self.front = new_node
            self.rear = new_node
        else:
            self.rear.next = new_node
            self.rear = new_node
        
        self._size += 1
    
    def dequeue(self):
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        
        item = self.front.data
        self.front = self.front.next
        
        if self.front is None:
            self.rear = None
        
        self._size -= 1
        return item
    
    def peek(self):
        if self.is_empty():
            raise IndexError("peek from empty queue")
        return self.front.data
    
    def is_empty(self):
        return self.front is None
    
    def size(self):
        return self._size
    
    def clear(self):
        self.front = None
        self.rear = None
        self._size = 0
    
    def to_list(self):
        result = []
        node = self.front
        while node is not None:
            result.append(node.data)
            node = node.next
        return result
    
    def __len__(self):
        return self._size
    
    def __repr__(self):
        return f"Queue({self.to_list()})"
    
    def __iter__(self):
        node = self.front
        while node is not None:
            yield node.data
            node = node.next


class Stack:
    
    def __init__(self):
        self.top = None
        self._size = 0
    
    def push(self, item):
        new_node = QueueNode(item)
        new_node.next = self.top
        self.top = new_node
        self._size += 1
    
    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        
        item = self.top.data
        self.top = self.top.next
        self._size -= 1
        return item
    
    def peek(self):
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self.top.data
    
    def is_empty(self):
        return self.top is None
    
    def size(self):
        return self._size
    
    def __len__(self):
        return self._size
