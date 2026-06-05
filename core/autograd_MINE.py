import numpy as np

np.random.seed(42)

def sigmoid(z):
    return 1/(1.0+np.exp(-z))

class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._children = children 
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), op='*')

        def _backward():
            # ∂f/∂a​
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), op='+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out
    def __sub__(self, other):
        out = Value(self.data - other.data, (self, other), op='-')

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad

        out._backward = _backward
        return out

class Neuron:
    def __init__(self, nin):
        self.w = [Value(np.random.uniform(-1, 1)) for neuron in range(nin)]
        self.b = Value(np.random.uniform(-1, 1))