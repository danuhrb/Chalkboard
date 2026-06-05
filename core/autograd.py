# Python implementation of autograd first, cpp later
# NO LIBRARIES OTHER THAN NUMPY
import numpy as np

def sigmoid(x):
    return 1.0/ (1.0 + np.exp(-x))

class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._children = children 
        
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), op='*')

        def _backward():
            # partial of a * b w.r.t. a is b, and vice versa
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
    def __pow__(self, other):
        out = Value(self.data ** other.data, (self, other), op='**')

        def _backward():
            self.grad += other.data * (self.data ** (other.data - 1)) * out.grad
            other.grad += self.data ** other.data * np.log(self.data) * out.grad

        out._backward = _backward
        return out
    def __truediv__(self, other):
        out = Value(self.data / other.data, (self, other), op='/')

        def _backward():
            self.grad += 1 / other.data * out.grad
            other.grad -= self.data / (other.data ** 2) * out.grad

        out._backward = _backward
        return out

    

class Neuron:
    def __init__(self, nin):
        self.w = [Value(np.random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(np.random.uniform(-1, 1))

    def __call__(self, x):
        return sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]