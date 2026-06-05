import numpy as np

class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._children = children

    def tanh(self):
        t = np.tanh(self.data)
        out = Value(t, (self,), op='tanh')
        def _backward():
            self.grad +=(1-t**2) * out.grad
       
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), op='*')

        def _backward():
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
            other.grad -=out.grad
        out._backward = _backward
        return out
   
    def __pow__(self,):
        out = Value(self.data ** other, (self,), op='**')
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        out = Value(self.data / other.data, (self, other), op='/')

        def _backward():
            self.grad += 1 / other.data * out.grad
            other.grad -= self.data / (other.data ** 2) * out.grad

        out._backward = _backward
        return out
   
    def backward(self):
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    build_topo(child)
                topo.append(node)
   
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

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