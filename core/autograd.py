# Python implementation of autograd first, cpp later
# NO LIBRARIES OTHER THAN NUMPY
import numpy as np

class Node:
    def __init__(self, value):
        self.value = value
        self.grad = 0
        self.bias = 0

        