import numpy as np


class Tensor:
    def __init__(self, data, _children=()):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor((self.data + other.data), (self, other))

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor((self.data * other.data), (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out


a = Tensor([2, 3])
b = Tensor([4, 5])
c = a + b
print("Out: ", c.data)
c.grad = np.array([1.0, 1.0])
c._backward()
print("a.grad :", a.grad)
