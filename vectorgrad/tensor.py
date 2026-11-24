import numpy as np


class Tensor:
    __array_ufunc__ = None

    def __init__(self, data, _children=()):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data: {self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor((self.data + other.data), (self, other))

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + other * -1

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, other):
        # Only taking ints or floats for now
        assert isinstance(other, (int, float))

        out = Tensor((self.data**other), (self,))

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor((self.data * other.data), (self, other))

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert self.data.ndim == 2 and other.data.ndim == 2
        out = Tensor((self.data @ other.data), (self, other))

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,))

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), (self,))

        def _backward():
            self.grad += np.where(self.data > 0, 1, 0) * out.grad

        out._backward = _backward
        return out

    def mean(self, axis=None):
        assert isinstance(axis, (type(None), int))
        n = np.size(self.data, axis=axis)
        out = self.sum(axis=axis) / n

        return out

    def backward(self):
        topo = []
        visited = set()

        # Recursive function to build a topological tree of the expression ( from left to right )
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones(self.data.shape)
        for node in reversed(topo):
            node._backward()

    def sum(self, axis=None):
        assert isinstance(axis, (type(None), int))
        out = Tensor(np.sum(self.data, axis=axis), (self,))

        def _backward():
            grad = out.grad

            if axis is not None:
                grad = np.expand_dims(grad, axis=axis)

            self.grad += np.ones_like(self.data) * grad

        out._backward = _backward
        return out


def _unbroadcast(grad, original_shape):
    # Remove extra dimensions added during broadcasting
    while grad.ndim > len(original_shape):
        grad = grad.sum(axis=0)

    # Sum dimensions that were originally size 1
    for axis, size in enumerate(original_shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad
