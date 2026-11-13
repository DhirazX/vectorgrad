import math


class Value:
    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data:{self.data})"

    # -----------------------

    # Addition Func
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    # ----------------------

    # Multiplication Functions
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    # --------------------------

    # Subtraction Functions
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    # -----------------------

    # Division Functions
    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    # ----------------------------

    # Power Function
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Unsupported type use int/ float"
        out = Value(self.data**other, (self,))

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    # ------------------------------

    def tanh(self):
        t = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = Value(t, (self,))

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    # ------------------------------

    def relu(self):
        out = Value(max(self.data, 0), (self,))

        def _backward():
            self.grad += (1 if self.data > 0 else 0) * out.grad

        out._backward = _backward
        return out

    # ------------------------------

    # Backward func which visits all previous nodes and calculates gradient for each node
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

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
