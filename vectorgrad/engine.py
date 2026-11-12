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

    def __rmul(self, other):
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
        out = Value(self.data**other, (self, other))

        def _backward():
            self.grad += other * (self.data**other - 1) * out.grad

        out.grad = _backward
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
            self.grad += 1 if self.data > 0 else 0

        out.grad = _backward
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


a = Value(2.0)
b = Value(-3.0)
c = a - b
print("c: ", c)
c.backward()
print(a.grad, b.grad)

x1 = Value(2.0)
x2 = Value(0.0)
w1 = Value(-3.0)
w2 = Value(1.0)
b = Value(6.8813735870195432)
x1w1 = x1 * w1
x2w2 = x2 * w2
x1w1x2w2 = x1w1 + x2w2
n = x1w1x2w2 + b
# n = (x1 * w1) + (x2 * w2) + b
o = n.tanh()


o.backward()
print("o: ", o)
print("n.grad: ", n.grad)
print("-------------------")
print("x1: ", x1)
print("x2: ", x2)
print("w1: ", w1)
print("w2: ", w2)
print("--------------------")
print("x1.grad: ", x1.grad)
print("x2.grad: ", x2.grad)
print("w1.grad: ", w1.grad)
print("w2.grad: ", w2.grad)
