class Value:
    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 1.0
        self._prev = set(_children)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data:{self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data)

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data)

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

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
c = a + b
c.backward()
print(a.grad, b.grad)
