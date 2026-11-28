import numpy as np

from .tensor import Tensor


class Layer:
    def __init__(self, nin, nout):
        rng = np.random.default_rng()
        self.bias = Tensor(rng.uniform(-1, 1, size=(1, nout)))
        self.weights = Tensor(rng.uniform(-1, 1, size=(nin, nout)))

    def __call__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = x @ self.weights + self.bias
        return out.tanh()

    def params(self):
        return [self.bias, self.weights]


class MLP:
    # eg : [3,[4,4,1]]
    def __init__(self, nin, nout):
        size = [nin] + nout
        self.layers = [Layer(size[i], size[i + 1]) for i in range(len(nout))]

    def __call__(self, x):
        for l in self.layers:
            x = l(x)

        return x

    def params(self):
        out = []
        for l in self.layers:
            out.extend(l.params())

        return out
