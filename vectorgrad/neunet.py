import random

from engine import Value


class Neuron:
    def __init__(self, nin):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w * x + b
        # w = weight, x = input , b = bias
        act = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        out = act.tanh()
        return out

    def parameters(self):
        return self.weights + [self.bias]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        out = []
        for n in self.neurons:
            out.extend(n.parameters())
        return out


class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def parameters(self):
        out = []
        for l in self.layers:
            out.extend(l.parameters())
        return out


# Neural Net with 3 inputs, two hidden layers each having 4 nodes and a single output node
x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
# print(n(x))


xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

for _ in range(20):
    ypred = [n(x) for x in xs]

    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))
    loss.backward()

    for p in n.parameters():
        p.data += -0.01 * p.grad

    print("Loss: ", loss)
    print("ypred: ", ypred)
