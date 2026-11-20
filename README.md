# vectorgrad

A small automatic differentiation engine written from scratch, based on Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). It starts with scalars and is being extended to NumPy-backed tensors.

## Usage

```python
from vectorgrad.engine import Value
from vectorgrad.neunet import MLP

# gradients of a small expression
a = Value(2.0)
b = Value(-3.0)
c = (a * b + 1).tanh()
c.backward()
print(a.grad, b.grad)

# a network with 3 inputs, two hidden layers of 4 neurons, and 1 output
model = MLP(3, [4, 4, 1])
out = model([2.0, 3.0, -1.0])
```

## Install and test

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The tests build the same computation in vectorgrad and PyTorch and check that the values and gradients match.

