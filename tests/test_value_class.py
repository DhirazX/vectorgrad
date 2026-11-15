import pytest
import torch

from vectorgrad.engine import Value


def expr(a):
    b = a + a * 2
    c = b**2
    d = c - 1
    return c + d / 2


def tanh(a):
    return (a * 2 - 1).tanh()


def relu(a):
    return (a * 2 + 1).relu()


@pytest.mark.parametrize("activation_func", [tanh, relu])
@pytest.mark.parametrize(("x"), [-2.0, -0.3, 0.8, 1.5, -1.0, -0.4, 0.7, 1.2])
def test_expr_against_torch(x, activation_func):

    # Vectorgrad
    a = Value(x)
    b = activation_func(a)
    e = expr(b)
    e.backward()

    # PyTorch
    apt = torch.tensor(x, requires_grad=True, dtype=torch.double)
    bpt = activation_func(apt)
    ept = expr(bpt)
    ept.backward()

    assert e.data == pytest.approx(ept.item())
    assert a.grad == pytest.approx(apt.grad.item())
