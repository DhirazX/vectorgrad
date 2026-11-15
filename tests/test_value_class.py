import pytest
import torch

from vectorgrad.engine import Value


def expr(a):
    b = a + a * 2
    c = b**2
    d = c - 1
    return c + d / 2


@pytest.mark.parametrize(("x"), [2.0, -2.0, 0.0, 0.5, 2.3])
def test_expr_against_torch(x):

    # Vectorgrad
    a = Value(x)
    e = expr(a)
    e.backward()

    # PyTorch
    apt = torch.tensor(x, requires_grad=True, dtype=torch.double)
    ept = expr(apt)
    ept.backward()

    assert e.data == pytest.approx(ept.item())
    assert a.grad == pytest.approx(apt.grad.item())
