import numpy as np
import pytest
import torch

from vectorgrad.tensor import Tensor


@pytest.mark.parametrize(
    "a_data",
    [
        [[1.0, 2.0], [3.0, 4.0]],
        [[2.0, -1.0], [0.5, 3.0]],
        [[1.5, 2.5], [-2.0, 4.0]],
    ],
)
@pytest.mark.parametrize(
    "b_data",
    [
        [[5.0, 6.0], [7.0, 8.0]],
        [[1.0, -2.0], [3.0, 4.0]],
    ],
)
def test_matmul_against_torch(a_data, b_data):

    # Vectorgrad
    a = Tensor(a_data)
    b = Tensor(b_data)

    c = a @ b
    e = c.sum()
    e.backward()

    # PyTorch
    apt = torch.tensor(a_data, requires_grad=True, dtype=torch.double)
    bpt = torch.tensor(b_data, requires_grad=True, dtype=torch.double)

    ept = (apt @ bpt).sum()
    ept.backward()

    assert e.data == pytest.approx(ept.item())
    assert a.grad == pytest.approx(apt.grad.numpy())
    assert b.grad == pytest.approx(bpt.grad.numpy())


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_sum_against_torch(axis):

    data = [[1.0, 2.0], [3.0, 4.0]]

    # Vectorgrad
    a = Tensor(data)
    e = a.sum(axis=axis)
    if axis is not None:
        e = e.sum()

    e.backward()

    # PyTorch
    apt = torch.tensor(data, requires_grad=True, dtype=torch.double)
    ept = apt.sum(dim=axis)
    if axis is not None:
        ept = ept.sum()

    ept.backward()

    assert e.data == pytest.approx(ept.item())
    assert a.grad == pytest.approx(apt.grad.numpy())
