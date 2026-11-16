import pytest
import torch
import torch.nn as nn

from vectorgrad.neunet import MLP

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]


def test_neunet_against_torch():

    # neunet
    net = MLP(3, [4, 4, 1])
    y_pred_nn = [net(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, y_pred_nn))
    print(loss)
    loss.backward()

    # pytorch implementation
    model = nn.Sequential(
        nn.Linear(3, 4),
        nn.Tanh(),
        nn.Linear(4, 4),
        nn.Tanh(),
        nn.Linear(4, 1),
        nn.Tanh(),
    ).double()
    linears = [m for m in model if isinstance(m, nn.Linear)]

    # copying weights
    with torch.no_grad():
        for layer, lin in zip(net.layers, linears):
            for i, neuron in enumerate(layer.neurons):
                lin.bias[i] = neuron.bias.data
                for j, w in enumerate(neuron.weights):
                    lin.weight[i, j] = w.data

    # pytorch forward + backward
    out = model(torch.tensor(xs, dtype=torch.double))  # shape (4, 1)
    out = out.squeeze(1)  # shape (4,)
    loss_pt = ((out - torch.tensor(ys, dtype=torch.double)) ** 2).sum()
    loss_pt.backward()

    # comparision
    assert loss.data == pytest.approx(loss_pt.item())
    for layer, lin in zip(net.layers, linears):
        for i, neuron in enumerate(layer.neurons):
            assert neuron.bias.grad == pytest.approx(lin.bias.grad[i].item())
            for j, w in enumerate(neuron.weights):
                assert w.grad == pytest.approx(lin.weight.grad[i, j].item())
