import torch
import numpy as np

def f(x):
    return np.exp((-x ** 2) - 2 * x - np.sin(x))


def torchf(x):
    return torch.exp((-x ** 2) - 2 * x - torch.sin(x))


def fdiff(x):
    return f(x) * ((-2 * x) - 2 - np.cos(x))


inp_x = 2.0
x = torch.tensor(inp_x, requires_grad=True)
y = torchf(x)

y.backward()

print("Pytorch Solution:", x.grad)
print("Analytical Solution:", fdiff(inp_x))