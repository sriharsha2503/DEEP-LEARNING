import torch

def f(x):
    return 8*(x**4) + 3*(x**3) + 7*(x**2) + 6*x + 3

def fdiff(x):
    return 32*(x**3) + 9*(x**2) + 14*x + 6

inp_x = 2.0
x = torch.tensor(inp_x,requires_grad = True)
y = f(x)

y.backward()

print("Pytorch Solution",x.grad)
print("Analytical Solution",fdiff(inp_x))