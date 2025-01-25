import torch
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
x = 2 * a + 3 * b
y = 5 * a**2 + 3 * b**3
z = 2 * x + 3 * y
z.backward()
print(f"The PyTorch gradient dz/da is: {a.grad.item()}")
analytical_grad = 4 + 30 * a.item()
print(f"The analytical gradient dz/da is: {analytical_grad}")