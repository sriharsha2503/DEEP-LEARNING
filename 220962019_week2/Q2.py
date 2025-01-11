import torch
x = torch.tensor(2.0,requires_grad = True)
w = torch.tensor(3.0,requires_grad = True)
b = torch.tensor(1.5,requires_grad = True)

u = w*x
v = u + b
a=torch.relu(v)
a.backward()
print(f"Gradient is {w.grad.item()}")