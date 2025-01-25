import torch
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def sigmoid_grad_manual(x):
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)

def manual(w, b, x):
    x = torch.tensor(x, dtype=torch.float32)
    w = torch.tensor(w, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)
    u = w * x
    v = u + b
    a = sigmoid(v)
    dadv = sigmoid_grad_manual(v)
    dadw = dadv * x
    dadb = dadv * 1
    return dadw, dadb

x = torch.tensor(5.0, dtype=torch.float32)
w = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
b = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)

u = w * x
v = u + b
a = sigmoid(v)

a.backward()

print("PyTorch Gradient w.r.t w:", w.grad.item())
print("PyTorch Gradient w.r.t b:", b.grad.item())

grad_w, grad_b = manual(w.item(), b.item(), x.item())
print("Manual Gradient w.r.t w:", grad_w.item())
print("Manual Gradient w.r.t b:", grad_b.item())

