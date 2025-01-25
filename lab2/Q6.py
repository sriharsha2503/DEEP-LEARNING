import torch
import numpy as np

def torchf(x,y,z):
    return torch.tanh_(torch.log(1+(z*2*x)/torch.sin(y)))

def f(x,y,z):
    return np.tanh(np.log(1+(z*2*x)/np.sin(y)))

def fdiff(x,y,z):
    a = 2*x
    b = a*z
    c = np.sin(y)
    e = b/c
    f = 1 + e
    g = np.log(f)
    h = np.tanh(g)

    dhdg = 1 - h**2
    dhdf = dhdg*1/f
    dhde = dhdf*1
    dhdc = dhde*-1/(c**2)*b
    dhdb = dhde/c
    dhdy = dhdc*np.cos(x)
    dhdz = dhdb*a
    dhda = dhdb*z
    dhdx = dhda*2

    return(dhdx,dhdy,dhdz)



inp_x,inp_y,inp_z = 2.0,2.0,2.0
x = torch.tensor(inp_x,requires_grad=True)
y = torch.tensor(inp_y,requires_grad=True)
z = torch.tensor(inp_z,requires_grad=True)

fxyz = torchf(x,y,z)
fxyz.backward()

print(x.grad,y.grad,z.grad)
print(fdiff(inp_x,inp_y,inp_z))