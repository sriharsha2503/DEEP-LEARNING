#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset


# In[11]:


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset = torchvision.datasets.MNIST(root='/home/student/Desktop/220962109/lab4/data',
                                       train=True, download=False, transform=transform)
testset = torchvision.datasets.MNIST(root='/home/student/Desktop/220962109/lab4/data',
                                     train=False, download=False, transform=transform)

trainloader = DataLoader(trainset, batch_size=64,  shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)


# In[12]:


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3),
                                 nn.ReLU(),
                                 nn.MaxPool2d((2,2), stride=2),
                                 nn.Conv2d(64, 128, kernel_size=3),
                                 nn.ReLU(),
                                 nn.MaxPool2d((2,2), stride=2),
                                 nn.Conv2d(128, 64, kernel_size=3),
                                 nn.ReLU(),
                                 nn.MaxPool2d((2,2), stride=2)
        )
        self.classification_head = nn.Sequential(nn.Linear(64, 20, bias=True),
                                                 nn.ReLU(),
                                                 nn.Linear(20,10,bias=True)
        )
        
    def forward(self, x):
        features = self.net(x)
        return self.classification_head(features.view(features.size(0),-1))  


# In[16]:


model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

num_epochs = 5

for epoch in range(num_epochs):
    model.train() 
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in trainloader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(trainloader)
    epoch_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")


# In[17]:


print("Model's state_dict")
for param_tensor in model.state_dict().keys():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
print()

print("Optimizer's state_dict")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])


# In[18]:


torch.save(model, "./MNISTParams/model.pt")


# In[19]:


mnist_testset = torchvision.datasets.FashionMNIST(root="./data", train=False,
                                                  download=True, transform=transform)
test_loader = DataLoader(mnist_testset, batch_size=64, shuffle=False)


# In[20]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[21]:


model = CNNClassifier()
model = torch.load("./MNISTParams/model.pt")
model.to(device)


# In[22]:


print("Model's state_dict:")
for param_tensor in model.state_dict().keys():
    print(param_tensor, "\t",model.state_dict()[param_tensor].size())
print()


# In[23]:


model.eval()
correct = 0
total = 0
for i, vdata in enumerate(test_loader):
    tinputs, tlabels = vdata
    tinputs = tinputs.to(device)
    tlabels = tlabels.to(device)
    toutputs = model(tinputs)
    
    _, predicted = torch.max(toutputs, 1)
    #print("True label:{}".format(tlabels))
    #print('Predicted: {}'.format(predicted))
    
    total += tlabels.size(0)
    
    correct += (predicted == tlabels).sum()
accuracy = 100.0 * correct / total
print("The overall accuracy is {}".format(accuracy))


# In[ ]:




