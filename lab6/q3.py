#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
import os
from torch.utils.data import DataLoader, Dataset


# In[5]:


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

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


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


# In[11]:


model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

os.makedirs("./checkpoints", exist_ok=True)

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
    
    checkpoint = {
        "last_loss": epoch_loss,
        "last_accuracy": epoch_accuracy,  # Saving the accuracy
        "last_epoch": epoch + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }

    torch.save(checkpoint, "./checkpoints/checkpoint.pt")
    print(f'Checkpoint saved at epoch {epoch + 1}')

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")


# In[14]:


model = CNNClassifier().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

checkpoint = torch.load("./checkpoints/checkpoint.pt")

model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optimizer_state"])

last_loss = checkpoint["last_loss"]
last_accuracy = checkpoint["last_accuracy"]
last_epoch = checkpoint["last_epoch"]
print(f'Resuming training from epoch {last_epoch} with last loss {last_loss} and accuracy {last_accuracy}%')

NEW_EPOCHS = 5

for epoch in range(last_epoch, last_epoch + NEW_EPOCHS):
    print(f'EPOCH {epoch + 1}')

    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    for images, labels in trainloader:

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total_preds += labels.size(0)
        correct_preds += (predicted == labels).sum().item()

    avg_loss = running_loss / len(trainloader)
    accuracy = 100 * correct_preds / total_preds
    print(f'Average Loss: {avg_loss}, Accuracy: {accuracy}%')

    checkpoint = {
        "last_loss": avg_loss,
        "last_accuracy": accuracy,  
        "last_epoch": epoch + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }

    torch.save(checkpoint, "./checkpoints/checkpoint.pt")
    print(f'Checkpoint saved at epoch {epoch + 1}')


# In[ ]:




