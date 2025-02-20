#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torchvision import datasets, transforms, models
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt


# In[ ]:


transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# In[3]:


train_dataset = datasets.ImageFolder(root="/home/student/Desktop/220962109/lab6/cats_and_dogs_filtered/train", transform=transform)
valid_dataset = datasets.ImageFolder(root="/home/student/Desktop/220962109/lab6/cats_and_dogs_filtered/validation", transform=transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)


# In[4]:


alexnet = models.alexnet(pretrained=True)

for param in alexnet.parameters():
    param.requires_grad = False

alexnet.classifier[6] = nn.Linear(in_features=4096, out_features=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alexnet = alexnet.to(device)


# In[5]:


criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(alexnet.classifier.parameters(), lr=0.001, momentum=0.9)


# In[7]:


epochs = 10

def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        
train_model(alexnet, train_loader, valid_loader, criterion, optimizer, epochs)


# In[ ]:




