#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn

# Load the dataset (Ensure to adjust the file path if necessary)
df = pd.read_csv("naturalgases.csv")  # Replace with the correct path
print(df.head())  # Inspect the first few rows

# Drop rows with missing values (if any)
df = df.dropna()

# Ensure 'Price' column exists and normalize the prices
y = df['Price'].values
x = np.arange(1, len(y) + 1, 1)

# Normalize the price range between 0 and 1
minm = y.min()
maxm = y.max()
y = (y - minm) / (maxm - minm)

# Sequence length (using the last 10 days to predict the 11th day)
Sequence_Length = 10
X = []
Y = []

for i in range(0, len(y) - Sequence_Length):
    list1 = []
    for j in range(i, i + Sequence_Length):
        list1.append(y[j])
    X.append(list1)
    Y.append(y[i + Sequence_Length])  # The target value is the price on the 11th day

# Convert lists to arrays
X = np.array(X)
Y = np.array(Y)

# Split the data into train and test sets (90% for training, 10% for testing)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42, shuffle=False)

# Create a custom Dataset for PyTorch
class NGTimeSeries(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len

# Create data loaders for training
dataset = NGTimeSeries(x_train, y_train)
train_loader = DataLoader(dataset, shuffle=True, batch_size=256)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=5, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        output, _status = self.lstm(x)
        output = output[:, -1, :]  # Get the last output
        output = self.fc1(torch.relu(output))
        return output

# Initialize the model, loss function, and optimizer
model = LSTMModel()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
epochs = 1500
for epoch in range(epochs):
    for i, data in enumerate(train_loader):
        # Reshape the input to (batch_size, sequence_length, input_size)
        y_pred = model(data[0].view(-1, Sequence_Length, 1)).reshape(-1)
        loss = criterion(y_pred, data[1])
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Iteration {i}: Loss = {loss.item()}")

# Testing the model
test_set = NGTimeSeries(x_test, y_test)
test_pred = model(test_set[:][0].view(-1, 10, 1)).view(-1)

# Plot the predicted and actual values
plt.plot(test_pred.detach().numpy(), label='Predicted')
plt.plot(test_set[:][1].view(-1), label='Actual')
plt.legend()
plt.show()

# Undo normalization
y_test_original = y_test * (maxm - minm) + minm
test_pred_original = test_pred.detach().numpy() * (maxm - minm) + minm

# Plot the original values
plt.plot(y_test_original)
plt.plot(range(len(y_test_original) - len(test_pred_original), len(y_test_original)), test_pred_original)
plt.show()


# In[2]:


import os
import string
import glob
import unicodedata
import torch
import random
import time
import math
import matplotlib.pyplot as plt


# In[3]:


# Download and extract the dataset
get_ipython().system('wget https://download.pytorch.org/tutorial/data.zip')
get_ipython().system('unzip -q data.zip')


# In[4]:


# List all language files
all_files = glob.glob('data/names/*.txt')
all_languages = [os.path.splitext(os.path.basename(f))[0] for f in all_files]

# Read file content
def find_files(path): return glob.glob(path)

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in all_letters
    )

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

category_lines = {}
all_categories = []

# Read lines from each file
for filename in find_files('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    with open(filename, encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
        category_lines[category] = [unicode_to_ascii(line) for line in lines]

n_categories = len(all_categories)


# In[5]:


# Letter to index and tensor
def letter_to_index(letter):
    return all_letters.find(letter)

def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor

def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor


# In[6]:


import torch.nn as nn

class NameLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NameLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor):
        lstm_out, _ = self.lstm(input_tensor)
        output = self.fc(lstm_out[-1])
        output = self.softmax(output)
        return output


# In[7]:


# Initialize model
hidden_size = 128
model = NameLSTM(n_letters, hidden_size, n_categories)

criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

# Random sample
def random_training_example():
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor

# Training function
def train(category_tensor, line_tensor):
    optimizer.zero_grad()
    output = model(line_tensor)
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()
    return output, loss.item()


# In[8]:


n_iters = 10000
print_every = 500
all_losses = []

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = random_training_example()
    output, loss = train(category_tensor, line_tensor)
    all_losses.append(loss)

    if iter % print_every == 0:
        guess = all_categories[output.topk(1)[1].item()]
        correct = '✓' if guess == category else f'✗ ({category})'
        print(f'{iter} {loss:.4f} {line} -> {guess} {correct}')


# In[9]:


def predict(input_line, n_predictions=3):
    print(f'\n> {input_line}')
    with torch.no_grad():
        output = model(line_to_tensor(input_line))

        topv, topi = output.topk(n_predictions, 1, True)
        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print(f'{all_categories[category_index]} ({value:.2f})')


# In[10]:


predict("Srinivasan")
predict("O'Connor")
predict("Yamamoto")


# In[11]:


import torch
import torch.nn as nn
import numpy as np
import random
import string
import unicodedata
import matplotlib.pyplot as plt


# In[12]:


import glob

# Define valid characters
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Convert Unicode to ASCII
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn' and c in all_letters)

# Load all names into one big string
all_names = []
for filename in glob.glob('data/names/*.txt'):
    with open(filename, encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
        all_names += [unicode_to_ascii(line) for line in lines]

text_data = ' '.join(all_names)
print("Sample data:", text_data[:100])


# In[13]:


# Mapping from char to index
char_to_idx = {ch: i for i, ch in enumerate(all_letters)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}

def char_tensor(char):
    tensor = torch.zeros(n_letters)
    tensor[char_to_idx[char]] = 1
    return tensor

def string_to_tensor(string):
    tensor = torch.zeros(len(string), 1, n_letters)
    for i, char in enumerate(string):
        tensor[i][0][char_to_idx[char]] = 1
    return tensor


# In[14]:


class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_seq, hidden=None):
        lstm_out, hidden = self.lstm(input_seq, hidden)
        output = self.fc(lstm_out[-1])
        return self.softmax(output), hidden


# In[15]:


def random_training_example(sequence_length=10):
    start_idx = random.randint(0, len(text_data) - sequence_length - 1)
    input_str = text_data[start_idx:start_idx+sequence_length]
    target_char = text_data[start_idx+sequence_length]
    return input_str, target_char

def train(model, optimizer, criterion, n_iters=5000):
    model.train()
    for iter in range(n_iters):
        input_str, target_char = random_training_example()
        input_tensor = string_to_tensor(input_str)
        target_tensor = torch.tensor([char_to_idx[target_char]])

        optimizer.zero_grad()
        output, _ = model(input_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()

        if iter % 500 == 0:
            print(f"[{iter}] Loss: {loss.item():.4f} | Input: '{input_str}' -> Target: '{target_char}'")


# In[16]:


def predict(model, input_str):
    model.eval()
    with torch.no_grad():
        input_tensor = string_to_tensor(input_str)
        output, _ = model(input_tensor)
        topv, topi = output.topk(1)
        predicted_char = idx_to_char[topi.item()]
        print(f"Input: '{input_str}' -> Predicted next char: '{predicted_char}'")
        return predicted_char


# In[17]:


# Initialize model
hidden_size = 128
model = CharLSTM(n_letters, hidden_size, n_letters)

# Loss and optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

# Train
train(model, optimizer, criterion, n_iters=5000)

# Predict a few samples
predict(model, "Ramanatha")
predict(model, "Alberti")
predict(model, "Krzyszto")


# In[ ]:




