#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Q1illustrate the functions for reshaping, veiwing,stacking,squeezing and unsqueezing of the tensors.
import torch

# Original tensor 
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
# Reshaping 
reshaped_tensor = tensor.reshape(3, 2)
print(reshaped_tensor)

tensor = torch.tensor([1, 2, 3, 4])
# Viewing 
viewed_tensor = tensor.view(2, 2)
print(viewed_tensor)

tensor1 = torch.tensor([1, 2])
tensor2 = torch.tensor([3, 4])
# Stacking 
stacked_tensor = torch.stack((tensor1, tensor2), dim=0)
print(stacked_tensor)

tensor = torch.tensor([[[1, 2], [3, 4]]])
# Squeezing 
squeezed_tensor = tensor.squeeze()
print(squeezed_tensor)

tensor = torch.tensor([1, 2, 3])
# Unsqueezing
unsqueezed_tensor = tensor.unsqueeze(0)
print(unsqueezed_tensor)


# In[12]:


#Q2illustrate the use of torch.permute()
tensor = torch.randn(3,1)
print("Original Tensor:")
print(tensor)
print("Shape:", tensor.shape)
permuted_tensor = tensor.permute(1,0)
print("\nPermuted Tensor:")
print(permuted_tensor)
print("Shape:", permuted_tensor.shape)


# In[4]:


#Q3illustrate indexing in tensors.

tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
element = tensor[1, 2]
print(f"Element at position (1, 2): {element}")


# In[5]:


# Slice rows 1 and 2 (indexing from 0) and columns 0 and 1
slice_tensor = tensor[1:3, 0:2]
print(f"Sliced Tensor (rows 1-2, columns 0-1):\n{slice_tensor}")


# In[6]:


# Use a list of indices to access multiple elements
indices = [0, 2]
extracted_elements = tensor[indices, 1]  # Extract elements from columns 1, at rows 0 and 2
print(f"Extracted Elements: {extracted_elements}")


# In[7]:


#Q4show how numpyarrays can be converted to tensors and back again to numpy arrays.
import numpy as np
numpy_array = np.array([1, 2, 3, 4, 5])
# Convert NumPy array to PyTorch tensor
tensor = torch.tensor(numpy_array)
print("PyTorch Tensor:", tensor)
# Convert PyTorch tensor to a NumPy array
numpy_array_back = tensor.numpy()
print("Converted back to NumPy Array:", numpy_array_back)


# In[8]:


#Q5create a random tensor with shape(7,7)
random_tensor = torch.randint(0, 10, (7, 7))
random_tensor


# In[16]:


#Q6perform a matrix multiplication on the tensor from 2(above problem) with another random tensor with shape (1,7) (hint :you may have to transpose the second tensor)
random_tensor = torch.randint(0, 10, (7, 7))
print("Random Tensor (7x7):")
print(random_tensor)
second_tensor = torch.randint(0, 10, (1, 7))
print("\nSecond Tensor (1x7):")
print(second_tensor)
second_tensor_transposed = second_tensor.T
print("\nTransposed Second Tensor (7x1):")
print(second_tensor_transposed)
result = torch.matmul(random_tensor, second_tensor_transposed)
print("\nResult of Matrix Multiplication (7x1):")
print(result)


# In[15]:


#Q7create two random tensors of shape (2,3) and send them both to the gpu(you'll need access to a gpu for this)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor_a = torch.randn(2, 3)
tensor_b = torch.randn(2, 3)
tensor_a = tensor_a.to(device)
tensor_b = tensor_b.to(device)
print("Tensor A on device:", tensor_a.device)
print(tensor_a)
print("\nTensor B on device:", tensor_b.device)
print(tensor_b)


# In[17]:


#Q8 perform a matrix multiplication on the tensors you created in 7 (again you may have to adjust the shape of the one of the tensors).
tensor_b_transposed = tensor_b.T  
result = torch.matmul(tensor_a, tensor_b_transposed)
print("\nMatrix Multiplication Result (2, 2):")
print(result)


# In[18]:


#Q9find the max and min values of the output of 8
max_value = torch.max(result)
min_value = torch.min(result)
print("\nMatrix Multiplication Result (2, 2):")
print(result)
print("\nMaximum value in the result:", max_value)
print("Minimum value in the result:", min_value)


# In[19]:


#Q10find the max and min index values of the output of 8
max_index = torch.argmax(result)
min_index = torch.argmin(result)
print("\nMatrix Multiplication Result (2, 2):")
print(result)
print("\nIndex of Maximum value:", max_index)
print("Index of Minimum value:", min_index)


# In[20]:


#Q11make a random tensor with shape (1,1,1,10) then create a new tensor with all the 1 dimensions removed to be left with a tensor of shape (10).set the seed to 7 when you create it and print out the first tensor and its shape as well as the sencond tensor and its shape.
torch.manual_seed(7)
tensor1 = torch.randn(1, 1, 1, 10)
tensor2 = tensor1.squeeze()
print("First Tensor (shape 1, 1, 1, 10):")
print(tensor1)
print("Shape of the first tensor:", tensor1.shape)
print("\nSecond Tensor (after squeeze, shape 10):")
print(tensor2)
print("Shape of the second tensor:", tensor2.shape)


# In[ ]:




