import torch
import torch.nn as nn

# Create an instance of nn.Linear with input and output dimensions (2, 2)
w_k = nn.Linear(2, 2)

# Define the input tensor with the correct shape (batch_size, input_features)
k = torch.tensor([[[5.0, 9.0]]])  # Example input tensor with shape (1, 2)

# Apply the linear transformation to the input tensor
key = w_k(k)

# Print the transformed tensor
print(key)