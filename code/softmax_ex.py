import torch

random_tensor = torch.rand(4)
print(f"random tensor: {random_tensor}")

# calculate softmax

# understanding PyTorch's torch.nn.Softmax
# Softmax is a class, which needs to be initialized with parameter for dimensionality
# once the class is initialized, it can be used like a function because of __call__() method

softmax_tensor = torch.nn.Softmax(dim=0)(random_tensor)
print(f"softmax tensor: {softmax_tensor}")
