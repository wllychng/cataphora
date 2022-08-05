
import torch

# using torch.unsqueeze() to add back dimensionality
x = torch.tensor([1, 2, 3, 4])
print(f"size of x: {x.size()}")
print(f"tensor x: {x}")
x = torch.unsqueeze(x, 0)
print(f"size of x after unsqueeze: {x.size()}")
print(f"tensor x after unsqueeze: {x}")
