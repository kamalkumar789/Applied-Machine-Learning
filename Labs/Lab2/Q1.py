
import torch


# 1
array = torch.rand((3,3))
print(array)

# 2 
nmp = array.numpy()
print(nmp)

# 3
array = torch.tensor(nmp);
print(array)