

import torch 

A = torch.arange(1, 17).reshape(4,4)

# torch.arrange(start, end, step)
B = torch.arange(17, 1, -1).reshape(4,4)


print(A)
print(B)

multiple = torch.matmul(A,B)

print(A[:, 0:2])