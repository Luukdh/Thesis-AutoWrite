import torchvision
import numpy as np
import torch

lst = np.array([[1,2,3,4,5,6,7,8,9],[10,11,12,13,14,15,16,17,18],[19,20,21,22,23,24,25,26,27]])
# lst = [[1,2,3],[4,5,6],[7,8,9]]

print(lst)

ten = torch.Tensor(lst)

print(ten[:, 2:-2])
print(ten.shape)
