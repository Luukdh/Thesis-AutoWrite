import torchvision
import numpy as np
import torch

lst = np.array([[1,2,3],[4,5,6],[7,8,9]])
# lst = [[1,2,3],[4,5,6],[7,8,9]]

print(lst)

ten = torch.Tensor(lst).T

print(ten)
