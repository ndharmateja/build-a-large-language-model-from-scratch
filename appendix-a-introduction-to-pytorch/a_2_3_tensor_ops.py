import torch

tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])

print(tensor_2d)
print(tensor_2d.shape)

tensor_2d_reshaped = tensor_2d.reshape(3, 2)
print(tensor_2d_reshaped)
print(tensor_2d_reshaped.shape)