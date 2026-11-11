import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def print_tensor_info(tensor: torch.tensor):
    print(tensor)
    print(f"Shape    : {tensor.shape}")
    print(f"Data type: {tensor.dtype}")
    print(f"Device   : {tensor.device}")
    print()


tensor_0d = torch.tensor(10, device=device)
tensor_1d = torch.tensor([3, 1, 3], device=device)
tensor_2d = torch.tensor([[1, 2], [3, 4]], device=device)
tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], device=device)

print_tensor_info(tensor_0d)
print_tensor_info(tensor_1d)
print_tensor_info(tensor_2d)
print_tensor_info(tensor_3d)


tensor_3d_float = tensor_3d.to(torch.float32)
print_tensor_info(tensor_3d_float)
