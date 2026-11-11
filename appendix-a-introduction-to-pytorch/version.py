import torch

print(f"PyTorch Version: {torch.__version__}")

# Check if Apple Silicon GPU (MPS) is available
if torch.backends.mps.is_available():
    print("Apple Silicon GPU (MPS) is AVAILABLE and active!")
else:
    print("MPS not available, falling back to CPU.")

# Select device accordingly
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(device)

# Test tensor creation on your M3 Pro GPU
x = torch.ones((1000, 1000), device=device)
print(f"Tensor is allocated on: {x.device}")
