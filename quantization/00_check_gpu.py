import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    major, minor = torch.cuda.get_device_capability(0)
    print(f"Compute capability: {major}.{minor}")
    print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Sanity check: run a small matmul on the GPU
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.randn(1000, 1000, device="cuda")
    z = x @ y
    torch.cuda.synchronize()
    print(f"Matmul on GPU succeeded, result shape: {z.shape}")
else:
    print("No GPU detected by PyTorch -- will need to fall back to CPU-only quantization demos.")
