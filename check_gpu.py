import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Allocated VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Cached VRAM: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
else:
    print("CUDA not available - will train on CPU (much slower)")
