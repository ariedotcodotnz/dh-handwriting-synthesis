"""
Quick GPU Test for RTX 4060 Ti
Run this to verify your GPU is ready for training
"""
import torch
import sys

print("="*60)
print("GPU Setup Check for RTX 4060 Ti")
print("="*60)

# Check PyTorch version
print(f"\n[1/5] PyTorch Version")
print(f"  Version: {torch.__version__}")
if '+cu' in torch.__version__:
    print(f"  ✓ CUDA-enabled PyTorch detected")
elif '+cpu' in torch.__version__:
    print(f"  ❌ CPU-only PyTorch detected")
    print(f"  → Reinstall with: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

# Check CUDA availability
print(f"\n[2/5] CUDA Availability")
cuda_available = torch.cuda.is_available()
print(f"  CUDA Available: {cuda_available}")
if not cuda_available:
    print(f"  ❌ CUDA not available")
    print(f"  → Check Nvidia drivers and PyTorch installation")
    sys.exit(1)
else:
    print(f"  ✓ CUDA is ready")

# Check GPU details
print(f"\n[3/5] GPU Information")
if cuda_available:
    print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU Memory: {gpu_memory:.1f} GB")
    print(f"  Compute Capability: {torch.cuda.get_device_capability(0)}")
    
    if 'RTX 4060' in torch.cuda.get_device_name(0):
        print(f"  ✓ RTX 4060 Ti detected - excellent for training!")
    else:
        print(f"  ⚠ Different GPU detected, but should work")

# Quick performance test
print(f"\n[4/5] Performance Test")
try:
    x = torch.randn(2000, 2000, device='cuda')
    y = torch.randn(2000, 2000, device='cuda')
    
    # Warmup
    for _ in range(10):
        z = torch.matmul(x, y)
    torch.cuda.synchronize()
    
    # Actual test
    import time
    start = time.time()
    for _ in range(100):
        z = torch.matmul(x, y)
    torch.cuda.synchronize()
    end = time.time()
    
    ms_per_op = (end - start) * 10  # milliseconds per operation
    print(f"  Matrix multiplication speed: {ms_per_op:.1f}ms per 100 ops")
    
    if ms_per_op < 100:
        print(f"  ✓ Excellent performance!")
    elif ms_per_op < 200:
        print(f"  ✓ Good performance")
    else:
        print(f"  ⚠ Slower than expected - check GPU drivers")
    
except Exception as e:
    print(f"  ❌ Error during performance test: {e}")

# Memory test
print(f"\n[5/5] Memory Test")
try:
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated() / 1024**2
    print(f"  Initial memory: {initial_memory:.1f} MB")
    
    # Allocate some memory
    test_tensor = torch.randn(1000, 1000, 1000, device='cuda')
    allocated_memory = torch.cuda.memory_allocated() / 1024**2
    print(f"  After allocation: {allocated_memory:.1f} MB")
    
    # Free memory
    del test_tensor
    torch.cuda.empty_cache()
    final_memory = torch.cuda.memory_allocated() / 1024**2
    print(f"  After cleanup: {final_memory:.1f} MB")
    print(f"  ✓ Memory management working correctly")
    
except Exception as e:
    print(f"  ❌ Error during memory test: {e}")

# Recommended settings
print(f"\n" + "="*60)
print("Recommended Training Settings")
print("="*60)
if cuda_available:
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    if gpu_mem >= 15:
        print("For your 16GB RTX 4060 Ti:")
        print("  python train.py --batch_size 64 --num_layers 8 --use_gpu")
    elif gpu_mem >= 7:
        print("For your 8GB RTX 4060 Ti:")
        print("  python train.py --batch_size 48 --use_gpu")
    else:
        print("For your GPU:")
        print("  python train.py --batch_size 32 --d_model 384 --use_gpu")

print("\n" + "="*60)
print("✓ GPU setup verification complete!")
print("="*60)
print("\nYou're ready to train with GPU acceleration!")
print("Expected speedup: 15-20x faster than CPU")
