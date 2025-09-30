import torch
import time

def test_cuda():
    print("Checking CUDA availability...")
    if not torch.cuda.is_available():
        print("❌ CUDA is not available on this system.")
        return

    device = torch.device("cuda")
    print(f"✅ CUDA is available. Using device: {torch.cuda.get_device_name(device)}")

    # Create two random tensors on CUDA
    size = 4096
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    print("Running matrix multiplication on GPU...")
    start_time = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()  # Ensure all CUDA ops are finished
    end_time = time.time()

    print(f"✅ Computation completed. Time taken: {end_time - start_time:.4f} seconds")
    print(f"Result tensor shape: {c.shape}")

if __name__ == "__main__":
    test_cuda()
