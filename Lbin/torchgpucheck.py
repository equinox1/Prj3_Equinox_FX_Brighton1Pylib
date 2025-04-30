import torch
import time

def check_gpu():
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("No GPU detected. Using CPU.")
        return False

def complex_calculation():
    print("\nStarting complex calculation...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate large random matrices
    size = 5000
    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)

    start_time = time.time()

    # Matrix multiplication
    C = torch.matmul(A, B)

    end_time = time.time()
    print(f"Calculation completed in {end_time - start_time:.2f} seconds.")

    # Simple checksum to verify computation
    print(f"Sum of result matrix: {C.sum().item()}")

if __name__ == "__main__":
    gpu_available = check_gpu()
    complex_calculation()
