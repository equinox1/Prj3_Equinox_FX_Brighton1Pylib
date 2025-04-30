import tensorflow as tf
import time

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected: {gpus[0].name}")
        return True
    else:
        print("No GPU detected. Using CPU.")
        return False

def complex_calculation():
    print("\nStarting complex calculation...")

    # Generate large random matrices
    size = 5000
    A = tf.random.normal((size, size))
    B = tf.random.normal((size, size))

    start_time = time.time()

    # Matrix multiplication
    C = tf.matmul(A, B)

    end_time = time.time()
    print(f"Calculation completed in {end_time - start_time:.2f} seconds.")

    # Just to make sure something is computed
    print(f"Sum of result matrix: {tf.reduce_sum(C).numpy()}")

if __name__ == "__main__":
    gpu_available = check_gpu()
    complex_calculation()
