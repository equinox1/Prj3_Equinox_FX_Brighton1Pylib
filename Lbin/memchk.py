
import numpy as np
import psutil
import os
import time

os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["TF_DISABLE_POOL_ALLOCATOR"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

def log_memory_usage(stage=""):
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / (1024 ** 3)
    print(f"[{stage}] Process memory usage: {mem_gb:.2f} GB")

# Create synthetic training data (large batch and feature space)
num_samples = 1000000   # 1 million
input_dim = 2048        # high input dimension
output_dim = 512

print("Generating synthetic data...")
X = np.random.randn(num_samples, input_dim).astype(np.float32)  # ~7.6 GB
y = np.random.randn(num_samples, output_dim).astype(np.float32) # ~2 GB

log_memory_usage("After data generation")

# Define a large model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(4096, activation='relu'),   # ~33.5 million weights
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dense(output_dim)
])

model.compile(optimizer='adam', loss='mse')

log_memory_usage("After model creation")

# Train with a large batch size to increase memory load
print("Starting training...")
model.fit(X, y, epochs=1, batch_size=131072, verbose=1)

log_memory_usage("After training")

# Hold for observation
time.sleep(30)
