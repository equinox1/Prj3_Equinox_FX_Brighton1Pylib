import torch

print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
   print("GPU:", torch.cuda.get_device_name(0))
   print("CUDA version:", torch.version.cuda)


print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
