import torch
import torch.nn as nn
import numpy as np

# Test dimensions
np.random.randn(220)  # 220 features
print(f"Original obs shape: {obs.shape}")

obs_tensor = torch.FloatTensor(obs)
print(f"Tensor shape before unsqueeze: {obs_tensor.shape}")

if len(obs_tensor.shape) == 1:
    obs_tensor = obs_tensor.unsqueeze(0)
print(f"Tensor shape after unsqueeze: {obs_tensor.shape}")

# Test linear layer
linear = nn.Linear(220, 256)
try:
    output = linear(obs_tensor)
    print(f"Linear output shape: {output.shape}")
    print("SUCCESS!")
except Exception as e:
    print(f"ERROR: {e}")

# Test with wrong shape
wrong_tensor = torch.FloatTensor(obs).reshape(220, 1)
print(f"\nWrong tensor shape: {wrong_tensor.shape}")
try:
    output = linear(wrong_tensor)
    print(f"Output shape: {output.shape}")
except Exception as e:
    print(f"ERROR with wrong shape: {e}")
