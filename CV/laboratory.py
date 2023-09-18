import numpy as np
import matplotlib.pyplot as plt

def get_positional_embedding(patch_size, hidden_dim):
    positions = np.arange(patch_size)
    angle_rads = 1 / np.power(10000, (2 * (np.arange(hidden_dim) // 2)) / np.float32(hidden_dim))
    angle_rads = angle_rads.reshape(1, -1)  # Reshape to broadcast across positions
    positions = positions.reshape(-1, 1)  # Reshape positions to broadcast across angles
    positional_embedding = np.sin(positions * angle_rads)  # Compute sine values
    return positional_embedding

patch_size = 196
hidden_dim = 256

positional_embedding = get_positional_embedding(patch_size, hidden_dim)

# Visualize the positional embedding for the first dimension
plt.figure(figsize=(12, 4))
plt.plot(np.arange(patch_size), positional_embedding[:, 0], marker='o', linestyle='-')
plt.title('Positional Embedding for the First Dimension')
plt.xlabel('Position')
plt.ylabel('Value')
plt.show()

[print(i) for i in sorted(positional_embedding[:, 0].tolist())]