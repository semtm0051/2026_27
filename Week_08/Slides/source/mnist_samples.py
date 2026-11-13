"""
Plot example MNIST test digits: two rows, one column per digit.
Assumes MNIST is already in ./data (as downloaded by mnist_autoencoder.py).
Saves mnist_samples.pdf.
"""

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets

N_ROWS = 2

test = datasets.MNIST("data", train=False, download=False)
images = test.data.numpy()
labels = test.targets.numpy()

fig, axes = plt.subplots(N_ROWS, 10, figsize=(10, 1.1 * N_ROWS))
for d in range(10):
    idx = np.where(labels == d)[0][:N_ROWS]
    for r in range(N_ROWS):
        axes[r, d].imshow(images[idx[r]], cmap="gray_r")
        axes[r, d].axis("off")

fig.tight_layout()
fig.savefig("mnist_samples.pdf")
print("saved mnist_samples.pdf")
