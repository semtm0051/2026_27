import numpy as np, matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10

ds = CIFAR10(root="data", train=True, download=True)
X, y = ds.data, np.array(ds.targets)
rng = np.random.default_rng(1)
n = 4
fig, ax = plt.subplots(n, 10, figsize=(10, n * 1.05))
for c in range(10):
    idx = rng.choice(np.where(y == c)[0], n, replace=False)
    for r, i in enumerate(idx):
        ax[r, c].imshow(X[i], interpolation="nearest")
        ax[r, c].axis("off")
    ax[0, c].set_title(ds.classes[c], fontsize=11)
plt.tight_layout(pad=0.2)
plt.savefig("cifar_grid.png", dpi=200, bbox_inches="tight")
