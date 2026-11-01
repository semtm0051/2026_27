import numpy as np
import matplotlib.pyplot as plt

# number of samples
N = 1000

# orthonormal basis vectors
u1 = np.array([1, 1]) / np.sqrt(2)    # (1,1) direction
u2 = np.array([1, -1]) / np.sqrt(2)   # (1,-1) direction

# sample along principal axes
z1 = np.random.normal(scale=np.sqrt(2), size=N)     # variance 2
z2 = np.random.normal(scale=np.sqrt(0.25), size=N)   # variance 0.5

# construct data
X = np.outer(z1, u1) + np.outer(z2, u2)

np.savetxt(
    "anisotropic_gaussian.txt",
    X,
    header="x y",
    comments=""
)

# plot
plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], s=10, alpha=0.5)
plt.axline((0,0), slope=1, linestyle="--", color="gray")
plt.axline((0,0), slope=-1, linestyle="--", color="gray")
plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")

# save as PNG
plt.savefig("anisotropic_gaussian.png", dpi=300, bbox_inches="tight")
plt.show()
