import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data
df = pd.read_csv("irish_ge_firstprefs_clean.csv")

party_cols = ["FF","SF","FG","SD","Lab","II","PBP–S","Aon","GP","Other","Ind"]

# data matrix
X = df[party_cols].to_numpy()

# centre
X = X - X.mean(axis=0)

# covariance and eigendecomposition
Sigma = np.cov(X, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(Sigma)

# sort by descending eigenvalue
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]
eigvals = eigvals[idx]

# project onto first two PCs
scores = X @ eigvecs[:, :2]

# plot
plt.figure(figsize=(6,6))
plt.scatter(scores[:,0], scores[:,1], s=30)

# label points
for i, name in enumerate(df["Constituency"]):
    plt.text(scores[i,0], scores[i,1], name, fontsize=8)

plt.axhline(0, linewidth=0.5)
plt.axvline(0, linewidth=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Irish General Election 2024: PCA of First-Preference Vote Shares")
plt.axis("equal")
plt.tight_layout()
plt.show()
