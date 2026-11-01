import numpy as np
import pandas as pd

# load cleaned data
df = pd.read_csv("irish_ge_firstprefs_clean.csv")

party_cols = ["FF","SF","FG","SD","Lab","II","PBP–S","Aon","GP","Other","Ind"]

# data matrix
X = df[party_cols].to_numpy()

# centre
X = X - X.mean(axis=0)

# covariance
Sigma = np.cov(X, rowvar=False)

# eigen-decomposition
eigvals, eigvecs = np.linalg.eigh(Sigma)

# sort by descending eigenvalue
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]

# take first three principal directions
P3 = eigvecs[:, :3]

# project data
scores = X @ P3   # shape: (n_constituencies, 3)

# assemble result
pcs = pd.DataFrame(
    scores,
    columns=["PC1", "PC2", "PC3"]
)
pcs.insert(0, "Constituency", df["Constituency"])

print(pcs)
