import numpy as np
import pandas as pd

# load cleaned data
df = pd.read_csv("irish_ge_firstprefs_clean.csv")

# party vote-share columns (PCA variables)
party_cols = ["FF","SF","FG","SD","Lab","II","PBP–S","Aon","GP","Other","Ind"]

# extract data matrix
X = df[party_cols].to_numpy()

# centre the data (important!)
X = X - X.mean(axis=0)

# covariance matrix
Sigma = np.cov(X, rowvar=False)

print("Covariance matrix:")
print(Sigma)

# eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eigh(Sigma)

# sort eigenvalues largest -> smallest
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]

print("\nEigenvalues:")
print(eigvals)
