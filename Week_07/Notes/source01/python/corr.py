import numpy as np

# load data (skip header line)
X = np.loadtxt("anisotropic_gaussian.txt", skiprows=1)

# covariance matrix
cov = np.cov(X, rowvar=False)

# correlation matrix
corr = np.corrcoef(X, rowvar=False)

print("Covariance matrix:")
print(cov)

print("\nCorrelation matrix:")
print(corr)
