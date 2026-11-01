import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- load ----------
df = pd.read_csv("irish_ge_firstprefs_clean.csv")

party_cols = ["FF","SF","FG","SD","Lab","II","PBP–S","Aon","GP","Other","Ind"]
X = df[party_cols].to_numpy(dtype=float)

# ---------- PCA using CORRELATIONS on ALL constituencies ----------
mu = X.mean(axis=0)
sd = X.std(axis=0, ddof=1)
Z = (X - mu) / sd

R = np.cov(Z, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(R)

idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]

scores = Z @ eigvecs[:, :3]

out = pd.DataFrame({
    "Constituency": df["Constituency"],
    "PC1": scores[:, 0],
    "PC2": scores[:, 1],
    "PC3": scores[:, 2],
})

# ---------- choose 12 constituencies to display ----------
# (reproducible)
plot_df = out.sample(n=12, random_state=39).reset_index(drop=True)

# ---------- plot ONLY those 12 ----------
cm = 1 / 2.54
fig, ax = plt.subplots(figsize=(7*cm, 7*cm))

ax.scatter(
    plot_df["PC1"],
    plot_df["PC2"],
    c=plot_df["PC3"],
    cmap="coolwarm",
    s=45
)

for _, r in plot_df.iterrows():
    ax.text(r["PC1"], r["PC2"], r["Constituency"], fontsize=7)

ax.axhline(0, linewidth=0.5, color="grey")
ax.axvline(0, linewidth=0.5, color="grey")
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.savefig(
    "irish_ge_pca_pc1_pc2_pc3colour_12plotted.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
