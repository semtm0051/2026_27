"""ICA versus PCA on pretrained word embeddings.

Loads GloVe (100-d, Wikipedia + Gigaword) via gensim, keeps the most frequent
words, and runs PCA and FastICA. Prints the top words on each component, and
plots words on two chosen ICA axes (the "two arms" picture).

  pip install gensim scikit-learn matplotlib
  python ica_word_embeddings.py

The first run downloads ~130 MB of embeddings (cached in ~/gensim-data).
"""

import numpy as np
import matplotlib.pyplot as plt
import gensim.downloader as api
from scipy.stats import skew
from sklearn.decomposition import PCA, FastICA

MODEL = "glove-wiki-gigaword-100"
N_WORDS = 20000      # most frequent words to keep (gensim lists them by frequency)
N_SKIP = 100         # skip the very commonest words ("the", "of", ...)
TOP = 10             # words to show per component
SEED = 0

# ------------------------------------------------------------------ load data
kv = api.load(MODEL)
words = [w for w in kv.index_to_key[N_SKIP:N_SKIP + N_WORDS] if w.isalpha()]
X = np.array([kv[w] for w in words])
X = X - X.mean(axis=0)
words = np.array(words)
print(f"{len(words)} words, {X.shape[1]} dimensions")


def top_words(S):
    """For each column of S, the TOP words with the largest values."""
    return [words[np.argsort(-S[:, k])[:TOP]] for k in range(S.shape[1])]


# ----------------------------------------------------------------------- PCA
S_pca = PCA(random_state=SEED).fit_transform(X)

# ----------------------------------------------------------------------- ICA
ica = FastICA(n_components=X.shape[1], whiten="unit-variance",
              max_iter=2000, tol=1e-4, random_state=SEED)
S_ica = ica.fit_transform(X)

# ICA cannot fix signs: flip so each component has positive skew, then
# order components by skewness (most heavy-tailed, usually cleanest, first).
S_ica *= np.sign(skew(S_ica, axis=0))
order = np.argsort(-skew(S_ica, axis=0))
S_ica = S_ica[:, order]

# --------------------------------------------------------------- word lists
N_SHOW = 20
print("\n=== PCA: top words on the first components ===")
for k, ws in enumerate(top_words(S_pca)[:N_SHOW]):
    print(f"pc{k:<3d} {' '.join(ws)}")

print("\n=== ICA: top words on the most skewed components ===")
for k, ws in enumerate(top_words(S_ica)[:N_SHOW]):
    print(f"ic{k:<3d} {' '.join(ws)}")

with open("ica_components.txt", "w") as f:
    for k, ws in enumerate(top_words(S_ica)):
        f.write(f"ic{k:<3d} {' '.join(ws)}\n")
with open("pca_components.txt", "w") as f:
    for k, ws in enumerate(top_words(S_pca)):
        f.write(f"pc{k:<3d} {' '.join(ws)}\n")


# ------------------------------------------------------- two-arm scatter plot
def find_component(probe):
    """Index of the ICA component on which the probe words score highest."""
    idx = [np.where(words == w)[0][0] for w in probe if w in words]
    return int(np.argmax(S_ica[idx].mean(axis=0)))


# pick two axes by naming a few words that should lie on each
AXIS_A = ["soup", "sauce", "cheese", "chicken", "bread"]
AXIS_B = ["football", "soccer", "rugby", "goal", "striker"]
LABEL_A, LABEL_B = "food", "football"

a, b = find_component(AXIS_A), find_component(AXIS_B)
print(f"\nscatter axes: ic{a} ({LABEL_A}) and ic{b} ({LABEL_B})")

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})
fig, ax = plt.subplots(figsize=(2.4, 2.4))
ax.scatter(S_ica[:, a], S_ica[:, b], s=2, alpha=0.3, color="#4C72B0", lw=0)

# list a few of the most extreme words beside the tip of each arm
top_a = words[np.argsort(-S_ica[:, a])[:4]]
top_b = words[np.argsort(-S_ica[:, b])[:4]]
ax.text(S_ica[:, a].max(), 0.6, "\n".join(top_a), fontsize=6, color="#C44E52",
        ha="right", va="bottom")
ax.text(0.8, S_ica[:, b].max(), "\n".join(top_b), fontsize=6, color="#DD8452",
        ha="left", va="top")

ax.axhline(0, color="0.85", lw=0.8, zorder=0)
ax.axvline(0, color="0.85", lw=0.8, zorder=0)
ax.set_xlabel(f"ic{a} ({LABEL_A})")
ax.set_ylabel(f"ic{b} ({LABEL_B})")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.2)
fig.savefig("ica_words.pdf")
fig.savefig("ica_words.png", dpi=200)
print("wrote ica_words.pdf/.png, ica_components.txt, pca_components.txt")
