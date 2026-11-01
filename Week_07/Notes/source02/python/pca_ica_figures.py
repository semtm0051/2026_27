"""Two illustrative figures: PCA on a rotated 2D Gaussian, and ICA-style data
concentrated along two non-orthogonal axes. Writes pca.pdf/.png and ica.pdf/.png."""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)

# sized for half the width of a 16:10 / 4:3 beamer frame (~6 cm wide)
FIGSIZE = (2.4, 2.4)
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})

POINT_COLOUR = "#4C72B0"
AXIS_COLOURS = ["#C44E52", "#DD8452"]
LIM = 6
TICKS = [-6, -3, 0, 3, 6]


def unit(theta_deg):
    t = np.deg2rad(theta_deg)
    return np.array([np.cos(t), np.sin(t)])


def style(ax):
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xticks(TICKS)
    ax.set_yticks(TICKS)
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- PCA figure
n = 500
angle = 30                      # orientation of the major axis, degrees
sd_major, sd_minor = 2.0, 0.6   # standard deviations along the two axes

R = np.column_stack([unit(angle), unit(angle + 90)])
cov = R @ np.diag([sd_major**2, sd_minor**2]) @ R.T
X = rng.multivariate_normal([0, 0], cov, size=n)

# estimate the principal axes from the sample, as PCA would
mu = X.mean(axis=0)
evals, evecs = np.linalg.eigh(np.cov(X.T))
order = np.argsort(evals)[::-1]
evals, evecs = evals[order], evecs[:, order]
evecs *= np.sign(evecs[0] + evecs[1])  # eigenvector signs are arbitrary; point them up/right

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.scatter(X[:, 0], X[:, 1], s=3, alpha=0.5, color=POINT_COLOUR, lw=0)
for k in range(2):
    v = evecs[:, k] * 2 * np.sqrt(evals[k])   # arrow length: 2 standard deviations
    ax.annotate("", xy=mu + v, xytext=mu,
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color=AXIS_COLOURS[k]))
    ax.annotate("", xy=mu - v, xytext=mu,
                arrowprops=dict(arrowstyle="-", lw=1.5, color=AXIS_COLOURS[k]))
    ax.text(*(mu + v + 0.55 * v / np.linalg.norm(v)), f"pc{k + 1}", color=AXIS_COLOURS[k],
            ha="center", va="center", fontsize=9)
style(ax)
fig.tight_layout(pad=0.2)
fig.savefig("pca.pdf")
fig.savefig("pca.png", dpi=200)

# ---------------------------------------------------------------- ICA figure
n = 600
a1, a2 = unit(0), unit(45)       # two non-orthogonal source directions: x axis and x = y
frac_strays = 0.06               # fraction of points far from both axes
big_sd, small_sd = 2.0, 0.15     # spread along the chosen axis, and off it

n_stray = int(frac_strays * n)
n_axis = n - n_stray
which = rng.integers(0, 2, size=n_axis)      # which axis each point lies near
big = rng.normal(0, big_sd, size=n_axis)     # "a lot of one axis"
small = rng.normal(0, small_sd, size=n_axis)  # "plus a bit of the other"

s1 = np.where(which == 0, big, small)
s2 = np.where(which == 0, small, big)
A = np.column_stack([a1, a2])                 # mixing matrix
X_axis = np.column_stack([s1, s2]) @ A.T
X_stray = rng.normal(0, 1.6, size=(n_stray, 2)) @ A.T
X = np.vstack([X_axis, X_stray])

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.scatter(X[:, 0], X[:, 1], s=3, alpha=0.5, color=POINT_COLOUR, lw=0)
for k, a in enumerate((a1, a2)):
    ax.plot(*np.column_stack([-1.5 * LIM * a, 1.5 * LIM * a]),
            color=AXIS_COLOURS[k], lw=1.5)
    side = 1 if k == 0 else -1   # put each label on the outer side of its line
    ax.text(*(0.8 * LIM * a + side * 0.6 * np.array([a[1], -a[0]])), f"ic{k + 1}",
            color=AXIS_COLOURS[k], ha="center", va="center", fontsize=9)
style(ax)
fig.tight_layout(pad=0.2)
fig.savefig("ica.pdf")
fig.savefig("ica.png", dpi=200)

print("wrote pca.pdf, pca.png, ica.pdf, ica.png")
