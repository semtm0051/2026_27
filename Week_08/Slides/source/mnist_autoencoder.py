"""
MNIST auto-encoder with a 2D latent space.

Trains a 784-512-256-2-256-512-784 MLP auto-encoder (or loads it from
ae_mnist.pt if that exists), then saves:

  latent_scatter.pdf        test set in latent space, coloured by digit
  latent_scatter_zoom.pdf   the same, zoomed in on the middle
  reconstructions.pdf       some test digits and their reconstructions
  latent_line_A_B.pdf       decoded images along a line from digit A to digit B
  latent_circle.pdf         decoded images around a circle in latent space
  worst_reconstructions.pdf test digits with the highest reconstruction error
  pca_scatter.pdf           first two principal components, for comparison
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------- settings
LATENT = 2
EPOCHS = 30
BATCH = 256
LR = 1e-3
SEED = 0
MODEL_FILE = "ae_mnist.pt"   # delete this file to force retraining

LINES = [(4, 7), (1, 0)]     # pairs of digits; line joins their latent centroids
N_STEPS = 12                 # images along each line

CIRCLE_CENTRE = (0.0, 0.0)
CIRCLE_RADIUS = 10.0         # latent scale is arbitrary: check the printed range
N_CIRCLE = 16                # images around the circle

ZOOM_PERCENTILES = (25, 75)  # zoom window: this range of each coordinate
N_RECON = 10                 # digits shown in reconstructions.pdf
N_WORST = 8                  # digits shown in worst_reconstructions.pdf

torch.manual_seed(SEED)
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

# ---------------------------------------------------------------- data
tf = transforms.ToTensor()  # pixels in [0, 1]
train = datasets.MNIST("data", train=True, download=True, transform=tf)
test = datasets.MNIST("data", train=False, download=True, transform=tf)
train_loader = DataLoader(train, batch_size=BATCH, shuffle=True)
test_loader = DataLoader(test, batch_size=1000, shuffle=False)


# ---------------------------------------------------------------- model
class AutoEncoder(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, latent),            # linear bottleneck
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784),               # logits; sigmoid gives pixels
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    @torch.no_grad()
    def decode_images(self, z):
        return torch.sigmoid(self.decoder(z)).view(-1, 28, 28)


model = AutoEncoder().to(device)

# ---------------------------------------------------------------- train or load
if os.path.exists(MODEL_FILE):
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    print(f"loaded {MODEL_FILE}")
else:
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")  # sharper digits than MSE
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            logits, _ = model(x)
            loss = loss_fn(logits, x.view(x.size(0), -1)) / x.size(0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        print(f"epoch {epoch:3d}   loss per image {total / len(train):.2f}")
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"saved {MODEL_FILE}")

# ---------------------------------------------------------------- encode test set
model.eval()
Z, Y, ERR, RECON = [], [], [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits, z = model(x)
        flat = x.view(x.size(0), -1)
        err = F.binary_cross_entropy_with_logits(logits, flat, reduction="none").sum(1)
        Z.append(z.cpu())
        Y.append(y)
        ERR.append(err.cpu())
        RECON.append(torch.sigmoid(logits).view(-1, 28, 28).cpu())
Z = torch.cat(Z).numpy()
Y = torch.cat(Y).numpy()
ERR = torch.cat(ERR).numpy()
RECON = torch.cat(RECON).numpy()
ORIG = test.data.numpy() / 255.0          # same order as test_loader
centroids = np.stack([Z[Y == d].mean(axis=0) for d in range(10)])

print(f"latent range: z1 in [{Z[:,0].min():.1f}, {Z[:,0].max():.1f}], "
      f"z2 in [{Z[:,1].min():.1f}, {Z[:,1].max():.1f}]")
print(f"mean test reconstruction error {ERR.mean():.2f}")


# ---------------------------------------------------------------- helpers
def scatter(ax, Z, Y, title, s=2, labels=True):
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=Y, cmap="tab10", s=s, alpha=0.6,
                    vmin=-0.5, vmax=9.5, rasterized=True)
    if labels:
        for d in range(10):
            c = Z[Y == d].mean(axis=0)
            ax.text(c[0], c[1], str(d), fontsize=14, weight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.2", fc="white", alpha=0.8))
    ax.set_title(title)
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    return sc


def show(ax, img, title=None):
    ax.imshow(img, cmap="gray_r", vmin=0, vmax=1)
    ax.axis("off")
    if title is not None:
        ax.set_title(title, fontsize=9)


def image_pairs(idx, fname, suptitle, show_err=False):
    """Originals on the top row, reconstructions underneath."""
    n = len(idx)
    fig, axes = plt.subplots(2, n, figsize=(1.0 * n, 2.6))
    for j, i in enumerate(idx):
        t = f"{Y[i]}" + (f"\n{ERR[i]:.0f}" if show_err else "")
        show(axes[0, j], ORIG[i], t)
        show(axes[1, j], RECON[i])
    axes[0, 0].text(-6, 14, "input", rotation=90, va="center", ha="center")
    axes[1, 0].text(-6, 14, "output", rotation=90, va="center", ha="center")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def path_figure(path, fname, title, closed=False, n_cols=None):
    """Scatter with a path drawn on it, and the decoded images underneath."""
    imgs = model.decode_images(torch.from_numpy(path.astype(np.float32)).to(device)).cpu().numpy()
    n = len(path)
    n_cols = n_cols or n
    n_rows = int(np.ceil(n / n_cols))
    fig = plt.figure(figsize=(max(n_cols * 0.9, 7), 5 + 0.9 * n_rows))
    gs = fig.add_gridspec(1 + n_rows, n_cols, height_ratios=[5] + [0.9] * n_rows)
    ax = fig.add_subplot(gs[0, :])
    scatter(ax, Z, Y, title)
    p = np.vstack([path, path[:1]]) if closed else path
    ax.plot(p[:, 0], p[:, 1], "k-", lw=1.5)
    ax.plot(path[:, 0], path[:, 1], "ko", ms=4)
    ax.plot(path[0, 0], path[0, 1], "ks", ms=7)   # square marks the start
    if closed:
        ax.set_aspect("equal", adjustable="datalim")
    for k in range(n):
        show(fig.add_subplot(gs[1 + k // n_cols, k % n_cols]), imgs[k])
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


# ---------------------------------------------------------------- 1. clusters
fig, ax = plt.subplots(figsize=(6, 5))
sc = scatter(ax, Z, Y, "Auto-encoder latent space (test set)")
fig.colorbar(sc, ax=ax, ticks=range(10), label="digit")
fig.tight_layout()
fig.savefig("latent_scatter.pdf")
plt.close(fig)

# ---------------------------------------------------------------- 1b. zoomed in on the middle
lo, hi = ZOOM_PERCENTILES
x0, x1 = np.percentile(Z[:, 0], [lo, hi])
y0, y1 = np.percentile(Z[:, 1], [lo, hi])
fig, ax = plt.subplots(figsize=(6, 5))
sc = scatter(ax, Z, Y, "Latent space, middle region", s=6, labels=False)
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
fig.colorbar(sc, ax=ax, ticks=range(10), label="digit")
fig.tight_layout()
fig.savefig("latent_scatter_zoom.pdf")
plt.close(fig)

# ---------------------------------------------------------------- 2. examples and reconstructions
idx = [np.where(Y == d % 10)[0][d // 10] for d in range(N_RECON)]  # one per digit, cycling
image_pairs(idx, "reconstructions.pdf", "Test digits and their reconstructions")

# ---------------------------------------------------------------- 3a. lines between centroids
ts = np.linspace(0, 1, N_STEPS)[:, None]
for a, b in LINES:
    line = (1 - ts) * centroids[a] + ts * centroids[b]
    path_figure(line, f"latent_line_{a}_{b}.pdf",
                f"Line from {a} to {b} in latent space")

# ---------------------------------------------------------------- 3b. circle
theta = np.linspace(0, 2 * np.pi, N_CIRCLE, endpoint=False)[:, None]
circle = np.array(CIRCLE_CENTRE) + CIRCLE_RADIUS * np.hstack([np.cos(theta), np.sin(theta)])
path_figure(circle, "latent_circle.pdf",
            f"Circle of radius {CIRCLE_RADIUS:g} in latent space",
            closed=True, n_cols=N_CIRCLE // 2)

# ---------------------------------------------------------------- 4. worst reconstructions
worst = np.argsort(ERR)[::-1][:N_WORST]
image_pairs(worst, "worst_reconstructions.pdf",
            "Highest reconstruction error (label, error)", show_err=True)

# ---------------------------------------------------------------- PCA for comparison
X = ORIG.reshape(-1, 784)
Xc = X - X.mean(axis=0)
_, _, Vt = np.linalg.svd(Xc, full_matrices=False)
P = Xc @ Vt[:2].T
fig, ax = plt.subplots(figsize=(6, 5))
sc = scatter(ax, P, Y, "PCA, first two components (test set)")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
fig.colorbar(sc, ax=ax, ticks=range(10), label="digit")
fig.tight_layout()
fig.savefig("pca_scatter.pdf")
plt.close(fig)

print("done")

"""
Figures motivating the VAE. Paste at the end of mnist_autoencoder.py
(it uses model, Z, Y, centroids, scatter, show and path_figure from there).

  latent_line_gap.pdf     a line that crosses empty latent space
  latent_line_out.pdf     a line that leaves the data altogether
  latent_grid.pdf         decoded images on a regular grid over latent space
  latent_samples.pdf      random latent points, Gaussian matched to the data, decoded
"""

# ---------------------------------------------------------------- lines through the gaps
GAP_LINE = (0, 2)                      # pick two clusters with empty space between them
OUT_TARGET = np.array([0.0, 30.0])     # a point outside the cloud

ts = np.linspace(0, 1, N_STEPS)[:, None]
a, b = GAP_LINE
line = (1 - ts) * centroids[a] + ts * centroids[b]
path_figure(line, "latent_line_gap.pdf", f"Line from {a} to {b}: crossing a gap")

line = (1 - ts) * centroids[b] + ts * OUT_TARGET
path_figure(line, "latent_line_out.pdf", f"Line from {b} out of the data")

# ---------------------------------------------------------------- decoded grid
N_GRID = 15
gx = np.linspace(np.percentile(Z[:, 0], 1), np.percentile(Z[:, 0], 99), N_GRID)
gy = np.linspace(np.percentile(Z[:, 1], 1), np.percentile(Z[:, 1], 99), N_GRID)
grid = np.array([[x, y] for y in gy[::-1] for x in gx], dtype=np.float32)  # top row first
imgs = model.decode_images(torch.from_numpy(grid).to(device)).cpu().numpy()
mosaic = imgs.reshape(N_GRID, N_GRID, 28, 28).transpose(0, 2, 1, 3).reshape(N_GRID * 28, N_GRID * 28)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5.2))
scatter(ax0, Z, Y, "Test set with the grid points")
ax0.plot(grid[:, 0], grid[:, 1], "k+", ms=5)
ax1.imshow(mosaic, cmap="gray_r", vmin=0, vmax=1,
           extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect="auto")
ax1.set_title("Decoded grid")
ax1.set_xlabel("$z_1$")
ax1.set_ylabel("$z_2$")
fig.tight_layout()
fig.savefig("latent_grid.pdf")
plt.close(fig)

# ---------------------------------------------------------------- random samples
N_SAMPLES = 16
rng = np.random.default_rng(SEED)
samples = rng.multivariate_normal(Z.mean(axis=0), np.cov(Z.T), N_SAMPLES).astype(np.float32)
imgs = model.decode_images(torch.from_numpy(samples).to(device)).cpu().numpy()

fig = plt.figure(figsize=(8, 7))
gs = fig.add_gridspec(3, 8, height_ratios=[5, 1, 1])
ax = fig.add_subplot(gs[0, :])
scatter(ax, Z, Y, "Random latent points (Gaussian matched to the data)")
ax.plot(samples[:, 0], samples[:, 1], "k*", ms=9)
for k in range(N_SAMPLES):
    ax.annotate(str(k + 1), samples[k], textcoords="offset points", xytext=(4, 4), fontsize=8)
    show(fig.add_subplot(gs[1 + k // 8, k % 8]), imgs[k], str(k + 1))
fig.tight_layout()
fig.savefig("latent_samples.pdf")
plt.close(fig)

print("saved latent_line_gap.pdf, latent_line_out.pdf, latent_grid.pdf, latent_samples.pdf")
