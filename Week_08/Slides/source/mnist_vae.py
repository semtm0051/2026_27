"""
MNIST variational auto-encoder with a 2D latent space, same dimensions as
mnist_autoencoder.py:

    784 -> 512 -> 256 -> (mu, log sigma^2) in R^2 -> z -> 256 -> 512 -> 784

Trains (or loads vae_mnist.pt) and saves:

  vae_latent_scatter.pdf     test set in latent space (encoder means), by digit
  vae_reconstructions.pdf    test digits and their reconstructions
  vae_latent_line_A_B.pdf    decoded images along lines between digit centroids
  vae_latent_circle_R.pdf    decoded images around circles of radius R
  vae_latent_grid.pdf        decoded images on a grid over latent space
  vae_samples.pdf            new digits: z ~ Normal(0,1), decoded
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------- settings
LATENT = 2
EPOCHS = 30
BATCH = 256
LR = 1e-3
BETA = 1.0                   # weight on the KL term; 1 is the standard VAE
SEED = 0
MODEL_FILE = "vae_mnist.pt"  # delete this file to force retraining

LINES = [(4, 7), (1, 0), (0, 2)]
N_STEPS = 12

CIRCLE_RADII = [1.0, 2.5]    # latent space is now roughly Normal(0,1)
N_CIRCLE = 16

N_RECON = 10
N_GRID = 15
GRID_LIM = 2.5               # grid covers [-GRID_LIM, GRID_LIM]^2
N_SAMPLES = 32

torch.manual_seed(SEED)
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

# ---------------------------------------------------------------- data
tf = transforms.ToTensor()
train = datasets.MNIST("data", train=True, download=True, transform=tf)
test = datasets.MNIST("data", train=False, download=True, transform=tf)
train_loader = DataLoader(train, batch_size=BATCH, shuffle=True)
test_loader = DataLoader(test, batch_size=1000, shuffle=False)


# ---------------------------------------------------------------- model
class VAE(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.to_mu = nn.Linear(256, latent)
        self.to_logvar = nn.Linear(256, latent)   # log sigma^2, so no positivity constraint
        self.decoder = nn.Sequential(
            nn.Linear(latent, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784),                  # logits
        )

    def encode(self, x):
        h = self.body(x)
        return self.to_mu(h), self.to_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps    # reparameterisation trick
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def decode_images(self, z):
        return torch.sigmoid(self.decoder(z)).view(-1, 28, 28)


def vae_loss(logits, x, mu, logvar):
    """Per-image -ELBO: Bernoulli reconstruction + KL(Normal(mu,sigma) || Normal(0,1))."""
    rec = F.binary_cross_entropy_with_logits(logits, x.view(x.size(0), -1),
                                             reduction="none").sum(1)
    kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(1)
    return rec, kl


model = VAE().to(device)

# ---------------------------------------------------------------- train or load
if os.path.exists(MODEL_FILE):
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    print(f"loaded {MODEL_FILE}")
else:
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot_rec = tot_kl = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            logits, mu, logvar = model(x)
            rec, kl = vae_loss(logits, x, mu, logvar)
            loss = (rec + BETA * kl).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_rec += rec.sum().item()
            tot_kl += kl.sum().item()
        n = len(train)
        print(f"epoch {epoch:3d}   reconstruction {tot_rec / n:.2f}   KL {tot_kl / n:.2f}"
              f"   -ELBO {(tot_rec + tot_kl) / n:.2f}")
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"saved {MODEL_FILE}")

# ---------------------------------------------------------------- encode test set
model.eval()
MU, Y, REC, KL, RECON = [], [], [], [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        mu, logvar = model.encode(x)
        logits = model.decoder(mu)                # reconstruct from the mean
        rec, kl = vae_loss(logits, x, mu, logvar)
        MU.append(mu.cpu()); Y.append(y)
        REC.append(rec.cpu()); KL.append(kl.cpu())
        RECON.append(torch.sigmoid(logits).view(-1, 28, 28).cpu())
Z = torch.cat(MU).numpy()
Y = torch.cat(Y).numpy()
REC = torch.cat(REC).numpy()
KL = torch.cat(KL).numpy()
RECON = torch.cat(RECON).numpy()
ORIG = test.data.numpy() / 255.0
centroids = np.stack([Z[Y == d].mean(axis=0) for d in range(10)])

print(f"latent range: z1 in [{Z[:,0].min():.1f}, {Z[:,0].max():.1f}], "
      f"z2 in [{Z[:,1].min():.1f}, {Z[:,1].max():.1f}]")
print(f"test: reconstruction {REC.mean():.2f}   KL {KL.mean():.2f}   -ELBO {(REC + KL).mean():.2f}")


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


def path_figure(path, fname, title, closed=False, n_cols=None):
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
    ax.plot(path[0, 0], path[0, 1], "ks", ms=7)
    if closed:
        ax.set_aspect("equal", adjustable="datalim")
    for k in range(n):
        show(fig.add_subplot(gs[1 + k // n_cols, k % n_cols]), imgs[k])
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


# ---------------------------------------------------------------- latent space
fig, ax = plt.subplots(figsize=(6, 5))
sc = scatter(ax, Z, Y, "VAE latent space (encoder means, test set)")
fig.colorbar(sc, ax=ax, ticks=range(10), label="digit")
fig.tight_layout()
fig.savefig("vae_latent_scatter.pdf")
plt.close(fig)

# ---------------------------------------------------------------- reconstructions
idx = [np.where(Y == d % 10)[0][d // 10] for d in range(N_RECON)]
fig, axes = plt.subplots(2, N_RECON, figsize=(1.0 * N_RECON, 2.6))
for j, i in enumerate(idx):
    show(axes[0, j], ORIG[i], str(Y[i]))
    show(axes[1, j], RECON[i])
axes[0, 0].text(-6, 14, "input", rotation=90, va="center", ha="center")
axes[1, 0].text(-6, 14, "output", rotation=90, va="center", ha="center")
fig.suptitle("VAE: test digits and their reconstructions")
fig.tight_layout()
fig.savefig("vae_reconstructions.pdf")
plt.close(fig)

# ---------------------------------------------------------------- lines
ts = np.linspace(0, 1, N_STEPS)[:, None]
for a, b in LINES:
    line = (1 - ts) * centroids[a] + ts * centroids[b]
    path_figure(line, f"vae_latent_line_{a}_{b}.pdf", f"VAE: line from {a} to {b}")

# ---------------------------------------------------------------- circles
theta = np.linspace(0, 2 * np.pi, N_CIRCLE, endpoint=False)[:, None]
for r in CIRCLE_RADII:
    circle = r * np.hstack([np.cos(theta), np.sin(theta)])
    path_figure(circle, f"vae_latent_circle_{r:g}.pdf",
                f"VAE: circle of radius {r:g}", closed=True, n_cols=N_CIRCLE // 2)

# ---------------------------------------------------------------- decoded grid
g = np.linspace(-GRID_LIM, GRID_LIM, N_GRID)
grid = np.array([[x, y] for y in g[::-1] for x in g], dtype=np.float32)
imgs = model.decode_images(torch.from_numpy(grid).to(device)).cpu().numpy()
mosaic = imgs.reshape(N_GRID, N_GRID, 28, 28).transpose(0, 2, 1, 3).reshape(N_GRID * 28, N_GRID * 28)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5.2))
scatter(ax0, Z, Y, "Test set with the grid points")
ax0.plot(grid[:, 0], grid[:, 1], "k+", ms=5)
ax1.imshow(mosaic, cmap="gray_r", vmin=0, vmax=1,
           extent=[-GRID_LIM, GRID_LIM, -GRID_LIM, GRID_LIM])
ax1.set_title("Decoded grid")
ax1.set_xlabel("$z_1$")
ax1.set_ylabel("$z_2$")
fig.tight_layout()
fig.savefig("vae_latent_grid.pdf")
plt.close(fig)

# ---------------------------------------------------------------- new digits from the prior
z = torch.randn(N_SAMPLES, LATENT, generator=torch.Generator().manual_seed(SEED))
imgs = model.decode_images(z.to(device)).cpu().numpy()
n_cols = 8
fig, axes = plt.subplots(N_SAMPLES // n_cols, n_cols, figsize=(n_cols, N_SAMPLES // n_cols))
for k, ax in enumerate(axes.flat):
    show(ax, imgs[k])
fig.suptitle(r"New digits: $z_i\sim$ Normal(0,1), decoded")
fig.tight_layout()
fig.savefig("vae_samples.pdf")
plt.close(fig)

print("done")
