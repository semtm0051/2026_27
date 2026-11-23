"""Small CNN vs an all-to-all net with about the same number of parameters, on MNIST.
Writes mnist_grid.png and filters.png and prints test accuracies (for the results slide)."""
import sys, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

torch.manual_seed(0)
fake = "--fake" in sys.argv          # quick smoke test without downloading MNIST
tf = transforms.ToTensor()
if fake:
    train = datasets.FakeData(2000, (1, 28, 28), 10, transform=tf)
    test  = datasets.FakeData(500,  (1, 28, 28), 10, transform=tf)
else:
    train = datasets.MNIST("data", train=True,  download=True, transform=tf)
    test  = datasets.MNIST("data", train=False, download=True, transform=tf)
train_dl = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
test_dl  = torch.utils.data.DataLoader(test,  batch_size=1000)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)    # 28x28x8
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)   # 14x14x16
        self.pool  = nn.MaxPool2d(2)
        self.fc    = nn.Linear(7 * 7 * 16, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))          # 14x14x8
        x = self.pool(F.relu(self.conv2(x)))          # 7x7x16
        return self.fc(x.flatten(1))

class SmallMLP(nn.Module):                            # all-to-all, similar size
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 12)
        self.fc2 = nn.Linear(12, 10)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x.flatten(1))))

def accuracy(model):
    model.eval(); right = 0
    with torch.no_grad():
        for x, y in test_dl:
            right += (model(x).argmax(1) == y).sum().item()
    return right / len(test)

def fit(model, epochs=3):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"  epoch {epoch+1}: test accuracy {accuracy(model):.4f}")
    return model

nparams = lambda m: sum(p.numel() for p in m.parameters())
cnn, mlp = SmallCNN(), SmallMLP()
print(f"CNN: {nparams(cnn)} parameters"); fit(cnn)
print(f"MLP: {nparams(mlp)} parameters"); fit(mlp)

# a grid of example digits
fig, ax = plt.subplots(3, 10, figsize=(10, 3.2))
for i, a in enumerate(ax.flat):
    a.imshow(train[i][0][0], cmap="gray_r"); a.axis("off")
plt.tight_layout(pad=0.2); plt.savefig("mnist_grid.png", dpi=200, bbox_inches="tight")

# the eight learned 3x3 filters of the first layer
w = cnn.conv1.weight.detach()[:, 0]
fig, ax = plt.subplots(1, 8, figsize=(8, 1.2), gridspec_kw={"wspace": 0.6})
for k, a in enumerate(ax):
    a.imshow(w[k], cmap="RdBu_r", vmin=-w.abs().max(), vmax=w.abs().max()); a.axis("off")
plt.savefig("filters.png", dpi=200, bbox_inches="tight")
