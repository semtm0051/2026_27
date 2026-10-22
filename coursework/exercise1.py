"""
exercise1.py -- a worked example to get you started.

Fits networks of different sizes to ALL the data and plots the result.
Read it, run it, then copy and adapt it for the later exercises.
"""
import numpy as np
import matplotlib.pyplot as plt
import fit

x, y = fit.load("data.csv")
xgrid = np.linspace(-2, 2, 400)            # where we evaluate the fitted curve

sizes = [0, 1, 2, 5, 20, 100]
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for ax, H in zip(axes.ravel(), sizes):
    net, history = fit.train(x, y, hidden=H, epochs=20000)
    ax.plot(x, y, "o", ms=4)
    ax.plot(xgrid, fit.predict(net, xgrid), "r")
    ax.set_ylim(-2.5, 3)
    ax.set_title(f"H = {H}: {fit.n_params(net)} parameters, "
                 f"train MSE = {history['train'][-1]:.3f}")
    print(f"H = {H:4d}  parameters = {fit.n_params(net):4d}  "
          f"train MSE = {history['train'][-1]:.3f}")
plt.tight_layout()
plt.savefig("exercise1.png", dpi=120)
plt.show()
