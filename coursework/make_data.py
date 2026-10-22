"""
make_data.py -- simulated data for the under/overfitting coursework.

Run this once:

    python make_data.py

It writes data.csv with 120 rows of (x, y) and prints nothing else.
The "true" function and the noise level are deliberately not printed;
you will be asked to estimate the noise level from your experiments.
"""
import numpy as np

SEED = 2026          # change this to get a different dataset
N = 120              # total number of points you will be given


def true_function(x):
    # The function the data are generated from. You may read this, but the
    # point of the exercise is to behave as if you could not.
    return np.sin(3.0 * x) + 0.3 * x**2


def make(n, rng, noise=0.3):
    x = rng.uniform(-2.0, 2.0, size=n)
    y = true_function(x) + noise * rng.standard_normal(n)
    return x, y


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    x, y = make(N, rng)
    np.savetxt("data.csv", np.column_stack([x, y]), delimiter=",",
               header="x,y", comments="")
    print(f"wrote data.csv with {N} rows")
