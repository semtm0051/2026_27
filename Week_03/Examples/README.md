# Foundations of Artificial Intelligence — Lecture 3

## Supporting data and code

These files reproduce the 100-observation binary classification examples and
the gradient-descent step-size comparison in the Lecture 3 slides. The notes use
separate examples, including a different set of coordinates for the geometric
illustrations and a seven-observation dataset for the worked gradient-descent
calculation.

### Files

| File | Contents |
| --- | --- |
| `Week_03_classification_distinct.csv` | The 100 observations used in the binary classification plots in the slides. |
| `Week_03_GD_step_sizes.csv` | Reference results for iterations 0–20 with each learning rate, including coefficients, loss, mistake count and gradient. |
| `Week_03_compare_step_sizes.py` | A Python 3 script that recalculates both runs from the observation CSV. No additional packages are required. |

If you download the ZIP, extract it before using the files. You can also
download the individual files from this folder. Keep the script and observation
CSV in the same folder. To use the data link on slide 3, also place the PowerPoint
file in that folder and keep the CSV filename unchanged.

## The dataset

Each row is one observation. The columns are:

- `x1`: first numerical input feature.
- `x2`: second numerical input feature.
- `y`: observed class label, either `0` or `1`.

There are 50 observations in each class. All 100 coordinate pairs are distinct.
The inputs are synthetic coordinates without physical units. They were adapted
from an Iris dataset example, with coordinates adjusted to separate the points
in the plots. They should not be interpreted as original flower measurements.
The labels 0 and 1 are generic class labels for this teaching example.

The CSV uses commas as separators and a decimal point. Its coordinates are the
actual inputs to the calculations, with no further jitter, scaling or centring.
The handwritten digits in the multiclass introduction illustrate a separate
classification problem. They are not observations in this CSV.

## Reproducing gradient descent

Run the following command from the folder containing the files:

```text
python Week_03_compare_step_sizes.py
```

Use `python3` instead of `python` if that is the command for Python 3 on your
system. The script prints the coefficients, logistic loss and mistake count
for every iteration of both runs. It does not require an internet connection.

Both runs use the following conventions:

- Parameter order: `theta = (b, w1, w2)^T`.
- Initial coefficients: `theta^(0) = (-5.75, 0, 1)^T`.
- Design matrix: each row of `X` is `(1, x1, x2)`.
- Score: `s = b + w1*x1 + w2*x2`.
- Predicted probability of class 1: `p = 1 / (1 + exp(-s))`.
- Predicted label: class 1 when `s > 0`, otherwise class 0 (including a tie).
- Objective: the **sum** of the 100 logistic losses, using natural logarithms.
- Update: `theta_new = theta - eta * X^T * (p - y)`.
- Learning rates: `eta = 0.001` and `eta = 0.002`.

Every update uses all observations and changes all three coefficients
simultaneously. Do not divide the gradient by 100 with these learning rates:
that would correspond to a different update. Iteration 0 is the initial state,
before any update. Calculations retain full precision between iterations.

### Reference values

| Learning rate | Iteration | Summed logistic loss | Mistakes out of 100 |
| --- | ---: | ---: | ---: |
| Either | 0 | 51.048117 | 31 |
| 0.001 | 1 | 42.732844 | 15 |
| 0.001 | 20 | 39.992016 | 7 |
| 0.002 | 1 | 64.556087 | 42 |
| 0.002 | 20 | 72.281827 | 48 |

In `Week_03_GD_step_sizes.csv`, `loss` is the summed logistic loss and
`mistakes` is a count, not a percentage. The columns `gradient_b`,
`gradient_w1` and `gradient_w2` give the gradient at the coefficients in that
row. It is the gradient used to obtain the next iteration.

## Reading the lecture plots

The plots that compare loss with the number of mistakes display
`logistic loss / ln(2)`, so the theoretical bound can be seen on the same axes.
The printed loss values, tables and the direct comparison of the two learning
rates use the **unscaled summed logistic loss**. Slide values are rounded for
display, so small differences in the final shown digit can arise from rounding.

All reported losses and mistake counts are measured on these same 100 training
observations. They do not measure performance on unseen data.

Two observations to look for:

- With `eta = 0.001`, the loss falls from 41.494823 at iteration 3 to 41.393242
  at iteration 4, while mistakes rise from 7 to 8. Lower logistic loss does
  not guarantee fewer mistakes at each update.
- With `eta = 0.002`, even the first update increases the loss. A step that is
  too large can increase the objective despite following the negative gradient.

## Data attribution

The starting layout uses two classes and two features from:

Fisher, R. (1936). *Iris* [Dataset]. UCI Machine Learning Repository.
[Dataset and citation](https://doi.org/10.24432/C56C76).
The source dataset is distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

For this teaching adaptation, coordinates were adjusted and rounded to three
decimal places, and class labels were recoded as 0 and 1.
