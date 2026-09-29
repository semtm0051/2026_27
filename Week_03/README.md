# Week 3

## Lab

The Week 3 lab applies PyTorch to linear regression.

### Linear regression

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/semtm0051/2026_27/blob/main/Week_03/Lab/Week_03_linear_regression.ipynb)

## Lecture notes

[Linear classification](Notes/Week_03%20-%20Linear%20Classification.pdf)

## Lecture examples: data and code

The linear classification lecture uses a synthetic dataset of 100 observations
to illustrate decision boundaries and gradient descent with two learning rates.

[Download the data and code package](Examples/Week_03_data_and_code.zip?raw=true)
and extract it before running the script. The package includes a README with
the data description, conventions, source attribution and reproduction instructions.

Individual files:

- [Observation data (CSV)](Examples/Week_03_classification_distinct.csv)
- [Gradient descent results for both learning rates (CSV)](Examples/Week_03_GD_step_sizes.csv)
- [Python reproduction script](Examples/Week_03_compare_step_sizes.py)
- [Instructions and data documentation](Examples/README.md)

The script requires Python 3 and no additional packages. Keep it in the same
folder as the observation CSV, then run:

```text
python Week_03_compare_step_sizes.py
```

These files reproduce the binary classification examples in the lecture slides.
They accompany the lecture, while the lab above covers linear regression.

