"""Reproduce Lecture3's step-size comparison, using Python3 without packages.

Run this file next to Week_03_classification_distinct.csv.
It prints all21 states for each learning rate and writes nothing.
"""
import csv
import math
from pathlib import Path

with Path(__file__).with_name('Week_03_classification_distinct.csv').open(
        encoding='utf-8', newline='') as handle:
    observations = [(1.0, float(r['x1']), float(r['x2']), int(r['y']))
                    for r in csv.DictReader(handle)]

print('eta,t,b,w1,w2,loss,mistakes')
for eta in (0.001, 0.002):
    theta = [-5.75, 0.0, 1.0]
    for t in range(21):
        scores = [sum(xj * wj for xj, wj in zip(row[:3], theta))
                  for row in observations]
        probabilities = [1.0 / (1.0 + math.exp(-s)) for s in scores]
        loss = sum(max(s, 0.0) + math.log1p(math.exp(-abs(s))) - row[3] * s
                   for s, row in zip(scores, observations))
        mistakes = sum(int(s >= 0.0) != row[3]
                       for s, row in zip(scores, observations))
        print(f'{eta},{t},{theta[0]:.12f},{theta[1]:.12f},'
              f'{theta[2]:.12f},{loss:.12f},{mistakes}')
        gradient = [sum(row[j] * (p - row[3])
                        for row, p in zip(observations, probabilities))
                    for j in range(3)]
        theta = [coefficient - eta * g
                 for coefficient, g in zip(theta, gradient)]
