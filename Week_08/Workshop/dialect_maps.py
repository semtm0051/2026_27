"""Small helpers for drawing maps of UK postcode areas.

You do not need to read or change this file for the workshop: just use

    from dialect_maps import plot_map, plot_rgb_map

plot_map(values, ax=None, title=None, cmap=None, diverging=True, labels=False)
    Colour each postcode area by one number, for example a principal
    component score or the percentage of people using a word.
    `values` should be a pandas Series indexed by postcode-area code
    ("BS", "G", "NE", ...). Areas that are not in the Series are drawn grey.

plot_rgb_map(scores, ax=None, title=None)
    Colour each area using three numbers at once, shown as red, green and
    blue. `scores` should be a DataFrame indexed by postcode-area code whose
    first three columns are the three numbers.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "postcode_areas.json")) as f:
    _OUTLINES = {k: [np.array(r) for r in v] for k, v in json.load(f).items()}
_AREAS = pd.read_csv(os.path.join(_HERE, "areas.csv"), index_col="area")

# Areas with no outline in the boundary file are drawn as a circle here.
_NO_OUTLINE = [a for a in _AREAS.index if a not in _OUTLINES]
_GREY = (0.85, 0.85, 0.85, 1.0)


def _draw(ax, colours, title, labels):
    polys, cols = [], []
    for area, rings in _OUTLINES.items():
        for ring in rings:
            polys.append(ring)
            cols.append(colours.get(area, _GREY))
    ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors="white", linewidths=0.4))
    for area in _NO_OUTLINE:
        ax.scatter(_AREAS.lon[area], _AREAS.lat[area], s=90, color=colours.get(area, _GREY),
                   edgecolor="k", linewidth=0.4, zorder=3)
        ax.annotate(area, (_AREAS.lon[area], _AREAS.lat[area]), fontsize=6,
                    xytext=(6, -2), textcoords="offset points")
    if labels:
        for area in _OUTLINES:
            ax.annotate(area, (_AREAS.lon[area], _AREAS.lat[area]), fontsize=5,
                        ha="center", va="center", color="k")
    ax.set_xlim(-8.3, 2.0)
    ax.set_ylim(49.0, 61.0)
    ax.set_aspect(1.7)  # roughly right for the latitude of Britain
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)


def plot_map(values, ax=None, title=None, cmap=None, diverging=True, labels=False, colorbar=True):
    """Colour each postcode area by one number (see the top of this file)."""
    values = pd.Series(values).dropna()
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 6.5))
    if diverging:
        # symmetric about zero, so white means "average"
        m = np.percentile(np.abs(values), 98) or 1.0
        norm = plt.Normalize(-m, m)
        cmap = cmap or "RdBu_r"
    else:
        norm = plt.Normalize(values.min(), np.percentile(values, 98) or 1.0)
        cmap = cmap or "viridis"
    cm = plt.get_cmap(cmap)
    _draw(ax, {a: cm(norm(v)) for a, v in values.items()}, title, labels)
    if colorbar:
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), ax=ax, shrink=0.5, pad=0.02)
    return ax


def plot_rgb_map(scores, ax=None, title=None):
    """Colour each area by three numbers at once, as red, green and blue."""
    scores = pd.DataFrame(scores).iloc[:, :3].dropna()
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 6.5))
    lo, hi = np.percentile(scores.values, 3, axis=0), np.percentile(scores.values, 97, axis=0)
    rgb = np.clip((scores.values - lo) / (hi - lo), 0, 1)
    _draw(ax, {a: tuple(c) for a, c in zip(scores.index, rgb)}, title, False)
    return ax
