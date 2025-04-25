#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions that handle plots.

Created on Wed Apr  9 08:30:59 2025
@author: CoustilletT
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
from scipy.integrate import trapezoid
from scipy.signal import find_peaks
import seaborn as sns
from . import featureextraction as features
from .utils import scientific_round


def plot_signal(x, y, show_segments, show_auc, highlight_time, highlight_auc):
    """To plot y versus x.

    Args:
    ----
        x (array): the values for the x-axis.
        y (array): the values for the y-axis.
        show_segments (bool): to distinguish between the positive and negative
                              parts of the curve.

    Returns:
    -------
        None. Plot the figure.

    """
    positive_segments = features.get_segments(x, y)[0]
    negative_segments = features.get_segments(x, y)[1]

    positive_auc = features.get_auc_value(
        segments=positive_segments, return_mean=False,
        verbose=False, n_digits=2, lower_threshold=-np.inf,
        upper_threshold=np.inf
    )

    negative_auc = features.get_auc_value(
        segments=negative_segments, return_mean=False,
        verbose=False, n_digits=2, lower_threshold=-np.inf,
        upper_threshold=np.inf
    )

    pos_normalized = positive_auc / np.max(positive_auc)
    neg_normalized = negative_auc / np.max(negative_auc)

    cmap_pos, cmap_neg = plt.cm.GnBu, plt.cm.RdPu
    cmap_pos_normalized = cmap_pos(np.linspace(
        np.min(pos_normalized), np.max(pos_normalized), len(positive_auc)
    ))
    cmap_neg_normalized = cmap_neg(np.linspace(
        np.min(neg_normalized), np.max(neg_normalized), len(negative_auc)
    ))
    global_cmap = mcolors.ListedColormap(
        np.concatenate([cmap_neg_normalized, cmap_pos_normalized])
    )

    positive_time = features.get_auc_time(
        segments=positive_segments, return_mean=False, verbose=False,
        n_digits=3, lower_threshold=-np.inf, upper_threshold=np.inf
    )
    negative_time = features.get_auc_time(
        segments=negative_segments, return_mean=False, verbose=False,
        n_digits=3, lower_threshold=-np.inf, upper_threshold=np.inf
    )

    fig, ax = plt.subplots(figsize=(14, 2))

    if show_segments:
        for i, (x_pos, y_pos) in enumerate(positive_segments):
            pos_label = "Air flow rate > 0" if i == 1 else ""
            ax.plot(x_pos, y_pos, label=pos_label, c="tab:blue")
        for i, (x_neg, y_neg) in enumerate(negative_segments):
            neg_label = "Air flow rate < 0" if i == 1 else ""
            ax.plot(x_neg, y_neg, label=neg_label, c="tab:orange")
    else:
        ax.plot(x, y, label="air flow rate", c="tab:gray", lw=1)
        ax.axhline(y=0, c="grey", linestyle=":", lw=1)
        zeros = np.where(y == 0)[0]
        ax.scatter(
            x[zeros], y[zeros], zorder=2, c="gold", s=9, lw=0.4, edgecolor="k"
        )

    if show_auc:
        for i, (xp, yp) in enumerate(positive_segments):
            color = cmap_pos(pos_normalized[i])
            ax.fill_between(xp, yp, color=color, alpha=1)

        for j, (xn, yn) in enumerate(negative_segments):
            color = cmap_neg(neg_normalized[j])
            ax.fill_between(xn, yn, color=color, alpha=1)

        sm = ScalarMappable(
            norm=mcolors.TwoSlopeNorm(
                vmin=np.min(negative_auc), vcenter=0, vmax=np.max(positive_auc)
            ),
            cmap=global_cmap
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical")
        cbar.set_label("Area under the curve", labelpad=15)

    if highlight_time:
        for s in (positive_segments, negative_segments):
            cmap = cmap_pos if s == positive_segments else cmap_neg
            normalized = pos_normalized if s == positive_segments else neg_normalized
            for t in highlight_time:
                for i, (xs, ys) in enumerate(s):
                    if scientific_round((xs[-1] - xs[0]), n_digits=3) == t:
                        color = cmap(normalized[i])
                        ax.fill_between(xs, ys, color=color, alpha=1)
                        max_y = max(ys) if max(ys) > 0 else min(ys)
                        ax.text(x=(xs[0] + xs[-1])/2, y=max_y, s=f"t={t}")

    if highlight_auc:
        for s in (positive_segments, negative_segments):
            cmap = cmap_pos if s == positive_segments else cmap_neg
            normalized = pos_normalized if s == positive_segments else neg_normalized
            for a in highlight_auc:
                for i, (xs, ys) in enumerate(s):
                    if scientific_round(trapezoid(y=ys, x=xs), n_digits=3) == a:
                        color = cmap(normalized[i])
                        ax.fill_between(xs, ys, color=color, alpha=1)
                        max_y = max(ys) if max(ys) > 0 else min(ys)
                        ax.text(x=(xs[0] + xs[-1])/2, y=max_y, s=f"auc={a}")

    ax.set_xlabel("time (s)", labelpad=10)
    ax.set_ylabel("Air flow rate", labelpad=10)
    ax.grid(alpha=0.8, linestyle=":", ms=0.1, zorder=1)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_peaks(x, y, which_peaks, distance):
    """
    Calibration of peaks detection
    = test of the distance that correctly detects all the peaks.

    Args:
    ----
        x (array): the values for the x-axis.
        y (array): the values for the y-axis.
        which_peaks (str): to consider either top or bottom peaks.
        distance (int): the minimum distance between two neighbouring peaks.

    Returns:
    -------
        None. Plots a control figure.

    """
    top_peaks, _ = find_peaks(y, distance=distance)
    bottom_peaks, _ = find_peaks(- y, distance=distance)

    fig, ax = plt.subplots(figsize=(14, 2))

    ax.plot(x, y, c="tab:blue", label="your signal", zorder=1)

    match which_peaks:
        case "top":
            ax.scatter(
                x[top_peaks], y[top_peaks], s=10, marker="x", lw=2, c="red",
                label=f"{len(top_peaks)} detected peaks (distance = {distance})",
                zorder=2,
            )
        case "bottom":
            ax.scatter(
                x[bottom_peaks], y[bottom_peaks], s=10, marker="x", lw=2, c="red",
                label=f"{len(bottom_peaks)} detected peaks (distance = {distance})",
                zorder=2
            )
        case _:
            raise ValueError(
                "Argument 'which_peaks' must be either 'top' or 'bottom'. "
                f"Not '{which_peaks}'."
                )

    min_s, max_s = min(y), max(y)
    s_amp = max_s - min_s
    min_b, max_b = min_s - 0.25 * s_amp, max_s + 0.25 * s_amp
    ax.set_ylim(min_b, max_b)
    ax.set_xlabel("time (s)", labelpad=10)
    ax.set_ylabel("Air flow rate", labelpad=10)
    ax.grid(alpha=0.8, linestyle=":", ms=0.5)
    ax.legend(prop={"size": 8}, loc="upper left", bbox_to_anchor=(0, 1.25))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_features_distribution(*args):
    """
    To get the distribution of each feature of the 'BreathingFlow' object.

    Args:
    ----
        *args (array): all the values of one of the signal features.

    Returns:
    -------
        None. Plots the distribution.

    """
    x1, x2, x3, x4 = args
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))

    sns.histplot(data=x1, kde=True, stat="probability", ax=ax1)
    sns.histplot(data=x2, kde=True, stat="probability", ax=ax2)
    sns.histplot(data=x3, kde=True, stat="probability", ax=ax3)
    sns.histplot(data=x4, kde=True, stat="probability", ax=ax4)

    for ax in fig.axes:
        ax.grid(alpha=0.8, linestyle=":", ms=0.5)
        ax.lines[0].set_color("crimson")
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    for ax, lab, x in zip(
        fig.axes,
        ("time (AUC > 0) (s)", "time (AUC < 0) (s)", "AUC > 0", "AUC < 0"),
        args
    ):
        ax.set_xlabel(lab, labelpad=10)
        if lab.endswith("(s)"):
            lab = lab[:-3]
        ax.set_ylabel(fr"$\mathbb{{P}}$ ({lab})", labelpad=10)
        ax.set_title(f"Distribution of {lab} (n = {len(x)})", pad=10)

    fig.tight_layout()
