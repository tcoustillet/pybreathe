#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions that handle plots.

Created on Wed Apr  9 08:30:59 2025
@author: CoustilletT
"""


import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from . import signalfeatures as sf


def plot_signal(x, y, show_segments):
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
    fig, ax = plt.subplots(figsize=(14, 2))

    if show_segments:
        for i, (x_pos, y_pos) in enumerate(sf.get_segments(x, y)[0]):
            pos_label = "Air flow rate > 0" if i == 1 else ""
            ax.plot(x_pos, y_pos, label=pos_label, c="tab:blue")
        for i, (x_neg, y_neg) in enumerate(sf.get_segments(x, y)[1]):
            neg_label = "Air flow rate < 0" if i == 1 else ""
            ax.plot(x_neg, y_neg, label=neg_label, c="tab:orange")
    else:
        ax.plot(x, y, label="air flow rate")

    ax.set_xlabel("time (s)", labelpad=10)
    ax.set_ylabel("Air flow rate", labelpad=10)
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
