#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script handling 'BreathingFlow' object: a discretized air flow rate.

Created on Wed Apr  2 08:40:50 2025
@author: CoustilletT
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend

from .signalfeatures import get_segments


class BreathingFlow:
    """Breathing Air Flow rate."""

    def __init__(self, filename):
        raw_data = pd.read_csv(
            filename, sep=r"\s+", usecols=[0, 1], names=["time", "values"],
            dtype=str
        )
        data = raw_data[raw_data["time"].str.match(r"^\d+(,\d+)?$")] \
            .reset_index(drop=True)
        data = data.apply(lambda col: col.str.replace(",", ".").astype(float))

        self.raw_time = data["time"].values
        self.raw_flow = data["values"].values

        time_len = len(self.raw_time)
        self.absolute_time = (
            np.linspace(0, time_len / self.get_hz(), time_len, endpoint=False)
        )

        self.detrended_flow = detrend(self.raw_flow)
        self.__zero_interpolation()

    def get_hz(self):
        """
        To get the sampling rate of the discretized breathing signal.

        Returns:
        -------
            int: the sampling rate in Hz (s-1).

        """
        time_delta = pd.to_timedelta(self.raw_time, unit="s")
        diff = time_delta.diff().value_counts().index.tolist()[0]

        return int(pd.Timedelta(seconds=1) / diff)

    def __zero_interpolation(self):
        """To find the 'True' zeros of a discretized signal."""
        x, y = self.absolute_time, self.detrended_flow

        crossing_indices = np.where(
            np.array(
                [0 if e in (-1, 1) else e for e in np.diff(np.sign(y))]
            )
        )[0]

        x_zeros, y_zeros = [], []
        for i in crossing_indices:
            x_upstream, x_downstream = x[i], x[i + 1]
            y_upstream, y_downstream = y[i], y[i + 1]

            zero = x_upstream - y_upstream * (x_downstream - x_upstream) / (
                y_downstream - y_upstream
            )
            x_zeros.append(zero)
            y_zeros.append(0.0)

        x_extended = np.concatenate((x, x_zeros))
        y_extended = np.concatenate((y, y_zeros))

        sorted_indices = np.argsort(x_extended)
        x_interpolated = x_extended[sorted_indices]
        y_interpolated = y_extended[sorted_indices]

        # Truncation to start and end with a 'True' zero.
        first_zero = np.where(y_interpolated == 0)[0][0]
        lasy_zero = np.where(y_interpolated == 0)[0][-1] + 1

        x_truncated = x_interpolated[first_zero:lasy_zero]
        y_truncated = y_interpolated[first_zero:lasy_zero]

        self.time = x_truncated
        self.flow = y_truncated

    def plot(self, y="flow", show_segments=False):
        """To plot the air flow rate."""
        fig, ax = plt.subplots(figsize=(12, 2))

        match y:
            case "flow":
                x, y = self.time, self.flow
            case "raw_flow":
                x, y = self.absolute_time, self.raw_flow
            case "detrended_flow":
                x, y = self.absolute_time, self.detrended_flow
            case _:
                raise AttributeError(
                    f"{self.__class__.__name__} object has no attribute '{y}'"
                )
                
        if show_segments:
            for i, (x_pos, y_pos) in enumerate(self.get_positive_segments()):
                pos_label = "Air flow rate > 0" if i == 1 else ""
                ax.plot(x_pos, y_pos, label=pos_label, c="tab:blue")
            for i, (x_neg, y_neg) in enumerate(self.get_negative_segments()):
                neg_label = "Air flow rate < 0" if i == 1 else ""
                ax.plot(x_neg, y_neg, label=neg_label, c="tab:orange")
        else:
            ax.plot(x, y, label="air flow rate")

        ax.set_xlabel("time (s)", labelpad=10)
        ax.set_ylabel("Air flow rate", labelpad=10)
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return None

    def get_positive_segments(self):
        """To get the pairs (x,y) for which the air flow rate is positive."""
        return get_segments(self.time, self.flow)[0]

    def get_negative_segments(self):
        """To get the pairs (x,y) for which the air flow rate is negative."""
        return get_segments(self.time, self.flow)[1]
