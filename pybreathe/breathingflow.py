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
from scipy.signal import detrend, find_peaks

from .argcontroller import enforce_bool_arg, enforce_str_arg
from .signalfeatures import get_segments, frequency


class BreathingFlow:
    """Breathing Air Flow rate."""

    def __init__(self, filename):
        raw_data = pd.read_csv(
            filename, sep=r"\s+", usecols=[0, 1], names=["time", "values"],
            dtype=str
        )
        pattern = r"^[+-]?\d+([.,]\d+)?([eE+][+-]?\d+)?$"
        data = raw_data[raw_data["time"].str.match(pattern)].reset_index(drop=True)
        data = data.apply(lambda col: col.str.replace(",", ".").astype(float))

        self.raw_time = data["time"].values
        self.raw_flow = data["values"].values

        time_len = len(self.raw_time)
        self.absolute_time = (
            np.linspace(0, time_len / self.get_hz(), time_len, endpoint=False)
        )

        self.detrended_flow = detrend(self.raw_flow, type="constant")
        self.detrended_flow[np.isclose(self.detrended_flow, 0, atol=1e-12)] = 0
        self.__zero_interpolation()
        self._distance = None

    @property
    def distance(self):
        """Getter."""
        return self._distance

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
        x, y = self.raw_time, self.detrended_flow

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

    @enforce_str_arg("y")
    @enforce_bool_arg("show_segments")
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

    @enforce_bool_arg("set_dist")
    @enforce_str_arg("which_peaks")
    def test_distance(self, which_peaks, distance=0, set_dist=False):
        """
        Calibration of peaks detection
        = test which distance should be assigned to the 'distance' attribute.

        Args:
        ----
            which_peaks (str): to consider either top or bottom peaks.
            distance (int): the minimum distance between two neighbouring peaks.
            set_dist (bool, optionnal): to set the distance or not. Defaults to False.

        Returns:
        -------
            None. Plots a control figure.

        Note:
        ----
            You will probably have to test several distance values
            to find the one that detects all the peaks.

        """
        t, y = self.time, self.flow
        top_peaks, _ = find_peaks(y, distance=distance)
        bottom_peaks, _ = find_peaks(- y, distance=distance)

        fig, ax = plt.subplots(figsize=(12, 2))

        ax.plot(t, y, c="tab:blue", label="your signal", zorder=1)

        match which_peaks:
            case "top":
                ax.scatter(
                    t[top_peaks], y[top_peaks], s=10, marker="x", lw=2, c="red",
                    label=f"{len(top_peaks)} detected peaks (distance = {distance})", zorder=2,
                )
            case "bottom":
                ax.scatter(
                    t[bottom_peaks], y[bottom_peaks], s=10, marker="x", lw=2, c="red",
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

        if set_dist:
            self._distance = distance

        return None

    def get_top_peaks(self):
        """To get the top signal peaks."""
        if not (isinstance(self.distance, int) and self.distance > 0):
            raise ValueError(
                "To get top peaks, distance should be a "
                f"positive integer. Not '{self.distance}'. "
                "Please use 'test_distance method to set the right distance."
            )

        top_peaks, _ = find_peaks(self.flow, distance=self.distance)

        return self.time[top_peaks]

    def get_bottom_peaks(self):
        """To get the bottom signal peaks."""
        if not (isinstance(self.distance, int) and self.distance > 0):
            raise ValueError(
                "To get bottom peaks, distance should be a "
                f"positive integer. Not '{self.distance}'. "
                "Please use 'test_distance method to set the right distance."
            )

        bottom_peaks, _ = find_peaks(- self.flow, distance=self.distance)

        return self.time[bottom_peaks]

    def get_frequency(self, method="welch", which_peaks=None):
        """Get breathing frequency of the air flow rate (in respirations.min-1)."""
        return frequency(
            signal=self.flow,
            sampling_rate=self.get_hz(),
            method=method,
            which_peaks=which_peaks,
            distance=self.distance
        )

    @enforce_bool_arg("return_mean")
    def get_positive_auc_time(self, return_mean=True):
        """
        To get the mean duration of segments when AUC is positive.

        Args:
        ----
            return_mean (bool, optional): to return all values or only the mean.
                                          Defaults to True (= the mean).

        Returns:
        -------
            positive_time: mean duration of segments when AUC is positive
                           (or all durations if return_mean = False).

        Note:
        ----
            AUC = Area Under the Curve.

        """
        positive_points = [ps[0] for ps in self.get_positive_segments()]
        positive_time = [(p[-1] - p[0]) for p in positive_points]

        if return_mean:
            return np.mean(positive_time)
        else:
            return positive_time

    @enforce_bool_arg("return_mean")
    def get_negative_auc_time(self, return_mean=True):
        """
        To get the mean duration of segments when AUC is negative.

        Args:
        ----
            return_mean (bool, optional): to return all values or only the mean.
                                          Defaults to True (= the mean).

        Returns:
        -------
            negative_time: mean duration of segments when AUC is negative
                           (or all durations if return_mean = False).

        Note:
        ----
            AUC = Area Under the Curve.

        """
        negative_points = [ps[0] for ps in self.get_negative_segments()]
        negative_time = [(p[-1] - p[0]) for p in negative_points]

        if return_mean:
            return np.mean(negative_time)
        else:
            return negative_time
