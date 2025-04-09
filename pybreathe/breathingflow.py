#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script handling 'BreathingFlow' object: a discretized air flow rate.

Created on Wed Apr  2 08:40:50 2025
@author: CoustilletT
"""


import numpy as np
import pandas as pd
from scipy.signal import detrend

from .argcontroller import enforce_type_arg
from . import featureextraction as features
from . import visualization


class BreathingFlow:
    """Breathing Air Flow rate."""

    def __init__(self, raw_time, raw_flow):
        self.raw_time = raw_time
        self.raw_flow = raw_flow

        time_len = len(self.raw_time)
        self.absolute_time = (
            np.linspace(0, time_len / self.get_hz(), time_len, endpoint=False)
        )

        self.detrended_flow = detrend(self.raw_flow, type="constant")
        self.detrended_flow[np.isclose(self.detrended_flow, 0, atol=1e-12)] = 0

        self.time, self.flow = features.zero_interpolation(
            self.raw_time, self.detrended_flow
        )

        self._distance = None

    @classmethod
    @enforce_type_arg(filename=str)
    def from_file(cls, filename):
        """
        To instantiate a 'BreathingFlow' objet from a file path.

        Args:
        ----
            filename (str): path to the two-column file representing
                            discretized time and discretized air flow rate.

        Returns:
        -------
            BreathingFlow: instantiate an objet of type 'BreathingFlow'.

        """
        col_names = ["time", "values"]
        time_pattern = r"^[+-]?\d+([.,]\d+)?([eE+][+-]?\d+)?$"

        match filename:
            case filename.endswith("txt"):
                raw_data = pd.read_csv(
                    filename, sep=r"\s+", usecols=[0, 1], names=col_names,
                    dtype=str
                )

        data = raw_data[raw_data["time"].str.match(time_pattern)].reset_index(drop=True)
        data = data.apply(lambda col: col.str.replace(",", ".").astype(float))

        return cls(raw_time=data["time"].values, raw_flow=data["values"].values)

    @classmethod
    def from_dataframe(cls, df):
        """
        To instantiate a 'BreathingFlow' objet from a dataframe.

        Args:
        ----
            df (pandas.DataFrame): two-column dataframe representing discretized
                                   time and discretized air flow rate.

        Returns:
        -------
            BreathingFlow: instantiate an objet of type 'BreathingFlow'.

        """
        if not {"time", "values"}.issubset(df.columns):
            raise ValueError(
                "DataFrame must contain a 'time' column and a 'values' column."
            )
        return cls(raw_time=df["time"].values, raw_flow=df["values"].values)

    @property
    def distance(self):
        """Getter."""
        return self._distance

    def get_hz(self):
        """To get the sampling rate of the discretized breathing signal."""
        return features.compute_sampling_rate(x=self.raw_time)

    @enforce_type_arg(y=str, show_segments=bool, show_auc=bool)
    def plot(self, y="flow", show_segments=False, show_auc=False):
        """To plot the air flow rate."""
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

        visualization.plot_signal(
            x=x, y=y, show_segments=show_segments, show_auc=show_auc
        )

    def get_positive_segments(self):
        """To get the pairs (x,y) for which the air flow rate is positive."""
        return features.get_segments(self.time, self.flow)[0]

    def get_negative_segments(self):
        """To get the pairs (x,y) for which the air flow rate is negative."""
        return features.get_segments(self.time, self.flow)[1]

    @enforce_type_arg(which_peaks=str, distance=int, set_dist=bool)
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
        visualization.plot_peaks(
            x=self.time, y=self.flow, which_peaks=which_peaks, distance=distance
        )

        if set_dist:
            self._distance = distance

    def get_top_peaks(self):
        """To get the top peaks of the air flow rate."""
        return features.get_peaks(
            x=self.time, y=self.flow, which_peaks="top", distance=self.distance
        )

    def get_bottom_peaks(self):
        """To get the bottom peaks of the air flow rate."""
        return features.get_peaks(
            x=self.time, y=self.flow, which_peaks="bottom", distance=self.distance
        )

    @enforce_type_arg(method=str)
    def get_frequency(self, method="welch", which_peaks=None):
        """Get breathing frequency of the air flow rate (in respirations.min-1)."""
        return features.frequency(
            signal=self.flow,
            sampling_rate=self.get_hz(),
            method=method,
            which_peaks=which_peaks,
            distance=self.distance
        )

    @enforce_type_arg(return_mean=bool)
    def get_positive_auc_time(self, return_mean=True):
        """To get the mean duration of positive segments (when AUC > 0).

        Args:
        ----
            return_mean (bool, optional): to return all values or only the mean.
                                          Defaults to True (= the mean).

        Returns:
        -------
            positive_time: mean duration of positive segments (when AUC > 0)
                           (or all durations if return_mean = False).

        Note:
        ----
            AUC = Area Under the Curve.

        """
        return features.get_auc_time(
            segments=self.get_positive_segments(), return_mean=return_mean
        )

    @enforce_type_arg(return_mean=bool)
    def get_negative_auc_time(self, return_mean=True):
        """
        To get the mean duration of negative segments (when AUC < 0).

        Args:
        ----
            return_mean (bool, optional): to return all values or only the mean.
                                          Defaults to True (= the mean).

        Returns:
        -------
            negative_time: mean duration of negative segments (when AUC < 0).
                           (or all durations if return_mean = False).

        Note:
        ----
            AUC = Area Under the Curve.

        """
        return features.get_auc_time(
            segments=self.get_negative_segments(), return_mean=return_mean
        )

    @enforce_type_arg(return_mean=bool)
    def get_positive_auc_value(self, return_mean=True):
        """
        To get the mean AUC of positive segments (when AUC > 0).

        Args:
        ----
            return_mean (bool, optional): to return all values or only the mean.
                                          Defaults to True (= the mean).

        Returns:
        -------
            positive_auc: mean AUC of positive segments (when AUC > 0).
                          (or each AUC of each segment if return_mean = False).

        Note:
        ----
            AUC = Area Under the Curve.

        """
        return features.get_auc_value(
            segments=self.get_positive_segments(), return_mean=return_mean
        )

    @enforce_type_arg(return_mean=bool)
    def get_negative_auc_value(self, return_mean=True):
        """
        To get the mean AUC of negative segments (when AUC < 0).

        Args:
        ----
            return_mean (bool, optional): to return all values or only the mean.
                                          Defaults to True (= the mean).

        Returns:
        -------
            negative_auc: mean AUC of negative segments (when AUC < 0).
                          (or each AUC of each segment if return_mean = False).

        Note:
        ----
            AUC = Area Under the Curve.

        """
        return features.get_auc_value(
            segments=self.get_negative_segments(), return_mean=return_mean
        )
