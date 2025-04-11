#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script handling 'BreathingFlow' object: a discretized air flow rate.

Created on Wed Apr  2 08:40:50 2025
@author: CoustilletT
"""


import os

import numpy as np
import pandas as pd
from scipy.signal import detrend

from .utils import enforce_type_arg
from . import featureextraction as features
from . import visualization


class BreathingFlow:
    """Breathing Air Flow rate."""

    @enforce_type_arg(detrend_y=bool)
    def __init__(self, identifier, raw_time, raw_flow, detrend_y=True):
        self.identifier = identifier
        self.raw_time = raw_time
        self.raw_flow = raw_flow

        time_len = len(self.raw_time)
        self.absolute_time = (
            np.linspace(0, time_len / self.get_hz(), time_len, endpoint=False)
        )

        if detrend_y:
            self.detrended_flow = detrend(self.raw_flow, type="constant")
            self.detrended_flow[np.isclose(self.detrended_flow, 0, atol=1e-12)] = 0

        y_to_be_interpolated = getattr(self, "detrended_flow", self.raw_flow)

        self.time, self.flow = features.zero_interpolation(
            x=self.raw_time, y=y_to_be_interpolated
        )

        self._distance = None

    @classmethod
    @enforce_type_arg(filename=str, detrend_y=bool)
    def from_file(cls, identifier, filename, detrend_y=True):
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
            case _ if filename.endswith("txt"):
                raw_data = pd.read_csv(
                    filename, sep=r"\s+", usecols=[0, 1], names=col_names,
                    dtype=str
                )

        data = raw_data[raw_data["time"].str.match(time_pattern)].reset_index(drop=True)
        data = data.apply(lambda col: col.str.replace(",", ".").astype(float))

        return cls(
            identifier=identifier,
            raw_time=data["time"].values,
            raw_flow=data["values"].values,
            detrend_y=detrend_y
        )

    @classmethod
    @enforce_type_arg(detrend_y=bool)
    def from_dataframe(cls, identifier, df, detrend_y=True):
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
        return cls(
            identifier=identifier,
            raw_time=df["time"].values,
            raw_flow=df["values"].values,
            detrend_y=detrend_y
        )

    def __getitem__(self, key):
        """
        To allow a 'BreathingFlow' object to be sliced and used as a new object.

        Args:
        ----
            key (list): slice of shape [start:stop:steps].

        Returns:
        -------
            A new 'BreathingFlow' sliced object.

        """
        sliced_object = self.__class__(
            identifier=self.identifier,
            raw_time=self.time[key],
            raw_flow=self.flow[key],
            detrend_y=False
        )

        if hasattr(self, "distance"):
            sliced_object._distance = self._distance

        return sliced_object

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

    @enforce_type_arg(method=str, decimals=int)
    def get_frequency(self, method="welch", which_peaks=None, decimals=1):
        """Get breathing frequency of the air flow rate (in respirations.min-1)."""
        return features.frequency(
            signal=self.flow,
            sampling_rate=self.get_hz(),
            method=method,
            which_peaks=which_peaks,
            distance=self.distance,
            decimals=decimals
        )


    @enforce_type_arg(return_mean=bool, verbose=bool, decimals=int)
    def get_positive_auc_time(self, return_mean=True, verbose=True, decimals=2):
        """To get the mean duration of positive segments (when AUC > 0).

        Args:
        ----
            return_mean (bool, optional): to return all values or only the mean.
                                          Defaults to True (= the mean).
            verbose (bool, optional): to print (or not) results in human
                                      readable format. Defaults to True.
            decimals (int, optional): to round auc time to the given
                                      number of decimals. Defaults to 2.

        Returns:
        -------
            positive_time: mean duration of positive segments (when AUC > 0)
                           (or all durations if return_mean = False).

        Note:
        ----
            AUC = Area Under the Curve.

        """
        return features.get_auc_time(
            segments=self.get_positive_segments(),
            return_mean=return_mean,
            verbose=verbose,
            decimals=decimals
        )

    @enforce_type_arg(return_mean=bool, verbose=bool, decimals=int)
    def get_negative_auc_time(self, return_mean=True, verbose=True, decimals=2):
        """
        To get the mean duration of negative segments (when AUC < 0).

        Args:
        ----
            return_mean (bool, optional): to return all values or only the mean.
                                          Defaults to True (= the mean).
            verbose (bool, optional): to print (or not) results in human
                                      readable format. Defaults to True.
            decimals (int, optional): to round auc time to the given
                                      number of decimals. Defaults to 2.

        Returns:
        -------
            negative_time: mean duration of negative segments (when AUC < 0).
                           (or all durations if return_mean = False).

        Note:
        ----
            AUC = Area Under the Curve.

        """
        return features.get_auc_time(
            segments=self.get_negative_segments(),
            return_mean=return_mean,
            verbose=verbose,
            decimals=decimals
        )

    @enforce_type_arg(return_mean=bool, verbose=bool, decimals=int)
    def get_positive_auc_value(self, return_mean=True, verbose=True, decimals=2):
        """
        To get the mean AUC of positive segments (when AUC > 0).

        Args:
        ----
            return_mean (bool, optional): to return all values or only the mean.
                                          Defaults to True (= the mean).
            verbose (bool, optional): to print (or not) results in human
                                      readable format. Defaults to True.
            decimals (int, optional): to round auc value to the given
                                      number of decimals. Defaults to 2.

        Returns:
        -------
            positive_auc: mean AUC of positive segments (when AUC > 0).
                          (or each AUC of each segment if return_mean = False).

        Note:
        ----
            AUC = Area Under the Curve.

        """
        return features.get_auc_value(
            segments=self.get_positive_segments(),
            return_mean=return_mean,
            verbose=verbose,
            decimals=decimals
        )

    @enforce_type_arg(return_mean=bool, verbose=bool, decimals=int)
    def get_negative_auc_value(self, return_mean=True, verbose=True, decimals=2):
        """
        To get the mean AUC of negative segments (when AUC < 0).

        Args:
        ----
            return_mean (bool, optional): to return all values or only the mean.
                                          Defaults to True (= the mean).
            verbose (bool, optional): to print (or not) results in human
                                      readable format. Defaults to True.
            decimals (int, optional): to round auc value to the given
                                      number of decimals. Defaults to 2.

        Returns:
        -------
            negative_auc: mean AUC of negative segments (when AUC < 0).
                          (or each AUC of each segment if return_mean = False).

        Note:
        ----
            AUC = Area Under the Curve.

        """
        return features.get_auc_value(
            segments=self.get_negative_segments(),
            return_mean=return_mean,
            verbose=verbose,
            decimals=decimals
        )

    def plot_distribution(self):
        """To get distribution of each feature of the 'BreathingFlow' object."""
        visualization.plot_features_distribution(
            self.get_positive_auc_time(return_mean=False),
            self.get_negative_auc_time(return_mean=False),
            self.get_positive_auc_value(return_mean=False),
            self.get_negative_auc_value(return_mean=False)
        )

    @enforce_type_arg(output_directory=str)
    def get_overview(self, output_directory=""):
        """
        To summarize the features of the 'BreathingFlow' object in a DataFrame.

        Args:
        ----
            output_directory (str, optional): where to save the backup file.
                                              It should not be the full path
                                              but just a path to a directory.
                                              Defaults to "" (no backup).

        Returns:
        -------
            pandas.DataFrame: dataframe summarising the features.

        """
        metrics = ["mean", "std", "n cycle(s)"]
        dict_data = {}
        dict_data["Bf (rpm)"] = {
            "mean": self.get_frequency(), "std": "-", "n cycle(s)": "-"
        }
        dict_data["time (AUC > 0) (s)"] = (
            dict(zip(metrics, self.get_positive_auc_time(verbose=False)))
        )
        dict_data["time (AUC < 0) (s)"] = (
            dict(zip(metrics, self.get_negative_auc_time(verbose=False)))
        )
        dict_data["AUC value (AUC > 0)"] = (
            dict(zip(metrics, self.get_positive_auc_value(verbose=False)))
        )
        dict_data["AUC value (AUC < 0)"] = (
            dict(zip(metrics, self.get_negative_auc_value(verbose=False)))
        )

        def to_dataframe(overview_dict, identifier):
            """
            To convert the dictionary into a specially formatted Dataframe.

            Args:
            ----
                overview_dict (dict): dictionary hosting all the signal features.
                identifier (str): identifier of the 'BreathingFlow' object.

            Returns:
            -------
                multicols_df (pandas.DataFrame): dataframe summarising the
                                                 features (freq, auc, times).

            """
            data_tuples = [
                ((key, sub_key), value) for key, sub_dict in overview_dict.items()
                for sub_key, value in sub_dict.items()
            ]
            multicols_df = pd.DataFrame.from_dict(dict(data_tuples), orient="index").T
            multicols_df.columns = pd.MultiIndex.from_tuples(multicols_df.columns)
            multicols_df.index.name = "identifier"
            multicols_df.index = [self.identifier]

            return multicols_df

        formatted_dataframe = to_dataframe(
            overview_dict=dict_data, identifier=self.identifier
        )

        if output_directory:
            output_path = os.path.join(output_directory, f"overview_{self.identifier}")
            formatted_dataframe.to_excel(excel_writer=f"{output_path}.xlsx")

        return formatted_dataframe
