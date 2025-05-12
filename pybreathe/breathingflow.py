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
import re
from scipy.signal import detrend

from .utils import enforce_type_arg, scientific_round
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

        # Features
        self._frequency = None
        self._positive_time = None
        self._negative_time = None
        self._positive_auc = None
        self._negative_auc = None
        self._positive_minute_ventilation = None
        self._negative_minute_ventilation = None

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
        time_pattern_1 = r"^\d{2}:\d{2}:\d{2}([.:])\d+$"
        time_pattern_2 = r"^[+-]?\d+([.,]\d+)?([eE+][+-]?\d+)?$"
        time_pattern = f"(?:{time_pattern_1})|(?:{time_pattern_2})"

        match filename:
            case _ if filename.endswith("txt"):
                raw_data = pd.read_csv(
                    filename, sep=r"\s+", usecols=[0, 1], names=col_names,
                    dtype=str
                )
            case _ if filename.endswith(("xlsx", "xls")):
                raw_data = pd.read_excel(filename, names=col_names, dtype=str)

        data = (raw_data[raw_data["time"]
                .str.match(time_pattern, na=False)]
                .reset_index(drop=True)
                )
        if not data["time"].str.contains(":").any():
            data["time"] = data["time"].str.replace(",", ".").astype(float)
        if data["values"].str.contains(",").any():
            data["values"] = data["values"].str.replace(",", ".")

        data["values"] = data["values"].astype(float)

        # To instantiate an object even if the time vector is not in absolute seconds.
        # Required format : HH:MM:SS.XXX
        if not all(re.match(time_pattern_2, time) for time in data["time"].values.astype(str)):
            data["time"] = pd.to_timedelta(data["time"]).dt.total_seconds()

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

    @property
    def frequency(self):
        """Getter."""
        return self._frequency

    @property
    def positive_time(self):
        """Getter."""
        return self._positive_time

    @property
    def negative_time(self):
        """Getter."""
        return self._negative_time

    @property
    def positive_auc(self):
        """Getter."""
        return self._positive_auc

    @property
    def negative_auc(self):
        """Getter."""
        return self._negative_auc

    @property
    def positive_minute_ventilation(self):
        """Getter."""
        return self._positive_minute_ventilation

    @property
    def negative_minute_ventilation(self):
        """Getter."""
        return self._negative_minute_ventilation

    def get_hz(self):
        """To get the sampling rate of the discretized breathing signal."""
        return features.compute_sampling_rate(x=self.raw_time)

    @enforce_type_arg(y=str, show_segments=bool, show_auc=bool,
        highlight_time=tuple, highlight_auc=tuple
    )
    def plot(
            self, y="flow", show_segments=False, show_auc=False,
            highlight_time=(), highlight_auc=()
    ):
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
            x=x, y=y, show_segments=show_segments, show_auc=show_auc,
            highlight_time=highlight_time, highlight_auc=highlight_auc
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

    @enforce_type_arg(method=str, n_digits=int, save=bool)
    def get_frequency(
            self, method="welch", which_peaks=None, n_digits=3, save=False
    ):
        """Get breathing frequency of the air flow rate (in respirations.min-1)."""
        freq = features.frequency(
            signal=self.flow,
            sampling_rate=self.get_hz(),
            method=method,
            which_peaks=which_peaks,
            distance=self.distance,
            n_digits=n_digits
        )

        if save:
            self._frequency = freq

        return freq

    @enforce_type_arg(
        return_mean=bool, verbose=bool, n_digits=int, lower_threshold=float,
        upper_threshold=float, save=bool
    )
    def get_positive_time(
            self, return_mean=True, verbose=True, n_digits=3,
            lower_threshold=-np.inf, upper_threshold=np.inf,
            save=False
    ):
        """To get the mean duration of positive segments (when AUC > 0).

        Args:
        ----
            return_mean (bool, optional): to return all values or only the mean.
                                          Defaults to True (= the mean).
            verbose (bool, optional): to print (or not) results in human
                                      readable format. Defaults to True.
            n_digits (int, optional): to round auc time to n_digits significant
                                      digits. Defaults to 3.
            lower_threshold (float, optional): to ignore values below the threshold.
                                         Defaults to - ∞.
            upper_threshold (float, optional): to ignore values above the threshold.
                                         Defaults to + ∞.
            save (bool, optional): to memorise or not the value obtained.
                                   Defaults to False.

        Returns:
        -------
            positive_time: mean duration of positive segments (when AUC > 0)
                           (or all durations if return_mean = False).

        Note:
        ----
            AUC = Area Under the Curve.

        """
        pos_auc_time = features.get_auc_time(
            segments=self.get_positive_segments(),
            return_mean=return_mean,
            verbose=verbose,
            n_digits=n_digits,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold
        )

        if save:
            self._positive_time = pos_auc_time

        return pos_auc_time

    @enforce_type_arg(
        return_mean=bool, verbose=bool, n_digits=int, lower_threshold=float,
        upper_threshold=float, save=bool
    )
    def get_negative_time(
            self, return_mean=True, verbose=True, n_digits=3,
            lower_threshold=-np.inf, upper_threshold=np.inf,
            save=False
    ):
        """
        To get the mean duration of negative segments (when AUC < 0).

        Args:
        ----
            return_mean (bool, optional): to return all values or only the mean.
                                          Defaults to True (= the mean).
            verbose (bool, optional): to print (or not) results in human
                                      readable format. Defaults to True.
            n_digits (int, optional): to round auc time to n_digits significant
                                      digits. Defaults to 3.
            lower_threshold (float, optional): to ignore values below the threshold.
                                         Defaults to - ∞.
            upper_threshold (float, optional): to ignore values above the threshold.
                                         Defaults to + ∞.
            save (bool, optional): to memorise or not the value obtained.
                                   Defaults to False.

        Returns:
        -------
            negative_time: mean duration of negative segments (when AUC < 0).
                           (or all durations if return_mean = False).

        Note:
        ----
            AUC = Area Under the Curve.

        """
        neg_auc_time = features.get_auc_time(
            segments=self.get_negative_segments(),
            return_mean=return_mean,
            verbose=verbose,
            n_digits=n_digits,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold
        )

        if save:
            self._negative_time = neg_auc_time

        return neg_auc_time

    @enforce_type_arg(
        return_mean=bool, verbose=bool, n_digits=int, lower_threshold=float,
        upper_threshold=float, save=bool
    )
    def get_positive_auc(
            self, return_mean=True, verbose=True, n_digits=3,
            lower_threshold=-np.inf, upper_threshold=np.inf, save=False
    ):
        """
        To get the mean AUC of positive segments (when AUC > 0).

        Args:
        ----
            return_mean (bool, optional): to return all values or only the mean.
                                          Defaults to True (= the mean).
            verbose (bool, optional): to print (or not) results in human
                                      readable format. Defaults to True.
            n_digits (int, optional): to round auc value to n_digits significant
                                      digits. Defaults to 3.
            lower_threshold (float, optional): to ignore values below the threshold.
                                         Defaults to - ∞.
            upper_threshold (float, optional): to ignore values above the threshold.
                                         Defaults to + ∞.
            save (bool, optional): to memorise or not the value obtained.
                                   Defaults to False.

        Returns:
        -------
            positive_auc: mean AUC of positive segments (when AUC > 0).
                          (or each AUC of each segment if return_mean = False).

        Note:
        ----
            AUC = Area Under the Curve.

        """
        pos_auc_val = features.get_auc_value(
            segments=self.get_positive_segments(),
            return_mean=return_mean,
            verbose=verbose,
            n_digits=n_digits,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold
        )

        if save:
            self._positive_auc = pos_auc_val

        return pos_auc_val

    @enforce_type_arg(
        return_mean=bool, verbose=bool, n_digits=int, lower_threshold=float,
        upper_threshold=float, save=bool
    )
    def get_negative_auc(
            self, return_mean=True, verbose=True, n_digits=3,
            lower_threshold=-np.inf, upper_threshold=np.inf, save=False
    ):
        """
        To get the mean AUC of negative segments (when AUC < 0).

        Args:
        ----
            return_mean (bool, optional): to return all values or only the mean.
                                          Defaults to True (= the mean).
            verbose (bool, optional): to print (or not) results in human
                                      readable format. Defaults to True.
            n_digits (int, optional): to round auc value to n_digits significant
                                      digits. Defaults to 3.
            lower_threshold (float, optional): to ignore values below the threshold.
                                         Defaults to - ∞.
            upper_threshold (float, optional): to ignore values above the threshold.
                                         Defaults to + ∞.
            save (bool, optional): to memorise or not the value obtained.
                                   Defaults to False.

        Returns:
        -------
            negative_auc: mean AUC of negative segments (when AUC < 0).
                          (or each AUC of each segment if return_mean = False).

        Note:
        ----
            AUC = Area Under the Curve.
            'threshold' must be negative as AUC is also negative, e.g.,
            if AUC values = [- 0.2, - 0.18, - 0.23, - 0.01], then to remove the
            - 0.01 value, threshold should be - 0.05 for example.

        """
        neg_auc_val = features.get_auc_value(
            segments=self.get_negative_segments(),
            return_mean=return_mean,
            verbose=verbose,
            n_digits=n_digits,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold
        )

        if save:
            self._negative_auc = neg_auc_val

        return neg_auc_val

    @enforce_type_arg(verbose=bool, n_digits=int, save=bool)
    def get_minute_ventilation(self, verbose=True, n_digits=3, save=False):
        """
        To get minute ventilation of positive and negative segments.

        Args:
        ----
            verbose (bool, optional): to print (or not) results in human
                                      readable format. Defaults to True.
            n_digits (int, optional): to round auc value to n_digits significant
                                      digits. Defaults to 3.
            save (bool, optional): to memorise or not the value obtained.
                                   Defaults to False.

        Raises:
        ------
            ValueError: The frequency and tidal volume must have been
                        set by calling up the corresponding methods.

        Returns:
        -------
            pos_mv (float): minute ventilation of positive segments
                            (in unit of tidal volume .min-1).
            neg_mv (float): minute ventilation of negative segments
                            (in unit of tidal volume .min-1).

        Note:
        ----
            Minute Ventilation is the product of frequency and tidal volume.

        """
        if self.frequency and self.positive_auc:
            pos_mv = scientific_round(
                (self.frequency * self.positive_auc[0]), n_digits=n_digits
            )
            if verbose:
                print(f"Minute ventilation of positive segments: {pos_mv}.")
        else:
            raise ValueError(
                "Please save the frequency and tidal volume by setting the "
                "'save' argument to 'True' in the method call. "
                "(get_frequency() and get_positive_auc())"
            )

        if self.frequency and self.negative_auc:
            neg_mv = scientific_round(
                (self.frequency * self.negative_auc[0]), n_digits=n_digits
            )
            if verbose:
                print(f"Minute ventilation of negative segments: {neg_mv}.")
        else:
            raise ValueError(
                "Please save the frequency and tidal volume by setting the "
                "'save' argument to 'True' in the method call. "
                "(get_frequency() and get_negative_auc())"
            )

        if save:
            self._positive_minute_ventilation = pos_mv
            self._negative_minute_ventilation = neg_mv

        return pos_mv, neg_mv

    def plot_distribution(self):
        """To get distribution of each feature of the 'BreathingFlow' object."""
        visualization.plot_features_distribution(
            self.get_positive_time(return_mean=False),
            self.get_negative_time(return_mean=False),
            self.get_positive_auc(return_mean=False),
            self.get_negative_auc(return_mean=False)
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
            "mean": self._frequency, "std": "-", "n cycle(s)": "-"
        }
        dict_data["time (AUC > 0) (s)"] = dict(zip(metrics, self.positive_time))
        dict_data["time (AUC < 0) (s)"] = dict(zip(metrics, self.negative_time))
        dict_data["AUC value (AUC > 0)"] = dict(zip(metrics, self.positive_auc))
        dict_data["AUC value (AUC < 0)"] = dict(zip(metrics, self.negative_auc))
        dict_data["Min. ventilation (AUC > 0)"] = {
            "mean": self._positive_minute_ventilation
        }
        dict_data["Min. ventilation (AUC < 0)"] = {
            "mean": self._negative_minute_ventilation
        }

        def to_dataframe(overview_dict):
            """
            To convert the dictionary into a specially formatted Dataframe.

            Args:
            ----
                overview_dict (dict): dictionary hosting all the signal features.

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

        formatted_dataframe = to_dataframe(overview_dict=dict_data)

        if output_directory:
            output_path = os.path.join(output_directory, f"overview_{self.identifier}")
            formatted_dataframe.to_excel(excel_writer=f"{output_path}.xlsx")

        return formatted_dataframe
