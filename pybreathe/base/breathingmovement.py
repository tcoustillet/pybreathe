#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script handling 'BreathingMovement' object: a discretized chest/abdominal movement.

Created on Fri Sep  5 08:12:19 2025
@author: CoustilletT
"""


import pandas as pd
from scipy.signal import detrend

from . import featureextraction as features
from .instantiationmethods import _from_file, _from_dataframe, _process_time
from . import visualization
from .utils import enforce_type_arg, create_absolute_time, ComparableMixin


class BreathingMovement(ComparableMixin):
    """Breathing movement: thorax or abdomen."""

    def __init__(self, time, movements, movement_type, identifier, detrend_y=False):
        self.time = time
        self.movements = movements
        self.movement_type = movement_type
        self.identifier = identifier

        self.processed_time = _process_time(time)

        self.absolute_time = create_absolute_time(self.time, self.get_hz())

        if detrend_y:
            self.movement = detrend(self.raw_movements, type="constant")

    from_file = classmethod(_from_file)
    from_dataframe = classmethod(_from_dataframe)

    def get_hz(self):
        """To get the sampling rate of the discretized breathing signal."""
        return features.compute_sampling_rate(x=self.processed_time)

    @enforce_type_arg(shape=str)
    def get_info(self, shape="dict"):
        """To get signal information (identifier, start, end and duration.)"""
        info = {
            "identifier": self.identifier,
            "start": self.time[0],
            "end": self.time[-1],
            "duration": str(pd.to_timedelta(
                self.absolute_time[-1], unit="s")
            ).split()[-1]
        }

        if shape == "dict":
            return info
        if shape == "df":
            return pd.DataFrame([info])
        raise ValueError(f"shape must be 'dict' or 'df', not '{shape}'.")

    @enforce_type_arg(output_path=str)
    def plot(self, output_path=""):
        visualization.plot_signal(
            x=self.absolute_time,
            y=self.movements,
            show_zeros=False,
            show_segments=False,
            show_auc=False,
            highlight_time=(),
            highlight_auc=(),
            label=f"{self.movement_type} movements",
            output_path=output_path,
        )
