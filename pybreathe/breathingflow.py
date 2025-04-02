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

        self.absolute_time = data["time"].values
        self.flow = data["values"].values

        time_len = len(self.absolute_time)
        self.time = (
            np.linspace(0, time_len / self.get_hz(), time_len, endpoint=False)
        )

    def get_hz(self):
        """
        To get the sampling rate of the discretized breathing signal.

        Returns:
        -------
            int: the sampling rate in Hz (s-1).

        """
        time_delta = pd.to_timedelta(self.absolute_time, unit="s")
        diff = time_delta.diff().value_counts().index.tolist()[0]

        return int(pd.Timedelta(seconds=1) / diff)

    def plot(self):
        """To plot the air flow rate."""
        fig, ax = plt.subplots(figsize=(12, 2))

        ax.plot(self.time, self.flow, label="air flow rate")
        ax.set_xlabel("time (s)", labelpad=10)
        ax.set_ylabel("Air flow rate", labelpad=10)
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return None
