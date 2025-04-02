#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script handling 'BreathingFlow' object: a discretized air flow rate.

Created on Wed Apr  2 08:40:50 2025
@author: CoustilletT
"""


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

        self.time = data["time"].values
        self.flow = data["values"].values
