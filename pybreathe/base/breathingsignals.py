#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script handling 'BreathingSignals' object: collection of air flow and movements.

Created on Mon Sep  8 14:26:44 2025
@author: CoustilletT
"""


import numpy as np

from .breathingflow import BreathingFlow
from .breathingmovement import BreathingMovement
from .utils import _check_type, enforce_type_arg
from .visualization import plot_movements


class BreathingSignals:
    """Object containing air flow and associated breathing movements."""

    def __init__(self, flow, thorax, abdomen):
        _check_type(flow, BreathingFlow, "flow", allow_none=True)
        _check_type(thorax, BreathingMovement, "thorax")
        _check_type(abdomen, BreathingMovement, "abdomen")

        if flow is not None and not np.array_equal(flow.raw_time, thorax.time):
            raise ValueError(
                "The time vectors of the three signals must match together. "
                "'flow.raw_time' does not match 'thorax.time'."
            )

        if not np.array_equal(thorax.time, abdomen.time):
            raise ValueError(
                "The time vectors of the three signals must match together. "
                "'thorax.raw_time' does not match 'abdomen.time'."
            )

        identifiers = [thorax.identifier, abdomen.identifier]
        if flow is not None:
            identifiers.append(flow.identifier)
        if not all(i == identifiers[0] for i in identifiers):
            raise ValueError(
                f"Cannot combine data with different identifiers: {identifiers}"
            )

        self.flow = flow
        self.thorax = thorax
        self.abdomen = abdomen

    @enforce_type_arg(overlay=bool, output_path=str)
    def plot(self, overlay=False, output_path=""):
        """
        To plot air flow rate and breathing movements on the same plot.

        Args:
        ----
            overlay (bool, optional): whether or not to superimpose respiratory movements.
                                      Defaults to False.

        Returns:
        -------
            None. Plots air flow rate and breathing movements.

        """
        plot_movements(
            y1=self.flow,
            y2=self.thorax,
            y3=self.abdomen,
            overlay=overlay,
            output_path=output_path
        )
