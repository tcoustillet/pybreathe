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
from .coherence import coherence
from .featureextraction import extract_local_minima
from .utils import _check_type, enforce_type_arg, ComparableMixin
from .visualization import plot_movements


class BreathingSignals(ComparableMixin):
    """Object containing air flow and associated breathing movements."""

    def __init__(self, flow, thorax, abdomen):
        _check_type(flow, BreathingFlow, "flow", allow_none=True)
        _check_type(thorax, BreathingMovement, "thorax")
        _check_type(abdomen, BreathingMovement, "abdomen")

        identifiers = [thorax.identifier, abdomen.identifier]
        if flow is not None:
            identifiers.append(flow.identifier)
        if not all(i == identifiers[0] for i in identifiers):
            raise ValueError(
                f"Cannot combine data with different identifiers: {identifiers}"
            )

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

        self.flow = flow
        self.thorax = thorax
        self.abdomen = abdomen

        self._truncate_movements()

    def _truncate_movements(self):
        """To force breathing movements to start and end with a local minimum."""
        first_min, last_min = extract_local_minima(self.thorax.movements)
        self.thorax = self.thorax[first_min:last_min]
        self.abdomen = self.abdomen[first_min:last_min]

    @enforce_type_arg(shape=str)
    def get_info(self, shape="dict"):
        """To get signal information (identifier, start, end and duration.)"""
        return self.thorax.get_info()

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

    @enforce_type_arg(
        segment_duration=float, output_path=str, view=bool, return_vals=bool
    )
    def get_coherence(
            self, segment_duration=-1.0, output_path="", view=True,
            return_vals=False
    ):
        """To plot the coherence between the breathing movements."""
        # Default value = duration of a respiratory cycle (period)
        if segment_duration == -1.0:
            segment_duration = 60 / self.thorax.get_frequency()

        return coherence(
            movement_1=self.thorax,
            movement_2=self.abdomen,
            segment_duration=segment_duration,
            output_path=output_path,
            view=view,
            return_vals=return_vals
        )
