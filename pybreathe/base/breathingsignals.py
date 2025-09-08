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
from .utils import _check_type


class BreathingSignals:
    def __init__(self, flow, thorax, abdomen):
        _check_type(flow, BreathingFlow, "flow")
        _check_type(thorax, BreathingMovement, "thorax")
        _check_type(abdomen, BreathingMovement, "abdomen")

        if not np.array_equal(flow.raw_time, thorax.time):
            raise ValueError(
                "The time vectors of the three signals must match together. "
                "'flow.raw_time' does not match 'thorax.time'."
            )

        if not np.array_equal(thorax.time, abdomen.time):
            raise ValueError(
                "The time vectors of the three signals must match together. "
                "'thorax.raw_time' does not match 'abdomen.time'."
            )

        identifiers = [flow.identifier, thorax.identifier, abdomen.identifier]
        if not (flow.identifier == thorax.identifier == abdomen.identifier):
            raise ValueError(
                f"Cannot combine data with different identifiers: {identifiers}"
            )

        self.flow = flow
        self.thorax = thorax
        self.abdomen = abdomen
