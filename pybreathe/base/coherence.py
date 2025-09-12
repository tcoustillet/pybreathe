#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script hosting functions to assess the synchronisation of respiratory muscles.

Created on Thu Sep 11 08:38:08 2025
@author: CoustilletT
"""


import numpy as np
from scipy.signal import welch, csd

from .visualization import plot_phase_difference


def coherence(movement_1, movement_2, segment_duration, output_path):
    """
    To get the coherence of the two movements: whether they are synchronised or not.

    Args:
    ----
        movement_1 (BreathingMovement): Thorax movements.
        movement_2 (BreathingMovement): Abdominal movements.
        segment_duration (float): the duration over which coherence is calculated.
        output_path (str): to choose where to save the figure, if applicable.

    Returns:
    -------
        None. Plots the figure.

    """
    time = movement_1.absolute_time
    y1 = movement_1.movements
    y2 = movement_2.movements
    hz = movement_1.get_hz()

    segment_samples = int(segment_duration * hz)

    segment_indices = []
    phase_diff = []

    for start in range(0, len(time), segment_samples):
        end = start + segment_samples
        if end > len(time):
            break

        mov_1 = y1[start:end]
        mov_2 = y2[start:end]

        # The two movements are assumed to have the same frequency.
        f, Pxx_den = welch(x=mov_1, fs=hz, nperseg=segment_samples)
        dominant_freq = f[np.argmax(Pxx_den)]

        f, Pxy = csd(x=mov_1, y=mov_2, fs=hz, nperseg=segment_samples)
        phase = np.angle(Pxy)

        freq_idx = np.argmin(np.abs(f - dominant_freq))
        segment_phase = phase[freq_idx]

        segment_indices.append((start+end)/(2*hz))
        phase_diff.append(segment_phase)

    plot_phase_difference(
        time=time,
        y1=movement_1,
        y2=movement_2,
        segment_indices=segment_indices,
        phase_diff=phase_diff,
        output_path=output_path
    )
