#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script hosting functions to assess the synchronisation of respiratory muscles.

Created on Thu Sep 11 08:38:08 2025
@author: CoustilletT
"""


import numpy as np
from numpy import ma
from scipy.signal import welch, csd

from .visualization import plot_phase_difference, PI


def coherence_score(phase_diff, v1, v2=None):
    """
    To quantify the phase difference between movements.

    Args:
    ----
        phase_diff (list): values of phase shifts.
        v1 (float): lower bound for quantifying phase shifts.
        v2 (float, optional): upper bound for quantifying phase shifts.
                              Defaults to None.

    Returns:
    -------
        float: percentage of values within the range of specified differences.

    """
    N = len(phase_diff)
    if v2 is not None:
        score = round(
            (ma.masked_outside(phase_diff, v1, v2).count()
             + ma.masked_outside(phase_diff, -v1, -v2).count()) / N * 100
        )
        return score
    return round(ma.masked_outside(phase_diff, -v1, v1).count() / N * 100)


def coherence(
        movement_1, movement_2, segment_duration, output_path, view, return_vals
 ):
    """
    To get the coherence of the two movements: whether they are synchronised or not.

    Args:
    ----
        movement_1 (BreathingMovement): Thorax movements.
        movement_2 (BreathingMovement): Abdominal movements.
        segment_duration (float): the duration over which coherence is calculated.
        output_path (str): to choose where to save the figure, if applicable.
        view (bool): whether or not to display the figure.
        return_vals (bool): whether or not to return the % spent in each phase.

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

    # Coherence score.
    normal = coherence_score(phase_diff, v1=PI/6)
    disrupted = coherence_score(phase_diff, v1=PI/2, v2=PI/6)
    asynchronous = coherence_score(phase_diff, v1=5*PI/6, v2=PI/2)
    paradoxical = coherence_score(phase_diff, v1=7*PI/6, v2=5*PI/6)

    coherence_overview = {
        "normal": normal,
        "disrupted": disrupted,
        "asynchronous": asynchronous,
        "paradoxical": paradoxical
    }

    if view:
        print("; ".join(f"{k}: {v}%" for k, v in coherence_overview.items()))

    plot_phase_difference(
        time=time,
        y1=movement_1,
        y2=movement_2,
        segment_indices=segment_indices,
        phase_diff=phase_diff,
        output_path=output_path,
        view=view
    )

    if return_vals:
        return coherence_overview
