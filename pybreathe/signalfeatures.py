#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions required to extract signal features.

Created on Thu Apr  3 10:10:59 2025
@author: thibaut
"""


import numpy as np
from scipy.signal import find_peaks, periodogram, welch, windows


def get_segments(x, y):
    """
    To get positive and negative segments of a signal.

    Args:
    ----
        x (array): a discretised time in seconds (0.001, 0.002, 0.003, ...).
        y (array): a discretized breathing air flow rate.

    Returns:
    -------
        positive_segments (array): pairs (x, y) for which y is positive.
        negative_segments (array): pairs (x, y) for which y in negative.

    """
    positive_segments, negative_segments = [], []
    pos_x, pos_y, neg_x, neg_y = [], [], [], []

    for i in range(len(y)):
        if y[i] > 0:
            if len(neg_x) > 0:
                negative_segments.append((neg_x, neg_y))
                neg_x, neg_y = [], []

            pos_x.append(x[i])
            pos_y.append(y[i])

        elif y[i] < 0:
            if len(pos_x) > 0:
                positive_segments.append((pos_x, pos_y))
                pos_x, pos_y = [], []

            neg_x.append(x[i])
            neg_y.append(y[i])

        else:
            if len(pos_x) > 0:
                positive_segments.append((pos_x, pos_y))
                pos_x, pos_y = [], []

            if len(neg_x) > 0:
                negative_segments.append((neg_x, neg_y))
                neg_x, neg_y = [], []

    if len(pos_x) > 0:
        positive_segments.append((pos_x, pos_y))

    if len(neg_x) > 0:
        negative_segments.append((neg_x, neg_y))

    def add_zeros(x, y, segments):
        """
        To make the positive and negative segments begin and end with 0.

        Args:
        ----
            x (array): a discretised time in seconds (0.001, 0.002, 0.003, ...).
            y (array): a discretized breathing air flow rate.
            segments (list): list of arrays where the values of y have always the same sign.

        Returns:
        -------
            segments (array): pairs (x, y) where y begins and ends with a 0.

        """
        for s in segments:
            first_point = s[0][0]
            last_point= s[0][-1]

            first_xzero = x[(np.where(x == first_point)[0][0] - 1)]
            last_xzero = x[(np.where(x == last_point)[0][0] + 1)]

            first_yzero = y[(np.where(x == first_point)[0][0] - 1)]
            last_yzero = y[(np.where(x == last_point)[0][0] + 1)]

            s[0].insert(0, first_xzero)
            s[0].append(last_xzero)
            s[1].insert(0, first_yzero)  # supposed to be 0.
            s[1].append(last_yzero)  # supposed to be 0.

        return segments

    positive_segments = add_zeros(x, y, positive_segments)
    negative_segments = add_zeros(x, y, negative_segments)

    return positive_segments, negative_segments


def frequency(signal, sampling_rate, method, which_peaks, distance):
    """Get the frequency of a given signal.

    Args:
    ----
        signal (array): signal to test (list of numbers = discretized signal).
        sampling_rate (int): the sampling rate of the discretized signal.
        method (str): method to commpute frequency.
        which_peaks (str): if the method is 'peaks', which peaks should be
                           considered (top or bottom) ?
        distance (int): the minimum distance between two neighbouring peaks.

    Returns:
    -------
        dominant_freq (float): breathing frequency (in respirations per min.)

    """
    match method:
        case "welch":
            fft_length = len(signal) // 3
            window = windows.hamming(fft_length)
            f, Pxx_den = welch(
                x=signal, fs=sampling_rate, window=window, nperseg=fft_length
            )

            dominant_freq = f[np.argmax(Pxx_den)]

        case "peaks":
            if not (isinstance(distance, int) and distance > 0):
                raise ValueError(
                    "To use the peak method, distance should be a "
                    f"positive integer. Not '{distance}'."
                    "Please use 'test_distance method to set the right distance."
                )

            if which_peaks == "top":
                peaks, _ = find_peaks(x=signal, distance=distance)
            elif which_peaks == "bottom":
                peaks, _ = find_peaks(x=-signal, distance=distance)
            else:
                raise ValueError(
                    "Argument 'which_peaks' must be either 'top' or 'bottom'. "
                    f"Not '{which_peaks}'."
                )

            dominant_freq = sampling_rate / np.mean(np.diff(peaks))

        case "periodogram":
            window = windows.hamming(len(signal))
            f, Pxx_den = periodogram(x=signal, fs=sampling_rate, window=window)
            dominant_freq = f[np.argmax(Pxx_den)]

    # The frequency is in rpm.s-1; we want it in min.-1.
    dominant_freq *= 60

    return dominant_freq
