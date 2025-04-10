#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Useful functions for managing secondary tasks.

Created on Thu Apr 10 09:48:33 2025
@author: CoustilletT
"""


import numpy as np


def scientific_round(x, decimals):
    """
    To evenly round x to the given number of decimals.

    Args:
    ----
        x (tuple, float): a float or a tuple of floats.
        decimals (int): number of decimal places to round to.

    Returns:
    -------
        tuple or float: rounded tuple or float.

    """
    if isinstance(x, (tuple, list, np.ndarray)):
        return tuple(float(f"{element:.{decimals - 1}e}") for element in x)
    elif isinstance(x, float):
        return float(f"{x:.{decimals - 1}e}")
