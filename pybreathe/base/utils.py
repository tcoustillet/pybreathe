#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Useful functions for managing secondary tasks.

Created on Thu Apr 10 09:48:33 2025
@author: CoustilletT
"""


from importlib.resources import files, as_file
from functools import wraps
from inspect import signature
import os
import pandas as pd

import numpy as np


class ComparableMixin:
    def __eq__(self, other_object, top_level=True):
        """To determine whether two instances are equal.

        Equality is based on a comparison of all instance attributes.

        """
        if not isinstance(other_object, self.__class__):
            return NotImplemented

        diffs = []

        for attr in self.__dict__:
            attr_1 = getattr(self, attr)
            attr_2 = getattr(other_object, attr)

            if isinstance(attr_1, np.ndarray) and isinstance(attr_2, np.ndarray):
                if not np.array_equiv(attr_1, attr_2):
                    diffs.append(attr.lstrip("_"))

            elif isinstance(attr_1, ComparableMixin) and isinstance(attr_2, ComparableMixin):
                if not attr_1.__eq__(attr_2, top_level=False):
                    diffs.append(attr.lstrip("_"))
            else:
                if attr_1 != attr_2:
                    diffs.append(attr.lstrip("_"))

        if diffs and top_level:
            print(f"Different attributes: {', '.join(diffs)}.")
            return False
        return not diffs


# Function (decorator) that handle argument types for object methods.
def enforce_type_arg(**arg_types):
    """Decorate a function to enforce a method argument to be of certain type."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()

            for arg_name, expected_type in arg_types.items():
                if arg_name not in bound_args.arguments:
                    continue

                arg_type = bound_args.arguments[arg_name]
                if not isinstance(arg_type, expected_type):
                    raise TypeError(
                        f"'{arg_name}' argument must be of type "
                        f"'{expected_type.__name__}', not "
                        f"'{type(arg_type).__name__}'."
                    )

            return func(*args, **kwargs)
        return wrapper
    return decorator


def scientific_round(x, n_digits):
    """
    To evenly round x to the given number significant digits.

    Args:
    ----
        x (tuple, float): a float or a tuple of floats.
        n_digits (int): number of significant digits.

    Returns:
    -------
        tuple or float: rounded tuple or float.

    """
    if isinstance(x, (tuple, list, np.ndarray)):
        return tuple(float(f"{element:.{n_digits - 1}e}") for element in x)
    if isinstance(x, float):
        return float(f"{x:.{n_digits - 1}e}")


def flow_merger(*args, table_name, output_directory=None):
    """
    To combine data from several respiratory signals into an unique summary table.

    Args:
    ----
        *args (BreathingFlow): Respiratory air flow rates instantiated
                               and analysed using 'BreathingFlow' object.
        table_name (str): name of the summary table.
        output_directory (str, optional): backup directory for the summary table.
                                          Defaults to None.

    Raises:
    ------
        TypeError: forces all arguments to be of the right type ('BreathingFlow').

    Returns:
    -------
        merged_files (pandas.DataFrame): merged data; 1 row = features for a file.

    """
    from .breathingflow import BreathingFlow

    if not all(isinstance(arg, BreathingFlow) for arg in args):
        raise TypeError("All arguments must be of type 'BreathingFlow'.")

    overview_1f = [arg.get_overview() for arg in args]
    merged_files = pd.concat(overview_1f)

    if output_directory:
        output_path = os.path.join(output_directory, f"overview_{table_name}")
        merged_files.to_excel(excel_writer=f"{output_path}.xlsx")

    return merged_files


def print_source():
    """To print the origin of breathing-like signals."""
    source = files("pybreathe.datasets").joinpath("readme.txt")
    with as_file(source) as f:
        blocs = f.read_text(encoding="utf-8").split("info:")[1:]
        return {
            b.split("\n", 1)[0]: b.split("\n", 1)[1].rstrip() for b in blocs
        }


def _check_type(value, expected_type, name, allow_none=False):
    """To ensure that the type passed to the class constructor is correct."""
    if value is None and allow_none:
        return
    if not isinstance(value, expected_type):
        raise TypeError(
            f"Expected '{name}' to be of type '{expected_type.__name__}', "
            f"got {type(value).__name__}."
        )


def create_absolute_time(time_vector, hz):
    """
    To create an absolute time vector starting from 0.

    Args:
    ----
        time_vector (array): any time vector.
        hz (int): the sampling rate of the time vector.

    Returns:
    -------
        array: time vector in an absolute format: [0.000, 0.002, 0.004, ...].

    """
    return np.linspace(
        0, len(time_vector) / hz, len(time_vector), endpoint=False
    )


def to_dataframe(identifier, overview_dict):
    """
    To convert a dictionary into a specially formatted Dataframe.

    Args:
    ----
        identifier (str): data dictionary identifier.
        overview_dict (dict): dictionary hosting all the signal features.

    Returns:
    -------
        multicols_df (pandas.DataFrame): dataframe summarising the
                                         features (freq, auc, times).

    """
    data_tuples = [
        ((key, sub_key), value)
        for key, sub_dict in overview_dict.items()
        for sub_key, value in sub_dict.items()
    ]
    multicols_df = pd.DataFrame.from_dict(dict(data_tuples), orient="index").T
    multicols_df.columns = pd.MultiIndex.from_tuples(multicols_df.columns)
    multicols_df.index.name = "identifier"
    multicols_df.index = [identifier]

    return multicols_df
