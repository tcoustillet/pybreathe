#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script handling the instantiation methods for the objects.

Created on Mon Sep  8 10:13:21 2025
@author: CoustilletT
"""


from importlib.resources import files, as_file
import pandas as pd
import re
from .utils import enforce_type_arg, print_source


def process_dataframe(raw_data):
    """
    To process the raw dataframe and properly format it for object instantiation.

    Args:
    ----
        raw_data (pd.DataFrame): Raw dataframe = output of plethysmography software.

    Returns:
    -------
        data (pd.DataFrame): Processed DataFrame = correctly formatted DataFrame.

    """
    time_pattern_1 = r"^\d{2}:\d{2}:\d{2}([.:])\d+$"
    time_pattern_2 = r"^[+-]?\d+([.,]\d+)?([eE+][+-]?\d+)?$"
    time_pattern = f"(?:{time_pattern_1})|(?:{time_pattern_2})"

    data = raw_data[raw_data["time"].str.match(time_pattern, na=False)].reset_index(
        drop=True
    )
    if not data["time"].str.contains(":").any():
        data["time"] = data["time"].str.replace(",", ".")
    if data["values"].str.contains(",").any():
        data["values"] = data["values"].str.replace(",", ".")

    data["values"] = data["values"].astype(float)

    is_wrong_format = data["time"].str.match(r"^\d{2}:\d{2}:\d{2}:\d+$").all()
    if is_wrong_format:
        data["time"] = data["time"].str.replace(r":(?=\d+$)", ".", regex=True)

    # To instantiate an object even if the time vector is not in absolute seconds.
    # Required format : HH:MM:SS.XXX
    if not all(
        re.match(time_pattern_2, time) for time in data["time"].values.astype(str)
    ):
        data["time"] = pd.to_timedelta(data["time"]).dt.total_seconds()

    data["time"] = data["time"].astype(float)

    return data


@enforce_type_arg(identifier=str, filename=str, sheet_name=str, detrend_y=bool, movement_type=str)
def _from_file(cls, identifier, filename, sheet_name="", detrend_y=False, movement_type=""):
    """
    To instantiate a 'BreathingFlow' or 'BreathingMovement' objet from a file path.

    Args:
    ----
        identifier (str): breathing signal identifier.
        filename (str): path to the two-column file representing
                        discretized time and discretized air flow rate/movements.
        sheet_name (str, optional): name of the sheet to be considered.
                                    Defaults to "".
        detrend_y (bool, optional): to set the mean of the air flow rate at 0.
                                    Defaults to False.
        movement_type (str, optional): type of breathing movements (thorax/abdomen),
                                       if any. Defaults to "".

    Returns:
    -------
        BreathingFlow: instantiate an objet of type 'BreathingFlow'.
        OR
        BreathingMovement: instantiate an objet of type 'BreathingMovement'.

    """
    col_names = ["time", "values"]
    match filename:
        case _ if filename.endswith("txt"):
            raw_data = pd.read_csv(
                filename, sep=r"\s+", usecols=[0, 1], names=col_names, dtype=str
            )
        case _ if filename.endswith(("xlsx", "xls")):
            sheet_name = sheet_name if sheet_name else 0
            raw_data = pd.read_excel(
                filename, names=col_names, sheet_name=sheet_name, dtype=str,
                header=None
            )

    data = process_dataframe(raw_data)

    return cls(
            identifier=identifier,
            raw_time=data["time"].values,
            raw_flow=data["values"].values,
            detrend_y=detrend_y,
        )


@enforce_type_arg(identifier=str, detrend_y=bool)
def _from_dataframe(cls, identifier, df, detrend_y=False):
    """
    To instantiate a 'BreathingFlow' objet from a dataframe.

    Args:
    ----
        identifier (str): breathing signal identifier.
        df (pandas.DataFrame): two-column dataframe representing discretized
                               time and discretized air flow rate.
        detrend_y (bool, optional): to set the mean of the air flow rate at 0.
                                    Defaults to False.

    Returns:
    -------
        BreathingFlow: instantiate an objet of type 'BreathingFlow'.

    """
    if not {"time", "values"}.issubset(df.columns):
        raise ValueError(
            "DataFrame must contain a 'time' column and a 'values' column."
        )

    data = process_dataframe(df)

    return cls(
        identifier=identifier,
        raw_time=data["time"].values,
        raw_flow=data["values"].values,
        detrend_y=detrend_y,
    )


def _load_sinus(cls):
    """
    Load and return a BreathingFlow object from the "sinus" dataset.

    Returns:
    -------
        BreathingFlow: a BreathingFlow object = the sinus function.

    Note:
    ----
        is used to demonstrate the 'proof of concept'.
    """
    sinus_resource = files("pybreathe.datasets").joinpath("sinus.txt")

    with as_file(sinus_resource) as sinus_path:
        with open(sinus_path, encoding="utf-8") as f:
            sinus = pd.read_csv(
                f, sep="\t", names=["time", "values"], dtype=float
            )

    return cls(
        identifier="example_sinus",
        raw_time=sinus["time"].values,
        raw_flow=sinus["values"].values,
        detrend_y=False,
    )


def _load_breathing_like_signal_01(cls):
    """
    Load and return a BreathingFlow object from the "breathing-like signal 01" dataset.

    Returns:
    -------
        BreathingFlow: a BreathingFlow object.
    """
    breathing_resource_01 = (
        files("pybreathe.datasets")
        .joinpath("breathing_like_signal_01.txt")
    )

    with as_file(breathing_resource_01) as breathing_01_path:
        with open(breathing_01_path, encoding="utf-8") as f:
            breathing_01 = pd.read_csv(
                f, sep="\t", names=["time", "values"], dtype=float
            )

    print(print_source()["breathing-like 01"])

    return cls(
        identifier="example_breathing-like_signal_01",
        raw_time=breathing_01["time"].values,
        raw_flow=breathing_01["values"].values,
        detrend_y=False,
    )


def _load_breathing_like_signal_02(cls):
    """
    Load and return a BreathingFlow object from the "breathing-like signal 02" dataset.

    Returns:
    -------
        BreathingFlow: a BreathingFlow object.
    """
    breathing_resource_02 = (
        files("pybreathe.datasets")
        .joinpath("breathing_like_signal_02.txt")
    )

    with as_file(breathing_resource_02) as breathing_02_path:
        with open(breathing_02_path, encoding="utf-8") as f:
            breathing_02 = pd.read_csv(
                f, sep="\t", names=["time", "values"], dtype=float
            )

    print(print_source()["breathing-like 02"])

    return cls(
        identifier="example_breathing-like_signal_02",
        raw_time=breathing_02["time"].values,
        raw_flow=breathing_02["values"].values,
        detrend_y=False,
    )
