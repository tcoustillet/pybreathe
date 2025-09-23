#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script handling 'BreathingSignals' object: collection of air flow and movements.

Created on Mon Sep  8 14:26:44 2025
@author: CoustilletT
"""


import numpy as np
import os
import pandas as pd

from .breathingflow import BreathingFlow
from .breathingmovement import BreathingMovement
from .coherence import coherence
from .featureextraction import extract_local_minima
from .utils import _check_type, enforce_type_arg, ComparableMixin, to_dataframe
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

        self.identifier = self.thorax.identifier

        self._thorax_freq = self.thorax.get_frequency()

        self._truncate_movements()

    def _truncate_movements(self):
        """To force breathing movements to start and end with a local minimum."""
        first_min, last_min = extract_local_minima(self.thorax.movements)
        self.thorax = self.thorax[first_min:last_min]
        self.abdomen = self.abdomen[first_min:last_min]

    @enforce_type_arg(shape=str)
    def get_info(self, shape="dict"):
        """To get signal information (identifier, start, end and duration.)"""
        return self.thorax.get_info(shape=shape)

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
            segment_duration = 60 / self._thorax_freq

        return coherence(
            movement_1=self.thorax,
            movement_2=self.abdomen,
            segment_duration=segment_duration,
            output_path=output_path,
            view=view,
            return_vals=return_vals
        )

    @enforce_type_arg(as_dict=bool, output_directory=str)
    def get_overview(self, as_dict=False, output_directory=""):
        """
        To summarize the features of the 'BreathingSignal' object.

        Args:
        ----
            as_dict (bool, optional): whether or not to return the data in
                                      dictionary form. Default to False.
            output_directory (str, optional): where to save the backup file.
                                              It should not be the full path
                                              but just a path to a directory.
                                              Defaults to "" (no backup).

        Returns:
        -------
            pandas.DataFrame: dataframe summarising the features.
            OR
            dict: dictionary summarising the features.

        """
        overview = {}

        if self.flow is not None:
            overview = self.flow.get_overview(as_dict=True)

        overview["movement coherence (%)"] = self.get_coherence(
            view=False, return_vals=True
        )
        formatted_dataframe = to_dataframe(
            identifier=self.identifier, overview_dict=overview
        )

        df_info = self.thorax.get_info(shape="df")

        if output_directory:
            backup_dir = os.path.join(output_directory, self.identifier)
            os.makedirs(backup_dir, exist_ok=True)
            excel_path = os.path.join(
                backup_dir, f"overview_{self.identifier}"
            )
            with pd.ExcelWriter(f"{excel_path}.xlsx", engine="xlsxwriter") as w:
                formatted_dataframe.to_excel(w, sheet_name=f"data_{self.identifier}")
                df_info.to_excel(w, sheet_name=f"info_{self.identifier}", index=False)

            ext = f"_{self.identifier}.pdf"
            self.plot(output_path=os.path.join(backup_dir, f"movements{ext}"))
            self.get_coherence(
                view=False,
                output_path=os.path.join(backup_dir, f"coherence{ext}")
            )

            if self.flow is not None:
                ext = f"_{self.identifier}.pdf"
                self.flow.plot(
                    show_auc=True,
                    output_path=os.path.join(backup_dir, f"flow{ext}")
                )

                self.flow.plot_distribution(
                    output_path=os.path.join(backup_dir, f"feat_distrib{ext}")
                )

                self.flow.plot_phase_portrait(
                    color_scheme="phases",
                    output_path=os.path.join(backup_dir, f"phase_portrait{ext}")
                    )

        if as_dict:
            return overview

        return formatted_dataframe
