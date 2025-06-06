#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script handling tests.

Created on Fri Jun  6 08:40:50 2025
@author: CoustilletT
"""


import pytest
from pybreathe import BreathingFlow


@pytest.fixture(scope="module")
def sinus():
    """Instantiation of a BreathingFlow representing a sinus."""
    return BreathingFlow.load_sinus()


@pytest.fixture(autouse=True, scope="class")
def _request_sinus(request, sinus):
    request.cls._sinus = sinus


class TestSinus:
    """Test Class for the sinus-type BreathingFlow."""

    def test_sinus_positive_time_should_return_pi(self):
        """The average duration of positive phases should be π."""
        assert self._sinus.get_positive_time()[0] == pytest.approx(3.14)

    def test_sinus_negative_time_should_return_pi(self):
        """The average duration of negative phases should be π."""
        assert self._sinus.get_negative_time()[0] == pytest.approx(3.14)

    def test_sinus_positive_auc_should_return_two(self):
        """The average area of positive phases should be 2."""
        assert self._sinus.get_positive_auc()[0] == pytest.approx(2)

    def test_sinus_negative_auc_should_return_minus_two(self):
        """The average area of negative phases should be - 2."""
        assert self._sinus.get_negative_auc()[0] == pytest.approx(-2)
