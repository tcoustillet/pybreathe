#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions (decorators) that handle argument types for object methods.

Created on Thu Apr  3 11:20:20 2025
@author: thibaut
"""


from functools import wraps
from inspect import signature


def enforce_type_arg(**arg_types):
    """Decorate a function to enforce a method argument to be of certain type ."""
    def decorateur(func):
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
    return decorateur
