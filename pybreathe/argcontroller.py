#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions (decorators) that handle argument types for object methods.

Created on Thu Apr  3 11:20:20 2025
@author: thibaut
"""


from functools import wraps


def enforce_bool_arg(arg_name):
    """Decorate a function to enforce a method argument to be of type bool."""
    def decorateur(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if arg_name in kwargs and not isinstance(kwargs[arg_name], bool):
                raise TypeError(
                    f"'{arg_name}' argument must be a booleen (True / False)."
                )
            return func(*args, **kwargs)
        return wrapper
    return decorateur
