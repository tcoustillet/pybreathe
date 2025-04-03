#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions required to extract signal features.

Created on Thu Apr  3 10:10:59 2025
@author: thibaut
"""


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
    
    return positive_segments, negative_segments
