#!/usr/bin/env python3
"""Something that function does"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Check if training should stop early based on validation cost"""

    if opt_cost - cost > threshold:
        # Improvement detected, reset count
        count = 0
        stop = False
    else:
        # No sufficient improvement
        count += 1
        stop = count >= patience

    return stop, count
