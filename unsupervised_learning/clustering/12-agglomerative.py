#!/usr/bin/env python3
"""Agglomerative clustering"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Performs agglomerative clustering with Ward linkage."""
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')

    clss = scipy.cluster.hierarchy.fcluster(
        Z,
        t=dist,
        criterion='distance'
    )

    scipy.cluster.hierarchy.dendrogram(
        Z,
        color_threshold=dist
    )

    plt.show()

    return clss
