#!/usr/bin/env python3
"""
Module that plots a histogram of student grades for Project A
"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plots a histogram of student grades for Project A

    - X-axis labeled 'Grades'
    - Y-axis labeled 'Number of Students'
    - Bins every 10 units
    - Bars outlined in black
    - Title 'Project A'
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Histogram with bins every 10 units and black edges
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')

    # Labels and title
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.xlim(0, 100)

    # Display plot
    plt.show()
