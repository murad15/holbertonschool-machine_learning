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

    # 1. Plot the histogram with bins every 10 units and black outlines
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')

    # 2. Set the x-axis and y-axis labels
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')

    # 3. Set the title
    plt.title('Project A')

    # 4. Set the axis limits to match the provided image
    plt.xlim(0, 100)
    plt.ylim(0, 30)

    # 5. Display the plot
    plt.show()
